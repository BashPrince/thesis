"""
Genetic Algorithm for Synthetic Data Augmentation.

This module implements a genetic algorithm that generates synthetic text data
by selecting semantically distant pairs and performing LLM-based crossover
and mutation on textual genes.
"""

import os
import json
import random
import asyncio
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import litellm
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from tqdm import tqdm

# Configuration
AUG_SIZES = [128, 256, 512, 1024]
MAX_PARENT_SELECTION = 2
MODEL = "gpt-4o"

# LLM Configuration
MAX_RETRIES = 5
MAX_CONCURRENT_REQUESTS = 5
RANDOM_SEED = 42

# Load prompt templates and genes
SCRIPT_DIR = Path(__file__).parent
POS_TEMPLATE = (SCRIPT_DIR / "prompt_templates" / "pos.txt").read_text()
NEG_TEMPLATE = (SCRIPT_DIR / "prompt_templates" / "neg.txt").read_text()
GENES = json.loads((SCRIPT_DIR / "prompt_templates" / "genes.json").read_text())


class AugmentationError(Exception):
    """Raised when augmentation fails after all retries."""
    pass


def compute_embeddings(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Compute sentence embeddings for a list of texts.

    Args:
        texts: List of text strings
        model: SentenceTransformer model

    Returns:
        Array of embeddings with shape (len(texts), embedding_dim)
    """
    return model.encode(texts, show_progress_bar=False)


def find_most_distant_pair(
    embeddings: np.ndarray,
    excluded_indices: set[int],
) -> Tuple[int, int]:
    """
    Find the most semantically distant pair from available samples.

    Args:
        embeddings: Array of embeddings
        excluded_indices: Set of indices to exclude from selection
                         (e.g., samples that reached max_selections)

    Returns:
        Tuple of (idx_i, idx_j) for the most distant pair

    Raises:
        ValueError: If there are fewer than 2 available samples
    """
    n = len(embeddings)

    if n < 2:
        raise ValueError(
            f"Need at least 2 samples per class to select parents, got {n}. "
            f"Increase BASE_SIZE to at least 4 (2 per class)."
        )

    available = [i for i in range(n) if i not in excluded_indices]

    if len(available) < 2:
        raise ValueError(
            f"Not enough available samples for parent selection. "
            f"Only {len(available)} sample(s) remain after excluding {len(excluded_indices)} "
            f"that reached max_selections. Consider increasing max_selections or BASE_SIZE."
        )

    # Compute pairwise distances for available samples
    available_embeddings = embeddings[available]
    distances = cdist(available_embeddings, available_embeddings, metric='cosine')

    # Find the pair with maximum distance
    max_dist = -1
    best_i, best_j = 0, 1
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            if distances[i, j] > max_dist:
                max_dist = distances[i, j]
                best_i, best_j = i, j

    return available[best_i], available[best_j]


def partition_genes(genes: list[str], rng: random.Random) -> Tuple[list[str], list[str], list[str]]:
    """
    Randomly partition textual genes into three groups: G1, G2, G3.

    G1 contains 3 genes to inherit from parent 1 (crossover).
    G2 contains 3 genes to inherit from parent 2 (crossover).
    G3 contains the remaining genes for mutation (must be different from both parents).

    The template expects:
    - genes[0][0], genes[0][1], genes[0][2] from parent 1
    - genes[1][0], genes[1][1], genes[1][2] from parent 2
    - genes[2] for mutation (formatted string)

    Args:
        genes: List of gene names (must have at least 7)
        rng: Random number generator

    Returns:
        Tuple of (G1, G2, G3) where G1 and G2 are lists of 3 genes, G3 is a list of remaining genes
    """
    shuffled = genes.copy()
    rng.shuffle(shuffled)

    # 3 for G1, 3 for G2, rest for G3
    g1 = shuffled[0:2]   # 3 genes from parent 1
    g2 = shuffled[2:4]   # 3 genes from parent 2
    g3 = shuffled[4:]    # remaining genes for mutation

    return g1, g2, g3


def format_gene_list(genes: list[str]) -> str:
    """
    Format a list of genes as a human-readable string.

    Examples:
        ["tone"] -> "tone"
        ["tone", "subject"] -> "tone and subject"
        ["tone", "subject", "length"] -> "tone, subject and length"

    Args:
        genes: List of gene names

    Returns:
        Formatted string
    """
    if len(genes) == 1:
        return genes[0]
    elif len(genes) == 2:
        return f"{genes[0]} and {genes[1]}"
    else:
        return ", ".join(genes[:-1]) + f" and {genes[-1]}"


def format_prompt(
    claim_1: str,
    claim_2: str,
    g1: list[str],
    g2: list[str],
    g3: list[str],
    class_label: str,
) -> str:
    """
    Format the prompt for LLM crossover and mutation.

    Args:
        claim_1: Text of parent 1
        claim_2: Text of parent 2
        g1: Genes to inherit from parent 1 (list of 3)
        g2: Genes to inherit from parent 2 (list of 3)
        g3: Genes to mutate (must be different from parents)
        class_label: "Yes" for positive, "No" for negative

    Returns:
        Formatted prompt string
    """
    template = POS_TEMPLATE if class_label == "Yes" else NEG_TEMPLATE

    # Format G3 as a human-readable string like "tone, subject, length and certainty"
    g3_formatted = format_gene_list(g3)

    # Build the genes structure for formatting
    # genes[0] = genes from parent 1, genes[1] = genes from parent 2, genes[2] = mutation genes (formatted)
    genes_formatted = {
        'genes': [g1, g2, g3_formatted]
    }

    return template.format(
        claim_1=claim_1,
        claim_2=claim_2,
        **genes_formatted,
    )


class InvalidResponseError(ValueError):
    """Raised when the LLM response has invalid formatting."""
    pass


def parse_llm_response(response: str) -> str:
    """
    Parse the LLM response to extract the generated claim.

    Args:
        response: Raw LLM response text

    Returns:
        The generated claim text

    Raises:
        InvalidResponseError: If the response contains multiple lines
    """
    text = response.strip()

    # Remove quotes if wrapped
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Reject multi-line responses
    lines = [line for line in text.split('\n') if line.strip()]
    if len(lines) > 1:
        raise InvalidResponseError(
            f"Response contains multiple lines ({len(lines)}): {text[:100]}..."
        )

    return text


async def generate_sample_async(
    parent_1: str,
    parent_2: str,
    g1: list[str],
    g2: list[str],
    g3: list[str],
    class_label: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """
    Generate a synthetic sample using LLM-based crossover and mutation.

    Args:
        parent_1: Text of parent 1
        parent_2: Text of parent 2
        g1: Genes to inherit from parent 1
        g2: Genes to inherit from parent 2
        g3: Genes to mutate
        class_label: "Yes" or "No"
        semaphore: Semaphore for rate limiting

    Returns:
        Generated sample text

    Raises:
        AugmentationError: If generation fails after retries
    """
    prompt = format_prompt(parent_1, parent_2, g1, g2, g3, class_label)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await litellm.acompletion(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
            response_text = response.choices[0].message.content
            return parse_llm_response(response_text)
        except Exception as e:
            last_error = e
            continue

    raise AugmentationError(
        f"Generation failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


class GeneticAugmenter:
    """
    Genetic algorithm-based augmenter for text data.

    Maintains a population pool with embeddings and generates new samples
    by selecting distant pairs and performing crossover/mutation.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        rng: random.Random | None = None,
        max_selections: int | None = None,
    ):
        """
        Initialize the augmenter.

        Args:
            embedding_model: SentenceTransformer for computing embeddings
            rng: Random number generator (uses default if None)
            max_selections: Maximum times a sample can be selected as parent.
                           None means no limit.
        """
        self.embedding_model = embedding_model
        self.rng = rng or random.Random()
        self.max_selections = max_selections

        # Population pools per class: {class_label: {'texts': [], 'embeddings': ndarray}}
        self.pools: dict[str, dict] = {}

        # Track which pairs have been used
        self.used_pairs: dict[str, set[tuple[int, int]]] = {}

        # Track selection statistics per class
        # selected_count[class_label][idx] = number of times sample was selected as parent
        self.selected_count: dict[str, dict[int, int]] = {}
        # entry_iteration[class_label][idx] = iteration when sample entered pool
        self.entry_iteration: dict[str, dict[int, int]] = {}
        # Current iteration per class
        self.current_iteration: dict[str, int] = {}

    def initialize_population(self, df: pd.DataFrame) -> None:
        """
        Initialize the population from a DataFrame.

        Args:
            df: DataFrame with 'Text' and 'class_label' columns
        """
        for class_label in ["Yes", "No"]:
            class_df = df[df["class_label"] == class_label]
            texts = class_df["Text"].tolist()
            embeddings = compute_embeddings(texts, self.embedding_model)

            self.pools[class_label] = {
                'texts': texts,
                'embeddings': embeddings,
            }
            self.used_pairs[class_label] = set()

            # Initialize statistics for real samples (iteration 0)
            self.selected_count[class_label] = {i: 0 for i in range(len(texts))}
            self.entry_iteration[class_label] = {i: 0 for i in range(len(texts))}
            self.current_iteration[class_label] = 0

    def add_to_pool(self, text: str, class_label: str) -> None:
        """
        Add a new sample to the population pool.

        Args:
            text: The sample text
            class_label: "Yes" or "No"
        """
        pool = self.pools[class_label]
        new_embedding = compute_embeddings([text], self.embedding_model)

        new_idx = len(pool['texts'])
        pool['texts'].append(text)
        pool['embeddings'] = np.vstack([pool['embeddings'], new_embedding])

        # Track statistics for the new sample
        self.selected_count[class_label][new_idx] = 0
        self.entry_iteration[class_label][new_idx] = self.current_iteration[class_label]

    def select_parents(self, class_label: str) -> Tuple[str, str, int, int]:
        """
        Select the most distant pair from unused samples as parents.

        Args:
            class_label: "Yes" or "No"

        Returns:
            Tuple of (parent_1_text, parent_2_text, idx_1, idx_2)
        """
        pool = self.pools[class_label]
        embeddings = pool['embeddings']
        used = self.used_pairs[class_label]

        # Compute excluded indices (samples that have reached max_selections)
        excluded = set()
        if self.max_selections is not None:
            for idx, count in self.selected_count[class_label].items():
                if count >= self.max_selections:
                    excluded.add(idx)

        idx_i, idx_j = find_most_distant_pair(embeddings, excluded)

        # Mark this pair as used (in both orders)
        used.add((idx_i, idx_j))
        used.add((idx_j, idx_i))

        # Track selection count
        self.selected_count[class_label][idx_i] += 1
        self.selected_count[class_label][idx_j] += 1

        return pool['texts'][idx_i], pool['texts'][idx_j], idx_i, idx_j

    async def generate_samples_for_class(
        self,
        class_label: str,
        target_count: int,
        semaphore: asyncio.Semaphore,
    ) -> list[str]:
        """
        Generate synthetic samples for a specific class.

        Samples are generated sequentially so each new sample is added to the
        pool before selecting parents for the next sample (per the genetic algorithm).

        Args:
            class_label: "Yes" or "No"
            target_count: Number of samples to generate
            semaphore: Semaphore for rate limiting

        Returns:
            List of generated sample texts
        """
        generated = []

        pbar = tqdm(
            range(target_count),
            desc=f"Generating {class_label} samples",
            leave=False,
        )
        for i in pbar:
            # Increment iteration counter before generating new sample
            self.current_iteration[class_label] += 1

            # Select parents (may include previously generated synthetic samples)
            parent_1, parent_2, _, _ = self.select_parents(class_label)

            # Partition genes
            g1, g2, g3 = partition_genes(GENES, self.rng)

            # Generate sample
            text = await generate_sample_async(
                parent_1, parent_2, g1, g2, g3, class_label, semaphore
            )

            # Add to pool immediately so it can be used as parent for next sample
            generated.append(text)
            self.add_to_pool(text, class_label)

        return generated

    def get_statistics(self, class_label: str) -> Tuple[list[int], list[int]]:
        """
        Get selection statistics for all samples in a class pool.

        Args:
            class_label: "Yes" or "No"

        Returns:
            Tuple of (selected_counts, iteration_counts) where:
            - selected_counts[i] = number of times sample i was selected as parent
            - iteration_counts[i] = total_iterations - entry_iteration (how many
              iterations the sample could have been picked)
        """
        total_iterations = self.current_iteration[class_label]
        n_samples = len(self.pools[class_label]['texts'])

        selected_counts = [self.selected_count[class_label][i] for i in range(n_samples)]
        iteration_counts = [
            total_iterations - self.entry_iteration[class_label][i]
            for i in range(n_samples)
        ]

        return selected_counts, iteration_counts



def create_augmented_dataset(
    base_set: pd.DataFrame,
    synthetic_yes: list[str],
    synthetic_no: list[str],
    stats_yes: Tuple[list[int], list[int]],
    stats_no: Tuple[list[int], list[int]],
) -> pd.DataFrame:
    """
    Combine base set with synthetic samples and selection statistics.

    Args:
        base_set: Real data DataFrame (ordered as Yes rows first, then No rows)
        synthetic_yes: List of synthetic positive samples
        synthetic_no: List of synthetic negative samples
        stats_yes: Tuple of (selected_counts, iteration_counts) for Yes class pool
        stats_no: Tuple of (selected_counts, iteration_counts) for No class pool

    Returns:
        Combined DataFrame with 'Text', 'class_label', 'selected_count', and
        'iteration_count' columns
    """
    n_base_yes = len(base_set[base_set["class_label"] == "Yes"])
    n_base_no = len(base_set[base_set["class_label"] == "No"])

    # Unpack statistics
    selected_yes, iteration_yes = stats_yes
    selected_no, iteration_no = stats_no

    # Build statistics in order matching final DataFrame:
    # [base_yes, base_no, syn_yes, syn_no]
    selected_counts = (
        selected_yes[:n_base_yes] +
        selected_no[:n_base_no] +
        selected_yes[n_base_yes:] +
        selected_no[n_base_no:]
    )
    iteration_counts = (
        iteration_yes[:n_base_yes] +
        iteration_no[:n_base_no] +
        iteration_yes[n_base_yes:] +
        iteration_no[n_base_no:]
    )

    # Create synthetic DataFrames
    yes_df = pd.DataFrame({
        'Text': synthetic_yes,
        'class_label': ['Yes'] * len(synthetic_yes),
    })
    no_df = pd.DataFrame({
        'Text': synthetic_no,
        'class_label': ['No'] * len(synthetic_no),
    })

    # Combine all
    combined = pd.concat([base_set, yes_df, no_df], ignore_index=True)

    # Add statistics columns
    combined['selected_count'] = selected_counts
    combined['iteration_count'] = iteration_counts

    return combined


async def run_augmentation(
    base_set: pd.DataFrame,
    seq_idx: int,
    aug_size: int,
    output_dir: Path,
    embedding_model: SentenceTransformer,
    semaphore: asyncio.Semaphore,
) -> Path:
    """
    Run the genetic augmentation for a single configuration.

    Args:
        base_set: Pre-built base set DataFrame with 'Text' and 'class_label' columns
        seq_idx: Sequence index (for filename)
        aug_size: Number of synthetic samples to generate
        output_dir: Directory to save output
        embedding_model: SentenceTransformer model
        semaphore: Semaphore for rate limiting

    Returns:
        Path to the saved file
    """
    rng = random.Random(RANDOM_SEED + seq_idx)

    # Initialize augmenter
    augmenter = GeneticAugmenter(embedding_model, rng, max_selections=MAX_PARENT_SELECTION)
    augmenter.initialize_population(base_set)

    # Generate synthetic samples (balanced)
    samples_per_class = aug_size // 2

    synthetic_yes = await augmenter.generate_samples_for_class(
        "Yes", samples_per_class, semaphore
    )
    synthetic_no = await augmenter.generate_samples_for_class(
        "No", samples_per_class, semaphore
    )

    # Get selection statistics
    stats_yes = augmenter.get_statistics("Yes")
    stats_no = augmenter.get_statistics("No")

    # Create augmented dataset
    augmented_df = create_augmented_dataset(
        base_set, synthetic_yes, synthetic_no, stats_yes, stats_no
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"seq_{seq_idx}_aug_{aug_size}.csv"
    augmented_df.to_csv(filename, index=False)

    return filename


async def main():
    """Main entry point."""
    # Set ENV variables
    with open('../../secrets/openai_api_key.txt', 'r') as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()

    # Discover base set files from unrestricted_wrup
    unrestricted_dir = SCRIPT_DIR.parent / "unrestricted_wrup"
    base_files = sorted(unrestricted_dir.glob("sequence_*/seq_*_aug_0.csv"))
    if not base_files:
        print(f"No base set files found in {unrestricted_dir}")
        return

    # Load embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Run augmentation for each configuration
    output_dir = Path("./output")

    # Check if output directory has existing files
    if output_dir.exists():
        existing_files = list(output_dir.glob("*.csv"))
        if existing_files:
            print(f"\nFound {len(existing_files)} existing file(s) in {output_dir}/")
            for f in existing_files[:5]:
                print(f"  - {f.name}")
            if len(existing_files) > 5:
                print(f"  ... and {len(existing_files) - 5} more")

            while True:
                response = input("\n[d]elete existing files and continue, or [a]bort? ").strip().lower()
                if response == 'd':
                    for f in existing_files:
                        f.unlink()
                    print(f"Deleted {len(existing_files)} file(s).")
                    break
                elif response == 'a':
                    print("Aborted.")
                    return
                else:
                    print("Please enter 'd' or 'a'.")

    # Build list of all configurations to run
    configurations = [
        (seq_idx, base_file, aug_size)
        for seq_idx, base_file in enumerate(base_files)
        for aug_size in AUG_SIZES
    ]

    print(f"\nGenerating {len(configurations)} augmented datasets in parallel...")
    print(f"Base files: {[f.name for f in base_files]}")
    print(f"AUG_SIZES: {AUG_SIZES}")

    # Create tasks for all configurations
    async def run_config(seq_idx: int, base_file: Path, aug_size: int) -> Path:
        """Run a single configuration and print progress."""
        print(f"Starting: {base_file.name}, aug={aug_size}")
        base_set = pd.read_csv(base_file)
        filepath = await run_augmentation(
            base_set=base_set,
            seq_idx=seq_idx,
            aug_size=aug_size,
            output_dir=output_dir,
            embedding_model=embedding_model,
            semaphore=semaphore,
        )
        print(f"Saved: {filepath}")
        return filepath

    # Run all configurations in parallel (semaphore controls LLM rate limiting)
    tasks = [run_config(si, bf, aus) for si, bf, aus in configurations]
    await asyncio.gather(*tasks)

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
