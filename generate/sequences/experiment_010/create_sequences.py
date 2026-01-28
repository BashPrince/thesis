import pandas as pd
import numpy as np
from pathlib import Path
from shutil import rmtree
from typing import Callable
import re
import os
import asyncio
import random

import litellm
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Configuration
# SIZES = [10, 20]
# AUG_SIZES = [10, 20]
SIZES = [100, 200, 400]
AUG_SIZES = [400, 800]
BATCH_SIZE = 5
NUM_SEQUENCES = 5
RANDOM_SEED = 42

# LLM Configuration
MODEL = "gpt-4o"
MAX_RETRIES = 5
MAX_CONCURRENT_REQUESTS = 5

# Load prompt templates
SCRIPT_DIR = Path(__file__).parent
POS_TEMPLATE = (SCRIPT_DIR / "prompt_templates" / "pos.txt").read_text()
NEG_TEMPLATE = (SCRIPT_DIR / "prompt_templates" / "neg.txt").read_text()


class AugmentationError(Exception):
    """Raised when augmentation fails after all retries."""
    pass


def format_examples(texts: list[str]) -> str:
    """Format a list of texts as bullet points for the prompt."""
    return "\n".join(f"- {text}" for text in texts)


def parse_llm_response(response: str, expected_count: int) -> list[str]:
    """
    Parse the LLM response to extract augmented samples.

    Args:
        response: Raw LLM response text
        expected_count: Number of samples expected

    Returns:
        List of parsed sample texts

    Raises:
        ValueError: If parsing fails or count doesn't match
    """
    # Match lines starting with "- " (with optional leading whitespace)
    pattern = r"^\s*-\s+(.+)$"
    matches = re.findall(pattern, response, re.MULTILINE)

    if len(matches) != expected_count:
        raise ValueError(
            f"Expected {expected_count} samples, got {len(matches)}. "
            f"Response: {response[:500]}..."
        )

    return [match.strip() for match in matches]


def augment_batch_llm(texts: list[str], class_label: str) -> list[str]:
    """
    Augment a batch of texts using an LLM.

    Args:
        texts: List of texts to augment
        class_label: "Yes" for positive, "No" for negative

    Returns:
        List of augmented texts (same length as input)

    Raises:
        AugmentationError: If augmentation fails after all retries
    """
    # Select appropriate template
    template = POS_TEMPLATE if class_label == "Yes" else NEG_TEMPLATE

    # Format the prompt
    examples_str = format_examples(texts)
    prompt = template.format(examples=examples_str)

    # Try with retries
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.choices[0].message.content

            # Parse and validate
            augmented = parse_llm_response(response_text, len(texts))
            return augmented

        except (ValueError, Exception) as e:
            last_error = e
            continue

    raise AugmentationError(
        f"Augmentation failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


async def augment_batch_llm_async(
    texts: list[str],
    class_label: str,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """
    Augment a batch of texts using an LLM (async version).

    Args:
        texts: List of texts to augment
        class_label: "Yes" for positive, "No" for negative
        semaphore: Semaphore to limit concurrent requests

    Returns:
        List of augmented texts (same length as input)

    Raises:
        AugmentationError: If augmentation fails after all retries
    """
    # Select appropriate template
    template = POS_TEMPLATE if class_label == "Yes" else NEG_TEMPLATE

    # Format the prompt
    examples_str = format_examples(texts)
    prompt = template.format(examples=examples_str)

    # Try with retries
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await litellm.acompletion(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
            response_text = response.choices[0].message.content

            # Parse and validate
            augmented = parse_llm_response(response_text, len(texts))
            return augmented

        except (ValueError, Exception) as e:
            last_error = e
            continue

    raise AugmentationError(
        f"Augmentation failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


def augment_batch_identity(texts: list[str], class_label: str) -> list[str]:
    """Placeholder augmenter - returns texts unchanged."""
    return texts


def generate_synthetic_samples(
    base_set: pd.DataFrame,
    aug_size: int,
    synthetic_pool: dict,
    augmenter: Callable[[list[str], str], list[str]],
    batch_size: int,
) -> list[dict]:
    """
    Generate synthetic samples for augmentation with reuse from pool.

    Each template in base_set is used approximately aug_size/base_size times.
    Samples are reused from the pool when available, new ones generated when needed.

    Args:
        base_set: DataFrame with 'Text' and 'class_label' columns
        aug_size: Number of synthetic samples needed
        synthetic_pool: Dict mapping (template_idx, gen_num) -> sample dict
                       Modified in place with newly generated samples
        augmenter: Function that takes list of texts and returns augmented texts
        batch_size: Number of samples per augmentation batch

    Returns:
        List of selected synthetic sample dicts (length = aug_size)
    """
    base_size = len(base_set)
    target_per_template = aug_size // base_size

    # Step 1: For each template, collect existing samples and identify shortfall
    # Group existing pool entries by template_idx
    samples_by_template = {i: [] for i in range(base_size)}
    for (template_idx, gen_num), sample in synthetic_pool.items():
        if template_idx < base_size:
            samples_by_template[template_idx].append((gen_num, sample))

    # Sort each template's samples by gen_num for consistent selection
    for template_idx in samples_by_template:
        samples_by_template[template_idx].sort(key=lambda x: x[0])

    # Select up to target_per_template from each, track what's needed
    selected = []
    templates_needing_more = []  # (template_idx, num_needed)

    for template_idx in range(base_size):
        existing = samples_by_template[template_idx]
        take_count = min(len(existing), target_per_template)

        # Take the first take_count samples (lowest gen_nums)
        for gen_num, sample in existing[:take_count]:
            selected.append((template_idx, gen_num, sample))

        if take_count < target_per_template:
            templates_needing_more.append((template_idx, target_per_template - take_count))

    # Step 2: Generate additional samples for templates that need more
    if templates_needing_more:
        # Track current generation numbers per template
        gen_counts = {}
        for (template_idx, gen_num), _ in synthetic_pool.items():
            gen_counts[template_idx] = max(gen_counts.get(template_idx, 0), gen_num + 1)

        # Split templates needing more by class for class-restricted batching
        yes_templates = [(idx, n) for idx, n in templates_needing_more
                         if base_set.iloc[idx]["class_label"] == "Yes"]
        no_templates = [(idx, n) for idx, n in templates_needing_more
                        if base_set.iloc[idx]["class_label"] == "No"]

        # Process each class
        for class_label, class_templates in [("Yes", yes_templates), ("No", no_templates)]:
            # Track remaining samples needed per template
            remaining = {idx: n for idx, n in class_templates}

            # Process in rounds - each round uses each template at most once
            while any(r > 0 for r in remaining.values()):
                # Get templates that still need samples
                available = [idx for idx, r in remaining.items() if r > 0]
                random.shuffle(available)

                # Process in batches (each batch has unique templates)
                while available:
                    batch_template_indices = available[:batch_size]
                    available = available[batch_size:]

                    # Get texts for batch
                    batch_texts = [base_set.iloc[idx]["Text"] for idx in batch_template_indices]

                    # Augment batch
                    augmented_texts = augmenter(batch_texts, class_label)

                    # Store results and decrement remaining counts
                    for idx, aug_text in zip(batch_template_indices, augmented_texts):
                        gen_num = gen_counts.get(idx, 0)
                        gen_counts[idx] = gen_num + 1
                        remaining[idx] -= 1

                        synthetic_pool[(idx, gen_num)] = {
                            "Text": aug_text,
                            "class_label": base_set.iloc[idx]["class_label"],
                            "example": base_set.iloc[idx]["Text"]
                        }

                        selected.append((idx, gen_num, synthetic_pool[(idx, gen_num)]))

    # Sort selected by (template_idx, gen_num) for consistent output
    selected.sort(key=lambda x: (x[0], x[1]))

    # Return just the sample dicts
    return [s[2] for s in selected[:aug_size]]


async def generate_synthetic_samples_async(
    base_set: pd.DataFrame,
    aug_size: int,
    synthetic_pool: dict,
    semaphore: asyncio.Semaphore,
    batch_size: int,
) -> list[dict]:
    """
    Generate synthetic samples for augmentation with reuse from pool (async version).

    Collects all batches first, then processes them in parallel for efficiency.

    Args:
        base_set: DataFrame with 'Text' and 'class_label' columns
        aug_size: Number of synthetic samples needed
        synthetic_pool: Dict mapping (template_idx, gen_num) -> sample dict
                       Modified in place with newly generated samples
        semaphore: Semaphore to limit concurrent LLM requests
        batch_size: Number of samples per augmentation batch

    Returns:
        List of selected synthetic sample dicts (length = aug_size)
    """
    base_size = len(base_set)
    target_per_template = aug_size // base_size

    # Step 1: For each template, collect existing samples and identify shortfall
    samples_by_template = {i: [] for i in range(base_size)}
    for (template_idx, gen_num), sample in synthetic_pool.items():
        if template_idx < base_size:
            samples_by_template[template_idx].append((gen_num, sample))

    for template_idx in samples_by_template:
        samples_by_template[template_idx].sort(key=lambda x: x[0])

    selected = []
    templates_needing_more = []

    for template_idx in range(base_size):
        existing = samples_by_template[template_idx]
        take_count = min(len(existing), target_per_template)

        for gen_num, sample in existing[:take_count]:
            selected.append((template_idx, gen_num, sample))

        if take_count < target_per_template:
            templates_needing_more.append((template_idx, target_per_template - take_count))

    # Step 2: Collect all batches to generate
    if templates_needing_more:
        gen_counts = {}
        for (template_idx, gen_num), _ in synthetic_pool.items():
            gen_counts[template_idx] = max(gen_counts.get(template_idx, 0), gen_num + 1)

        yes_templates = [(idx, n) for idx, n in templates_needing_more
                         if base_set.iloc[idx]["class_label"] == "Yes"]
        no_templates = [(idx, n) for idx, n in templates_needing_more
                        if base_set.iloc[idx]["class_label"] == "No"]

        # Collect all batches with their metadata
        batches_to_process = []  # (batch_texts, class_label, batch_template_indices)

        for class_label, class_templates in [("Yes", yes_templates), ("No", no_templates)]:
            # Track remaining samples needed per template
            remaining = {idx: n for idx, n in class_templates}

            # Process in rounds - each round uses each template at most once
            while any(r > 0 for r in remaining.values()):
                # Get templates that still need samples
                available = [idx for idx, r in remaining.items() if r > 0]
                random.shuffle(available)

                # Create batches (each batch has unique templates)
                while available:
                    batch_template_indices = available[:batch_size]
                    available = available[batch_size:]

                    # Decrement remaining counts
                    for idx in batch_template_indices:
                        remaining[idx] -= 1

                    batch_texts = [base_set.iloc[idx]["Text"] for idx in batch_template_indices]
                    batches_to_process.append((batch_texts, class_label, batch_template_indices))

        # Step 3: Process all batches in parallel
        async def process_batch(batch_texts, class_label, batch_indices):
            augmented_texts = await augment_batch_llm_async(batch_texts, class_label, semaphore)
            return list(zip(batch_indices, augmented_texts))

        tasks = [
            process_batch(texts, label, indices)
            for texts, label, indices in batches_to_process
        ]

        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Batches (real={base_size}, aug={aug_size})",
            leave=False,
        )

        # Step 4: Update synthetic_pool with results
        for batch_results in results:
            for idx, aug_text in batch_results:
                gen_num = gen_counts.get(idx, 0)
                gen_counts[idx] = gen_num + 1

                synthetic_pool[(idx, gen_num)] = {
                    "Text": aug_text,
                    "class_label": base_set.iloc[idx]["class_label"],
                    "example": base_set.iloc[idx]["Text"]
                }

                selected.append((idx, gen_num, synthetic_pool[(idx, gen_num)]))

    selected.sort(key=lambda x: (x[0], x[1]))
    return [s[2] for s in selected[:aug_size]]


def build_base_set(yes_pool: pd.DataFrame, no_pool: pd.DataFrame, size: int) -> pd.DataFrame:
    """Build a balanced base set from class pools."""
    half_size = size // 2
    subset_yes = yes_pool.iloc[:half_size]
    subset_no = no_pool.iloc[:half_size]
    base_set = pd.concat([subset_yes, subset_no], ignore_index=True)
    if "Sentence_id" in base_set.columns:
        base_set = base_set.drop(columns=["Sentence_id"])
    return base_set


def create_augmented_dataset(
    base_set: pd.DataFrame,
    synthetic_samples: list[dict],
) -> pd.DataFrame:
    """Combine real base set with synthetic samples into augmented dataset."""
    # Build real dataframe with example=NaN
    real_df = base_set.copy()
    real_df["example"] = np.nan

    # Build synthetic dataframe
    synthetic_df = pd.DataFrame(synthetic_samples)

    # Combine real + synthetic
    return pd.concat([real_df, synthetic_df], ignore_index=True)


async def main():
    """Main async entry point."""
    # Check for existing sequence directories
    existing_dirs = [d for d in Path('.').glob('sequence_*') if d.is_dir()]
    if existing_dirs:
        print(f"Found existing sequence directories: {[str(d) for d in existing_dirs]}")
        resp = input("Delete these directories before continuing? [y/N]: ").strip().lower()
        if resp == 'y':
            for d in existing_dirs:
                rmtree(d)
            print("Deleted existing sequence directories.")
        else:
            print("Aborting.")
            return

    # Load the data
    df = pd.read_csv("train.csv")

    # Split by class and shuffle each
    df_yes = df[df["class_label"] == "Yes"].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df_no = df[df["class_label"] == "No"].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Calculate samples needed per class per sequence (half of largest size)
    samples_per_class_per_sequence = max(SIZES) // 2
    total_per_class_needed = samples_per_class_per_sequence * NUM_SEQUENCES

    if len(df_yes) < total_per_class_needed or len(df_no) < total_per_class_needed:
        raise ValueError(
            f"Not enough samples per class. Need {total_per_class_needed} each, "
            f"have Yes: {len(df_yes)}, No: {len(df_no)}"
        )

    # Split into non-overlapping pools for each sequence (per class)
    yes_pools = []
    no_pools = []
    for i in range(NUM_SEQUENCES):
        start_idx = i * samples_per_class_per_sequence
        end_idx = start_idx + samples_per_class_per_sequence
        yes_pools.append(df_yes.iloc[start_idx:end_idx])
        no_pools.append(df_no.iloc[start_idx:end_idx])

    # Create semaphore for limiting concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Calculate total augmentation tasks for progress tracking
    total_aug_tasks = NUM_SEQUENCES * len(SIZES) * len(AUG_SIZES)

    # Overall progress bar for augmentation tasks
    with tqdm(total=total_aug_tasks, desc="Overall progress", position=0) as pbar_overall:
        # Create sequences
        for seq_idx in range(NUM_SEQUENCES):
            # Create directory for this sequence
            seq_dir = Path(f"sequence_{seq_idx}")
            seq_dir.mkdir(exist_ok=True)

            # For each size, create base set files
            for size in SIZES:
                base_set = build_base_set(yes_pools[seq_idx], no_pools[seq_idx], size)
                base_set["example"] = ""
                filename = seq_dir / f"seq_{seq_idx}_real_{size}_aug_0.csv"
                base_set.to_csv(filename, index=False)

            # Augmentation: generate synthetic samples with reuse across base sets and aug_sizes
            synthetic_pool = {}

            # Process base sets in order
            for base_size in SIZES:
                base_set = build_base_set(yes_pools[seq_idx], no_pools[seq_idx], base_size)

                # Process aug_sizes in order
                for aug_size in AUG_SIZES:
                    pbar_overall.set_postfix_str(f"seq={seq_idx}, real={base_size}, aug={aug_size}")

                    synthetic_samples = await generate_synthetic_samples_async(
                        base_set=base_set,
                        aug_size=aug_size,
                        synthetic_pool=synthetic_pool,
                        semaphore=semaphore,
                        batch_size=BATCH_SIZE,
                    )

                    augmented_df = create_augmented_dataset(base_set, synthetic_samples)

                    # Save
                    filename = seq_dir / f"seq_{seq_idx}_real_{base_size}_aug_{aug_size}.csv"
                    augmented_df.to_csv(filename, index=False)

                    pbar_overall.update(1)

    print(f"\nDone! Created {NUM_SEQUENCES} sequences with non-overlapping samples and augmented datasets.")


if __name__ == "__main__":
    # Set ENV variables
    with open('../../secrets/openai_api_key.txt', 'r') as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()

    asyncio.run(main())
