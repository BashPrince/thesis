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
SIZES = [50, 100, 200, 400]
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


def parse_filename(filename: str) -> tuple[int, int, int] | None:
    """
    Parse a sequence filename to extract seq_idx, base_size, aug_size.

    Args:
        filename: Filename like 'seq_0_real_100_aug_800.csv'

    Returns:
        Tuple of (seq_idx, base_size, aug_size) or None if not a valid filename
    """
    pattern = r"seq_(\d+)_real_(\d+)_aug_(\d+)\.csv"
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def detect_existing_sequences(base_path: Path = Path('.')) -> dict[int, dict]:
    """
    Detect existing sequence directories and their configurations.

    Args:
        base_path: Path to search for sequence directories

    Returns:
        Dict mapping seq_idx -> {
            'path': Path to sequence directory,
            'base_sizes': set of base sizes found,
            'aug_sizes': set of aug sizes found (excluding 0),
            'files': dict mapping (base_size, aug_size) -> filename
        }
    """
    sequences = {}

    for seq_dir in sorted(base_path.glob('sequence_*')):
        if not seq_dir.is_dir():
            continue

        # Extract sequence index from directory name
        match = re.match(r'sequence_(\d+)', seq_dir.name)
        if not match:
            continue

        seq_idx = int(match.group(1))
        base_sizes = set()
        aug_sizes = set()
        files = {}

        for csv_file in seq_dir.glob('seq_*.csv'):
            parsed = parse_filename(csv_file.name)
            if parsed and parsed[0] == seq_idx:
                _, base_size, aug_size = parsed
                base_sizes.add(base_size)
                if aug_size > 0:
                    aug_sizes.add(aug_size)
                files[(base_size, aug_size)] = csv_file.name

        sequences[seq_idx] = {
            'path': seq_dir,
            'base_sizes': base_sizes,
            'aug_sizes': aug_sizes,
            'files': files,
        }

    return sequences


def check_compatibility(
    existing_sequences: dict[int, dict],
    new_sizes: list[int],
    new_aug_sizes: list[int],
) -> tuple[bool, str]:
    """
    Check if existing sequences are compatible with new size parameters.

    Compatibility requires that existing base_sizes and aug_sizes are subsets
    of the new parameters.

    Args:
        existing_sequences: Dict from detect_existing_sequences
        new_sizes: New SIZES configuration
        new_aug_sizes: New AUG_SIZES configuration

    Returns:
        Tuple of (is_compatible, message)
    """
    new_sizes_set = set(new_sizes)
    new_aug_sizes_set = set(new_aug_sizes)

    for seq_idx, seq_info in existing_sequences.items():
        existing_base = seq_info['base_sizes']
        existing_aug = seq_info['aug_sizes']

        if not existing_base.issubset(new_sizes_set):
            extra = existing_base - new_sizes_set
            return False, f"Sequence {seq_idx} has base sizes {extra} not in new SIZES {new_sizes}"

        if not existing_aug.issubset(new_aug_sizes_set):
            extra = existing_aug - new_aug_sizes_set
            return False, f"Sequence {seq_idx} has aug sizes {extra} not in new AUG_SIZES {new_aug_sizes}"

    return True, "All existing sequences are compatible with new parameters"


def load_synthetic_pool_from_files(
    seq_dir: Path,
    base_set: pd.DataFrame,
    existing_files: dict[tuple[int, int], str],
) -> dict:
    """
    Reconstruct synthetic_pool from existing augmented CSV files.

    Args:
        seq_dir: Path to sequence directory
        base_set: The largest base set (used for template text lookup)
        existing_files: Dict mapping (base_size, aug_size) -> filename

    Returns:
        Dict mapping (template_idx, gen_num) -> sample dict
    """
    synthetic_pool = {}

    # Create a mapping from template text to template index
    text_to_idx = {row['Text']: idx for idx, row in base_set.iterrows()}

    # Track generation numbers per template
    gen_counts = {}

    # Process files in order (smaller aug_sizes first to maintain gen_num consistency)
    sorted_keys = sorted(existing_files.keys(), key=lambda x: (x[0], x[1]))

    for base_size, aug_size in sorted_keys:
        if aug_size == 0:
            continue  # Skip base sets

        filename = existing_files[(base_size, aug_size)]
        filepath = seq_dir / filename

        df = pd.read_csv(filepath)

        # Extract synthetic samples (non-empty 'example' column)
        synthetic_rows = df[df['example'].notna() & (df['example'] != '')]

        for _, row in synthetic_rows.iterrows():
            template_text = row['example']
            if template_text not in text_to_idx:
                continue  # Template not in current base set

            template_idx = text_to_idx[template_text]

            # Check if we already have this sample
            sample_text = row['Text']
            already_exists = any(
                s['Text'] == sample_text and s['example'] == template_text
                for s in synthetic_pool.values()
            )
            if already_exists:
                continue

            gen_num = gen_counts.get(template_idx, 0)
            gen_counts[template_idx] = gen_num + 1

            synthetic_pool[(template_idx, gen_num)] = {
                'Text': sample_text,
                'class_label': row['class_label'],
                'example': template_text,
            }

    return synthetic_pool


def get_tasks_to_generate(
    existing_sequences: dict[int, dict],
    sizes: list[int],
    aug_sizes: list[int],
    num_sequences: int,
) -> list[tuple[int, int, int, bool]]:
    """
    Determine which (seq_idx, base_size, aug_size) combinations need to be generated.

    Args:
        existing_sequences: Dict from detect_existing_sequences
        sizes: SIZES configuration
        aug_sizes: AUG_SIZES configuration
        num_sequences: Number of sequences

    Returns:
        List of (seq_idx, base_size, aug_size, already_exists) tuples
    """
    tasks = []

    for seq_idx in range(num_sequences):
        seq_info = existing_sequences.get(seq_idx, {'files': {}})
        existing_files = seq_info.get('files', {})

        for base_size in sizes:
            for aug_size in aug_sizes:
                already_exists = (base_size, aug_size) in existing_files
                tasks.append((seq_idx, base_size, aug_size, already_exists))

    return tasks


async def main():
    """Main async entry point."""
    # Detect existing sequences
    existing_sequences = detect_existing_sequences()
    extend_mode = False

    if existing_sequences:
        print(f"Found {len(existing_sequences)} existing sequence(s):")
        for seq_idx, info in existing_sequences.items():
            print(f"  sequence_{seq_idx}: base_sizes={sorted(info['base_sizes'])}, aug_sizes={sorted(info['aug_sizes'])}")

        print(f"\nCurrent configuration: SIZES={SIZES}, AUG_SIZES={AUG_SIZES}")

        # Check compatibility
        is_compatible, message = check_compatibility(existing_sequences, SIZES, AUG_SIZES)

        if is_compatible:
            print(f"\n{message}")
            print("\nOptions:")
            print("  [d] Delete existing sequences and start fresh")
            print("  [e] Extend existing sequences with new sizes")
            print("  [a] Abort")
            resp = input("Choose an option [d/e/a]: ").strip().lower()

            if resp == 'd':
                for info in existing_sequences.values():
                    rmtree(info['path'])
                print("Deleted existing sequences.")
            elif resp == 'e':
                extend_mode = True
                print("Extending existing sequences...")
            else:
                print("Aborting.")
                return
        else:
            print(f"\nIncompatible: {message}")
            print("\nOptions:")
            print("  [d] Delete existing sequences and start fresh")
            print("  [a] Abort")
            resp = input("Choose an option [d/a]: ").strip().lower()

            if resp == 'd':
                for info in existing_sequences.values():
                    rmtree(info['path'])
                print("Deleted existing sequences.")
                existing_sequences = {}
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

    # Determine what needs to be generated
    tasks = get_tasks_to_generate(existing_sequences, SIZES, AUG_SIZES, NUM_SEQUENCES)
    tasks_to_run = [(s, b, a) for s, b, a, exists in tasks if not exists]
    tasks_existing = [(s, b, a) for s, b, a, exists in tasks if exists]

    if extend_mode:
        print(f"\nExisting datasets: {len(tasks_existing)}, New to generate: {len(tasks_to_run)}")
    else:
        print(f"\nDatasets to generate: {len(tasks_to_run)}")

    # Overall progress bar for augmentation tasks
    with tqdm(total=len(tasks_to_run), desc="Overall progress", position=0) as pbar_overall:
        # Create sequences
        for seq_idx in range(NUM_SEQUENCES):
            # Create directory for this sequence
            seq_dir = Path(f"sequence_{seq_idx}")
            seq_dir.mkdir(exist_ok=True)

            # For each size, create base set files if they don't exist
            for size in SIZES:
                filename = seq_dir / f"seq_{seq_idx}_real_{size}_aug_0.csv"
                if not filename.exists():
                    base_set = build_base_set(yes_pools[seq_idx], no_pools[seq_idx], size)
                    base_set["example"] = ""
                    base_set.to_csv(filename, index=False)

            # Get the largest base set for this sequence (for loading synthetic pool)
            largest_base_set = build_base_set(yes_pools[seq_idx], no_pools[seq_idx], max(SIZES))

            # Load existing synthetic pool if extending
            if extend_mode and seq_idx in existing_sequences:
                seq_info = existing_sequences[seq_idx]
                synthetic_pool = load_synthetic_pool_from_files(
                    seq_info['path'],
                    largest_base_set,
                    seq_info['files'],
                )
                print(f"\nLoaded {len(synthetic_pool)} existing synthetic samples for sequence {seq_idx}")
            else:
                synthetic_pool = {}

            # Process base sets in order
            for base_size in SIZES:
                base_set = build_base_set(yes_pools[seq_idx], no_pools[seq_idx], base_size)

                # Process aug_sizes in order
                for aug_size in AUG_SIZES:
                    filename = seq_dir / f"seq_{seq_idx}_real_{base_size}_aug_{aug_size}.csv"

                    # Skip if already exists
                    if filename.exists():
                        continue

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
                    augmented_df.to_csv(filename, index=False)

                    pbar_overall.update(1)

    print(f"\nDone! {NUM_SEQUENCES} sequences now have datasets for SIZES={SIZES}, AUG_SIZES={AUG_SIZES}")


if __name__ == "__main__":
    # Set ENV variables
    with open('../../secrets/openai_api_key.txt', 'r') as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()

    asyncio.run(main())
