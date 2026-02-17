# Experiment 010: LLM-Augmented Sequence Generation

## Purpose

Generate augmented training datasets for a check-worthy claim detection classifier. The experiment creates multiple sequences of nested datasets with varying amounts of real and synthetic (LLM-generated) samples to study the effect of data augmentation on model performance.

## Task

Binary classification: determine if a statement is "check-worthy" (Yes) or not (No). Check-worthy claims are verifiable factual statements of public interest that should be verified by fact-checkers.

## Data Structure

Each sequence contains datasets with:
- **Real samples**: Actual examples from the training set
- **Synthetic samples**: LLM-generated paraphrases/variations of real samples

Files are named: `seq_{i}_real_{base_size}_aug_{aug_size}.csv`
- `base_size`: Number of real samples (100, 200, 400, 800, 1600)
- `aug_size`: Number of synthetic samples (0, 800, 1600)

## Key Design Decisions

### Nested Base Sets
Smaller base sets are strict subsets of larger ones (e.g., the 100 samples in `real_100` are the first 100 of `real_200`). This ensures consistent comparison across dataset sizes.

### Synthetic Sample Reuse
Synthetic samples are reused across configurations when possible:
- Samples generated for `real_100_aug_800` are reused in `real_100_aug_1600`
- When base size increases, new templates get new synthetic samples while existing templates reuse their samples

### Template Distribution
Each template (real sample) is used equally as a source for synthetic samples:
- `real_100_aug_800`: each of 100 templates → 8 synthetic samples
- `real_200_aug_800`: each of 200 templates → 4 synthetic samples

### Class-Restricted Batching
LLM augmentation batches contain only positive OR negative samples, never mixed. This allows the prompt to be tailored for each class. Each batch also contains unique templates (no duplicates).

## Files

- `create_sequences.py` - Main script for generating sequences
- `test_augmentation.py` - Tests for the augmentation logic
- `prompt_templates/pos.txt` - Prompt template for positive (check-worthy) samples
- `prompt_templates/neg.txt` - Prompt template for negative samples
- `train.csv` - Source training data

## Configuration

In `create_sequences.py`:
```python
SIZES = [100, 200, 400, 800, 1600]  # Base set sizes
AUG_SIZES = [800, 1600]             # Synthetic sample counts
BATCH_SIZE = 5                       # Samples per LLM call
NUM_SEQUENCES = 5                    # Independent sequences
MODEL = "gpt-4o"                     # LLM for augmentation
MAX_CONCURRENT_REQUESTS = 10         # Parallel API calls
```

## Running

```bash
# Run tests
python -m pytest test_augmentation.py -v

# Generate sequences (requires OpenAI API key in ../../secrets/openai_api_key.txt)
python create_sequences.py
```

## Extending Existing Sequences

The script supports extending existing sequences when configuration changes:

1. **Compatibility Check**: When existing sequences are found, the script checks if their sizes are subsets of the new `SIZES` and `AUG_SIZES` configuration.

2. **Options when compatible**:
   - `[d]` Delete and start fresh
   - `[e]` Extend existing sequences (generate only missing datasets)
   - `[a]` Abort

3. **Options when incompatible**:
   - `[d]` Delete and start fresh
   - `[a]` Abort

When extending, the script:
- Detects existing sequence files and their configurations
- Reconstructs the synthetic pool from saved augmented files
- Generates only the missing (base_size, aug_size) combinations
- Maintains consistency with the reuse scheme

Example: Changing from `SIZES=[100,200,400]` to `SIZES=[100,200,400,800]` will extend by generating only the new `real_800` datasets.

## Output

Creates `sequence_0/` through `sequence_4/` directories, each containing:
- `seq_X_real_Y_aug_0.csv` - Base sets (no augmentation)
- `seq_X_real_Y_aug_Z.csv` - Augmented datasets

CSV columns:
- `Text`: The claim text
- `class_label`: "Yes" or "No"
- `example`: Empty for real samples, template text for synthetic samples
