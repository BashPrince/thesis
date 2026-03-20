"""Upload v7 output datasets to W&B as artifacts.

For each run_i, uploads four artifacts (seq_{i}_aug_{method}):
  - none:  real.csv as train split (no augmentation)
  - real:  augmented_real.csv as train split
  - tfidf: augmented_tfidf.csv as train split
  - embed: augmented_embedding.csv as train split

All share the same dev / dev-test / test splits.

Usage:
    python v7_upload.py
    python v7_upload.py --output-dir v7_output --artifact-base v7_poolfilter
"""
import argparse
from pathlib import Path

import wandb

OUTPUT_DIR = Path(__file__).parent / "v7_output"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "CT24_checkworthy_english"

DEV_FILE      = DATA_DIR / "CT24_checkworthy_english_dev.csv"
DEV_TEST_FILE = DATA_DIR / "CT24_checkworthy_english_dev-test.csv"
TEST_FILE     = DATA_DIR / "CT24_checkworthy_english_test_gold.csv"

# Maps method name → train file inside each run directory
METHODS = {
    "none":       "real.csv",
    "real":       "augmented_real.csv",
    "unfiltered": "augmented_unfiltered.csv",
    "tfidf":      "augmented_tfidf.csv",
    "embed":      "augmented_embedding.csv",
}


def upload_dataset(dataset_name, description, files, metadata=None):
    with wandb.init(project="thesis", job_type="generate-data", group="datagen") as run:
        data = wandb.Artifact(
            dataset_name,
            type="dataset",
            description=description,
            metadata=metadata,
        )
        for name, filepath in files.items():
            suffix = Path(filepath).suffix
            data.add_file(str(filepath), name=f"{name}{suffix}")
        run.log_artifact(data)


def main(output_dir, artifact_base):
    output_dir = Path(output_dir)
    run_dirs = sorted(output_dir.iterdir(), key=lambda p: p.name)
    run_dirs = [d for d in run_dirs if d.is_dir() and d.name.startswith("run_")]

    if not run_dirs:
        print(f"No run directories found in {output_dir}")
        return

    print(f"Found {len(run_dirs)} runs in {output_dir}")

    for run_dir in run_dirs:
        run_idx = int(run_dir.name.split("_")[1])

        for method, train_filename in METHODS.items():
            train_path = run_dir / train_filename
            if not train_path.exists():
                print(f"  Skipping seq_{run_idx}_aug_{method}: {train_filename} not found")
                continue

            dataset_name = f"{artifact_base}_seq_{run_idx}_aug_{method}"
            files = {
                "train":    train_path,
                "dev":      DEV_FILE,
                "dev-test": DEV_TEST_FILE,
                "test":     TEST_FILE,
            }

            print(f"  Uploading {dataset_name} (train={train_filename})")
            upload_dataset(
                dataset_name=dataset_name,
                description=f"{artifact_base} seq {run_idx}, augmentation method: {method}",
                files=files,
                metadata={"method": method, "run_idx": run_idx},
            )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--artifact-base", type=str, default=None,
                        help="Base name for artifacts (default: output dir name)")
    args = parser.parse_args()
    artifact_base = args.artifact_base or Path(args.output_dir).name
    main(output_dir=args.output_dir, artifact_base=artifact_base)
