"""Upload embed-gap datasets to W&B (one artifact per run_i).

Mirrors v7_upload.py's scheme but only uploads the new embed-gap method.
Artifact name: {artifact_base}_seq_{i}_aug_embed-gap

Usage:
    python v7_upload_gap.py                                            # defaults to v7_output_extend
    python v7_upload_gap.py --output-dir v7_output
"""
import argparse
from pathlib import Path

import wandb

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "v7_output_extend"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "CT24_checkworthy_english"

DEV_FILE      = DATA_DIR / "CT24_checkworthy_english_dev.csv"
DEV_TEST_FILE = DATA_DIR / "CT24_checkworthy_english_dev-test.csv"
TEST_FILE     = DATA_DIR / "CT24_checkworthy_english_test_gold.csv"

METHOD_NAME = "embed-gap"
TRAIN_FILENAME = "augmented_embedding_gap.csv"


def upload_dataset(run, dataset_name, description, files, metadata=None):
    for name, filepath in files.items():
        p = Path(filepath)
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(f"{name} -> {filepath} missing or empty")

    data = wandb.Artifact(dataset_name, type="dataset",
                          description=description, metadata=metadata)
    for name, filepath in files.items():
        data.add_file(str(filepath), name=f"{name}{Path(filepath).suffix}")
    logged = run.log_artifact(data)
    logged.wait()
    print(f"    committed {dataset_name} "
          f"({len(data.manifest.entries)} entries, {data.size} bytes)")


def main(output_dir, artifact_base):
    output_dir = Path(output_dir)
    run_dirs = sorted([d for d in output_dir.iterdir()
                       if d.is_dir() and d.name.startswith("run_")],
                      key=lambda p: int(p.name.split("_")[1]))
    if not run_dirs:
        print(f"No run directories found in {output_dir}")
        return

    print(f"Found {len(run_dirs)} runs in {output_dir}")
    with wandb.init(project="thesis", job_type="generate-data",
                    group="datagen", name=f"upload_{artifact_base}_{METHOD_NAME}") as run:
        for run_dir in run_dirs:
            run_idx = int(run_dir.name.split("_")[1])
            train_path = run_dir / TRAIN_FILENAME
            if not train_path.exists():
                print(f"  Skipping seq_{run_idx}: {TRAIN_FILENAME} not found")
                continue

            dataset_name = f"{artifact_base}_seq_{run_idx}_aug_{METHOD_NAME}"
            files = {"train": train_path, "dev": DEV_FILE,
                     "dev-test": DEV_TEST_FILE, "test": TEST_FILE}
            print(f"  Uploading {dataset_name} (train={TRAIN_FILENAME})")
            upload_dataset(
                run,
                dataset_name=dataset_name,
                description=f"{artifact_base} seq {run_idx}, method: {METHOD_NAME}",
                files=files,
                metadata={"method": METHOD_NAME, "run_idx": run_idx},
            )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--artifact-base", type=str, default=None,
                        help="Base name for artifacts (default: output dir name)")
    args = parser.parse_args()
    artifact_base = args.artifact_base or Path(args.output_dir).name
    main(output_dir=args.output_dir, artifact_base=artifact_base)
