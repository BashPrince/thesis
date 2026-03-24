"""Build a holdout dataset artifact from samples not seen during training.

Given one or more WandB groups, finds every data artifact used by runs in those
groups, downloads their train.csv splits, and collects all Text values that
appeared in training.  Then takes a local reference file and keeps only the rows
whose Text was NOT in any training set.  The resulting holdout samples are
uploaded as a new WandB dataset artifact (with identical content in every split
for compatibility with run_classification.py).

Usage:
    python util/make_holdout_artifact.py group1 group2 \
        --artifact-name "holdout_exp1" \
        --local-file data/CT24_checkworthy_english/CT24_checkworthy_english_train.csv
"""

import argparse
import os
import tempfile

import pandas as pd
import wandb

WANDB_PROJECT = "thesis"
WANDB_ENTITY = "redstag"


def collect_training_texts(entity, project, groups):
    """Return the set of all Text values used for training across all runs
    in the given groups, plus the list of unique dataset artifact names."""
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project

    seen_artifacts = set()
    all_texts = set()

    for group in groups:
        print(f"Querying group '{group}'...")
        runs = api.runs(path, filters={"group": group})

        checked = 0
        for run in runs:
            checked += 1
            print(f"\r  {checked} run(s) checked, {len(seen_artifacts)} unique artifact(s)...", end="", flush=True)
            for artifact in run.used_artifacts():
                if artifact.type != "dataset":
                    continue
                if artifact.name in seen_artifacts:
                    continue
                seen_artifacts.add(artifact.name)

                artifact_dir = artifact.download()
                train_path = os.path.join(artifact_dir, "train.csv")
                if not os.path.exists(train_path):
                    print(f"\n  WARNING: no train.csv in {artifact.name}, skipping")
                    continue

                df = pd.read_csv(train_path)
                if "Text" not in df.columns:
                    print(f"\n  WARNING: no Text column in {artifact.name}/train.csv, skipping")
                    continue

                all_texts.update(df["Text"].dropna().tolist())

        print()

    print(f"Collected {len(all_texts)} unique training texts from {len(seen_artifacts)} artifact(s).")
    return all_texts


def main():
    parser = argparse.ArgumentParser(description="Build a holdout dataset artifact from unseen samples.")
    parser.add_argument("groups", nargs="+", help="WandB group name(s) whose training data to exclude.")
    parser.add_argument("--artifact-name", required=True, help="Name for the uploaded WandB artifact.")
    parser.add_argument(
        "--local-file",
        default="data/CT24_checkworthy_english/CT24_checkworthy_english_train.csv",
        help="Local CSV with reference samples (default: CT24 english train).",
    )
    args = parser.parse_args()

    print(f"Reading local file: {args.local_file}")
    local_df = pd.read_csv(args.local_file)
    total = len(local_df)

    training_texts = collect_training_texts(WANDB_ENTITY, WANDB_PROJECT, args.groups)

    holdout_df = local_df[~local_df["Text"].isin(training_texts)].reset_index(drop=True)
    print(f"Holdout: {len(holdout_df)} / {total} samples (excluded {total - len(holdout_df)} seen in training)")

    if holdout_df.empty:
        print("No holdout samples — all local samples were used in training. Aborting.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        for split in ("train", "dev", "dev-test", "test"):
            holdout_df.to_csv(os.path.join(tmp, f"{split}.csv"), index=False)

        with wandb.init(project=WANDB_PROJECT, job_type="generate-data", group="datagen") as run:
            artifact = wandb.Artifact(args.artifact_name, type="dataset",
                                      description=f"Holdout samples not seen in groups: {', '.join(args.groups)}")
            for split in ("train", "dev", "dev-test", "test"):
                artifact.add_file(os.path.join(tmp, f"{split}.csv"), name=f"{split}.csv")
            run.log_artifact(artifact)

    print(f"Uploaded artifact '{args.artifact_name}' with {len(holdout_df)} samples.")


if __name__ == "__main__":
    main()
