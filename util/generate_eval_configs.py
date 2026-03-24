"""Generate eval configs for all runs in one or more WandB groups.

For each run in the given groups, finds the best/last model artifact and writes
an eval config JSON (from the eval.json template) into finetune/configs/.

Usage:
    python util/generate_eval_configs.py group1 group2 \
        --data-artifact "ct24:latest" \
        --test-file "dev-test.csv"
"""

import argparse
import json
import os
import sys

import wandb

WANDB_PROJECT = "thesis"
WANDB_ENTITY = "redstag"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "..", "finetune", "config_templates", "eval.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "finetune", "configs")


def find_model_artifact(run):
    """Return the model artifact identifier for a run, preferring best_model."""
    best = None
    last = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model":
            continue
        if artifact.name.startswith("best_model:"):
            best = artifact.name
        elif artifact.name.startswith("last_model:"):
            last = artifact.name
    return best or last


def get_runs_with_artifacts(entity, project, group):
    """Return list of (run_name, artifact_id) for all runs in a group."""
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"group": group})

    results = []
    checked = 0
    for run in runs:
        checked += 1
        artifact_id = find_model_artifact(run)
        status = f"  {checked} run(s) checked, {len(results)} artifact(s) found..."
        print(f"\r{status}", end="", flush=True)
        if artifact_id:
            results.append((run.name, artifact_id))
    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate eval configs from WandB groups.")
    parser.add_argument("groups", nargs="+", help="WandB group name(s) to generate eval configs for.")
    parser.add_argument("--data-artifact", required=True, help="Dataset artifact (e.g. 'ct24:latest').")
    parser.add_argument("--test-file", default="test.csv", help="Test file inside the dataset artifact (default: test.csv).")
    parser.add_argument("--model", default="roberta-base", help="Model name (default: roberta-base).")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory to write eval configs to.")
    args = parser.parse_args()

    with open(TEMPLATE_PATH) as f:
        template = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Find the next eval config number in the output dir
    existing = [f for f in os.listdir(args.output_dir) if f.startswith("eval_") and f.endswith(".json")]
    if existing:
        max_num = max(int(f.removeprefix("eval_").removesuffix(".json")) for f in existing)
        counter = max_num + 1
    else:
        counter = 1

    total_written = 0
    for group in args.groups:
        eval_group = f"{group}_eval"
        print(f"Querying WandB group '{group}'...")
        runs_artifacts = get_runs_with_artifacts(WANDB_ENTITY, WANDB_PROJECT, group)

        if not runs_artifacts:
            print(f"  No model artifacts found in group '{group}', skipping.")
            continue

        print(f"  Found {len(runs_artifacts)} artifact(s). Generating configs...")

        for run_name, artifact_id in runs_artifacts:
            cfg = dict(template)
            cfg["run_name"] = run_name
            cfg["wandb_group_name"] = eval_group
            cfg["data_artifact"] = args.data_artifact
            cfg["test_file"] = args.test_file
            cfg["prediction_model_artifact"] = artifact_id
            cfg["model_name_or_path"] = args.model

            filename = f"eval_{counter:02d}.json"
            out_path = os.path.join(args.output_dir, filename)
            with open(out_path, "w") as f:
                json.dump(cfg, f, indent=4)

            print(f"    {filename}  {run_name} -> {artifact_id}")
            counter += 1
            total_written += 1

    print(f"\nWrote {total_written} eval config(s) to {args.output_dir}/")


if __name__ == "__main__":
    main()
