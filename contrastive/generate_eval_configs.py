"""Generate eval configs for saved models from a WandB group.

Queries WandB to find all runs matching GROUP + RUN_NAMES, retrieves the
'best_model' artifact logged by each run, and writes one eval config per
artifact into finetune/configs/.

All configuration is done via the variables below — no CLI args.
"""

import json
import os

import wandb

# ---------------------------------------------------------------------------
# Configuration — edit these variables
# ---------------------------------------------------------------------------

WANDB_PROJECT = "thesis"
WANDB_ENTITY = "redstag"  # set to your entity string if needed, e.g. "my-team"
MODEL = "answerdotai/ModernBERT-base"

# Group containing the runs to evaluate
SOURCE_GROUP = "unrestricted_wrup_extended"

aug_steps = [0, 128, 256, 512, 1024, 2048, 4096]
# Run names to evaluate (all must belong to GROUP)
run_names = []
for i in range(10):
    for aug in aug_steps:
        run_names.append(f"seq_{i}_aug_{aug}")

# WandB group name for the generated eval runs
EVAL_GROUP = "unrestricted_wrup_extended_dev_test_eval"

# Evaluation dataset artifact (e.g. "my_eval_dataset:latest")
DATA_ARTIFACT = "ct24:latest"

# Which file inside the artifact to use as the test split
TEST_FILE = "dev-test.csv"

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "..", "finetune", "config_templates", "eval.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "finetune", "configs")

# ---------------------------------------------------------------------------


def get_best_model_artifacts(entity, project, group, run_names):
    """Return a list of (run_name, artifact_identifier) for every best_model
    artifact logged by runs in the given group whose name is in run_names."""
    api = wandb.Api()
    run_name_set = set(run_names)

    filters = {"group": group, "display_name": {"$in": list(run_name_set)}}
    runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)

    results = []
    runs_checked = 0
    for run in runs:
        runs_checked += 1
        print(f"\r  {runs_checked} run(s) checked, {len(results)} artifact(s) found...", end="", flush=True)
        for artifact in run.logged_artifacts():
            if artifact.name.startswith("best_model") and artifact.type == "model":
                # Full identifier: entity/project/name:version
                identifier = artifact.name
                results.append((run.name, run.id, identifier))
                print(f"\r  {runs_checked} run(s) checked, {len(results)} artifact(s) found...", end="", flush=True)

    print()  # newline after the progress line
    return results


def main():
    with open(TEMPLATE_PATH) as f:
        template = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    entity = WANDB_ENTITY
    if entity is None:
        # Fall back to the default entity from the local wandb settings
        api = wandb.Api()
        entity = api.default_entity

    print(f"Querying WandB project '{entity}/{WANDB_PROJECT}', group '{SOURCE_GROUP}'...")
    artifacts = get_best_model_artifacts(entity, WANDB_PROJECT, SOURCE_GROUP, run_names)

    if not artifacts:
        print("No best_model artifacts found. Check GROUP and RUN_NAMES.")
        return

    print(f"Found {len(artifacts)} artifact(s). Generating eval configs...")

    for i, (run_name, _, artifact_id) in enumerate(artifacts, start=1):
        eval_run_name = run_name
        filename = f"eval_{i:03d}.json"

        cfg = dict(template)
        cfg["run_name"] = eval_run_name
        cfg["wandb_group_name"] = EVAL_GROUP
        cfg["data_artifact"] = DATA_ARTIFACT
        cfg["test_file"] = TEST_FILE
        cfg["prediction_model_artifact"] = artifact_id
        cfg["model_name_or_path"] = MODEL

        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=4)

        print(f"  {filename}  ({artifact_id})")

    print(f"\nWrote {len(artifacts)} config(s) to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
