import os
import json
import wandb

run_names = [
    "ct24_synth_0_10k",
    "ct24_synth_2k_8k",
    "ct24_synth_4k_6k",
    "ct24_synth_6k_4k",
    "ct24_synth_8k_2k",
    "ct24_synth_10k_0",
    "gc_synth_0_10k",
    "gc_synth_2k_8k",
    "gc_synth_4k_6k",
    "gc_synth_6k_4k",
    "gc_synth_8k_2k",
    "gc_synth_10k_0",
]

eval_datasets = [
    ("synth_test", "synth"),
]

def main():
    wandb.login()  # assumes env vars or config for API key
    api = wandb.Api()
    wandb_project_base = 'redstag/thesis'

    template_path = os.path.join(os.path.dirname(__file__), "../finetune/config_templates/eval.json")
    output_dir = os.path.join(os.path.dirname(__file__), "../finetune/configs")
    counter = 0

    for run_name in run_names:
        runs = api.runs(path=wandb_project_base, filters={"display_name": run_name})
        for run in runs:
            # Find all artifacts named "last_model:vXX"
            for artifact in run.logged_artifacts():
                if artifact.name.startswith("last_model:v") and artifact.type == "model":
                    # For each dataset, create a config
                    for dataset_name, dataset_eval_prefix in eval_datasets:

                        with open(template_path, "r") as f:
                            config = json.load(f)

                        config["run_name"] = f"{run_name}_{dataset_eval_prefix}_eval"
                        config["data_artifact"] = dataset_name + ':latest'
                        config["prediction_model_artifact"] = artifact.name
                        config["wandb_group_name"] = "eval"
                        out_name = f"eval_{counter:02d}.json"
                        out_path = os.path.join(output_dir, out_name)

                        with open(out_path, "w") as f:
                            json.dump(config, f, indent=2)

                        counter += 1

if __name__ == "__main__":
    main()