import os
import json
import wandb

run_names = [
    "ct24_200_synth_seq_0_0",
    "ct24_200_synth_seq_0_200",
    "ct24_200_synth_seq_0_400",
    "ct24_200_synth_seq_0_800",
    "ct24_200_synth_seq_0_1600",
    "ct24_200_synth_seq_1_0",
    "ct24_200_synth_seq_1_200",
    "ct24_200_synth_seq_1_400",
    "ct24_200_synth_seq_1_800",
    "ct24_200_synth_seq_1_1600",
    "ct24_200_synth_seq_2_0",
    "ct24_200_synth_seq_2_200",
    "ct24_200_synth_seq_2_400",
    "ct24_200_synth_seq_2_800",
    "ct24_200_synth_seq_2_1600",
    "ct24_200_synth_seq_3_0",
    "ct24_200_synth_seq_3_200",
    "ct24_200_synth_seq_3_400",
    "ct24_200_synth_seq_3_800",
    "ct24_200_synth_seq_3_1600",
    "ct24_200_synth_seq_4_0",
    "ct24_200_synth_seq_4_200",
    "ct24_200_synth_seq_4_400",
    "ct24_200_synth_seq_4_800",
    "ct24_200_synth_seq_4_1600",
]
#wandb_run_group_name = "ct24_synth"
model_name = "roberta-base"

eval_datasets = [
    ("ct24_test_combined", "combined")
]

wandb_eval_group_name = "ct24_200_synth_combined_eval"
test_file = "test.csv"

def main():
    wandb.login()  # assumes env vars or config for API key
    api = wandb.Api()
    wandb_project_base = 'redstag/thesis'

    template_path = os.path.join(os.path.dirname(__file__), "../finetune/config_templates/eval.json")
    output_dir = os.path.join(os.path.dirname(__file__), "../finetune/configs")
    counter = 0

    for run_name in run_names:
        runs = api.runs(
            path=wandb_project_base,
            filters={
            "display_name": run_name,
            #"group": wandb_run_group_name
            }
        )
        for run in runs:
            # Find all artifacts named "last_model:vXX"
            for artifact in run.logged_artifacts():
                if (artifact.name.startswith("last_model:v") or artifact.name.startswith("best_model:v")) and artifact.type == "model":
                    # For each dataset, create a config
                    for dataset_name, dataset_eval_prefix in eval_datasets:

                        with open(template_path, "r") as f:
                            config = json.load(f)

                        config["run_name"] = f"{run_name}_{dataset_eval_prefix}_eval"
                        config["data_artifact"] = dataset_name + ':latest'
                        config["prediction_model_artifact"] = artifact.name
                        config["wandb_group_name"] = wandb_eval_group_name
                        config["model_name_or_path"] = model_name
                        config["test_file"] = test_file
                        out_name = f"eval_{counter:02d}.json"
                        out_path = os.path.join(output_dir, out_name)

                        with open(out_path, "w") as f:
                            json.dump(config, f, indent=2)

                        counter += 1

if __name__ == "__main__":
    main()