import json
import os


def make_config(data_artifact_name: str, group_name: str, seed: int, file_num: int):
        # Load template
        template_path = os.path.join(os.path.dirname(__file__), "../finetune/config_templates/train.json")
        with open(template_path, "r") as f:
            config = json.load(f)

        # Update fields
        config["run_name"] = data_artifact_name
        config["data_artifact"] = data_artifact_name + ":latest"
        config["wandb_group_name"] = group_name
        config["seed"] = seed

        # Prepare output path
        out_dir = os.path.join(os.path.dirname(__file__), "../finetune/configs")
        out_path = os.path.join(out_dir, f"train_{file_num:02d}.json")

        # Save new config
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)


run_prefixes = ["ct24_", "gc_"]
run_mid = "synth"
dataset_mixes = [
    # ("10k", "0"),
    # ("8k", "2k"),
    # ("6k", "4k"),
    # ("4k", "6k"),
    # ("2k", "8k"),
    ("0", "10k"),
]

base_seed = 42
num_seeds = 3
file_num = 0

for run_prefix in run_prefixes:
    for data_mix in dataset_mixes:

        group_name = run_prefix + run_mid
        data_artifact_name = f"{group_name}_{data_mix[0]}_{data_mix[1]}"

        for seed in range(base_seed, base_seed + num_seeds):
            make_config(data_artifact_name=data_artifact_name, group_name=group_name, seed=seed, file_num=file_num)
            file_num += 1