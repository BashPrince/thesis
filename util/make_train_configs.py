import json
import os


def make_config(model_name: str, data_artifact_name: str, group_name: str, seed: int, batch_size: int, file_num: int):
        # Load template
        template_path = os.path.join(os.path.dirname(__file__), "../finetune/config_templates/train.json")
        with open(template_path, "r") as f:
            config = json.load(f)

        # Update fields
        config["model_name_or_path"] = model_name
        config["run_name"] = data_artifact_name
        config["data_artifact"] = data_artifact_name + ":latest"
        config["wandb_group_name"] = group_name
        config["seed"] = seed
        config["per_device_train_batch_size"] = batch_size
        config["per_device_eval_batch_size"] = batch_size

        # Prepare output path
        out_dir = os.path.join(os.path.dirname(__file__), "../finetune/configs")
        out_path = os.path.join(out_dir, f"train_{file_num:02d}.json")

        # Save new config
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)


model_name = "roberta-base"
run_prefixes = ["ct24_"]
run_mid = "synth"
dataset_mixes = [
    ("9k", "1k"),
    ("7k", "3k"),
]

base_seed = 42
num_seeds = 3
file_num = 0
batch_size = 128

for run_prefix in run_prefixes:
    for data_mix in dataset_mixes:

        data_prefix = run_prefix + run_mid
        data_artifact_name = f"{data_prefix}_{data_mix[0]}_{data_mix[1]}"
        group_name = data_prefix

        for seed in range(base_seed, base_seed + num_seeds):
            make_config(model_name=model_name, data_artifact_name=data_artifact_name, group_name=group_name, seed=seed, batch_size=batch_size, file_num=file_num)
            file_num += 1