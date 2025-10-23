import json
import os
import random


def make_config(model_name: str, data_artifact_name: str, group_name: str, seed: int, batch_size: int, file_num: int, num_train_epochs: int):
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
        config["num_train_epochs"] = num_train_epochs

        # Prepare output path
        out_dir = os.path.join(os.path.dirname(__file__), "../finetune/configs")
        out_path = os.path.join(out_dir, f"train_{file_num:02d}.json")

        # Save new config
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)


model_name = "roberta-base"
run_base_names = ["ct24_seq", "ct24_synth_seq"]
run_suffixes = [
    "0_size_156",
    "0_size_312",
    "0_size_625",
    "0_size_1250",
    "0_size_2500",
    "0_size_5000",
    "0_size_10000",
    "1_size_156",
    "1_size_312",
    "1_size_625",
    "1_size_1250",
    "1_size_2500",
    "1_size_5000",
    "1_size_10000",
    "2_size_156",
    "2_size_312",
    "2_size_625",
    "2_size_1250",
    "2_size_2500",
    "2_size_5000",
    "2_size_10000",
]

num_seeds = 1
file_num = 0
batch_size = 128

for run_base_name in run_base_names:
    for run_suffix in run_suffixes:

        data_artifact_name = f"{run_base_name}_{run_suffix}"
        group_name = run_base_name

        size = int(run_suffix.split("_")[-1])
        if "synth" in run_base_name:
             size += 2000
        num_train_epochs = min(70000 // size, 100)

        for _ in range(num_seeds):
            seed = random.randint(0, 2**16)
            make_config(
                 model_name=model_name,
                 data_artifact_name=data_artifact_name,
                 group_name=group_name,
                 seed=seed,
                 batch_size=batch_size,
                 file_num=file_num,
                 num_train_epochs=num_train_epochs)
            file_num += 1