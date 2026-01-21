import os
import glob
import pandas as pd
import numpy as np
import wandb
import random
import json

def upload_dataset(
    dataset_name: str,
    description: str,
    files: dict,
    metadata: dict = None,
):
    with wandb.init(project="thesis", job_type="generate-data", group='datagen') as run:

        # 🏺 create our Artifact
        data = wandb.Artifact(
            dataset_name,
            type="dataset",
            description=description,
            metadata=metadata)

        # 📦 Add files to the Artifact
        for name, file in files.items():
            suffix = file.split(".")[-1]
            data.add_file(file, name=f"{name}.{suffix}")

        # ✍️ Save the artifact to W&B.
        run.log_artifact(data)

def delete_configs():
    # Delete existing config files before creating new ones
    config_dir = os.path.join(os.path.dirname(__file__), "../../../finetune/configs")
    if os.path.exists(config_dir):
        for f in os.listdir(config_dir):
            file_path = os.path.join(config_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

def make_config(model_name: str, data_artifact_name: str, group_name: str, seed: int, batch_size: int, file_num: int, num_epochs: int, load_best_model: bool):
        # Load template
        template_path = os.path.join(os.path.dirname(__file__), "../../../finetune/config_templates/train.json")
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
        config["num_train_epochs"] = num_epochs
        config["load_best_model_at_end"] = load_best_model

        # Prepare output path
        out_dir = os.path.join(os.path.dirname(__file__), "../../../finetune/configs")
        out_path = os.path.join(out_dir, f"train_{file_num:02d}.json")

        # Save new config
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)

def make_configs(artifact_base_name: str, total_train_samples: int, train_model: str, batch_size: int, load_best_model: bool, num_seeds: int):
    delete_configs()
    config_file_suffix = 0

    for root, dirs, files in os.walk(os.getcwd()):
        # Skip the 'wandb' folder
        if 'wandb' in dirs:
            dirs.remove('wandb')
        for file in files:
            # Skip this file
            if "create_data.py" in file:
                continue

            file_path = os.path.join(root, file)
            dataset_name = f"{artifact_base_name}_{file.replace('.csv', '')}"

            dataset_size = len(pd.read_csv(file_path))
            num_epochs = total_train_samples // dataset_size

            for _ in range(num_seeds):
                seed = random.randint(0, 2**16)
                make_config(
                    model_name=train_model,
                    data_artifact_name=dataset_name,
                    group_name=artifact_base_name,
                    seed=seed,
                    batch_size=batch_size,
                    file_num=config_file_suffix,
                    num_epochs=num_epochs,
                    load_best_model=load_best_model)
                config_file_suffix += 1


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_GLOB = os.path.join(BASE_DIR, "sequence_*")
TRAIN_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../data/CT24_checkworthy_english/train.csv"))
AUG_SIZES = [100, 200, 400, 800]
DEV_FILE = "../../../data/CT24_checkworthy_english/dev-wo-id.csv"
DEV_TEST_FILE = "../../../data/CT24_checkworthy_english/dev-test-wo-id.csv"
TEST_FILE = "../../../data/CT24_checkworthy_english/test-combined-wo-id.csv"

make_sequences = False
do_upload = False
do_make_configs = True
total_train_samples = 45000
batch_size = 64
num_seeds = 3
load_best_model = True

# Load and prepare train.csv
train_df = pd.read_csv(TRAIN_PATH)
train_df = train_df.drop(columns=["Sentence_id"])
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# Find all sequence directories
seq_dirs = sorted(glob.glob(SEQ_GLOB))

# Delete any present augmented sequences from previous runs (except aug_0)
if make_sequences:
    for seq_dir in seq_dirs:
        aug_files = glob.glob(os.path.join(seq_dir, "seq_*_aug_*.csv"))
        for f in aug_files:
            if not f.endswith("_aug_0.csv"):
                os.remove(f)

# Gather all "Text" values from base sequences
base_texts = set()
for seq_dir in seq_dirs:
    seq_idx = os.path.basename(seq_dir).split("_")[-1]
    base_csv = os.path.join(seq_dir, f"seq_{seq_idx}_aug_0.csv")
    base_df = pd.read_csv(base_csv)
    base_texts.update(base_df["Text"].tolist())

# Remove any train_df rows whose "Text" is in any base sequence
train_df = train_df[~train_df["Text"].isin(base_texts)].reset_index(drop=True)

# Split train_df by class
yes_df = train_df[train_df["class_label"] == "Yes"].reset_index(drop=True)
no_df = train_df[train_df["class_label"] == "No"].reset_index(drop=True)

# Track used indices to avoid duplicates across all sequences
used_yes_idx = set()
used_no_idx = set()

if make_sequences:
    for seq_dir in seq_dirs:
        seq_idx = os.path.basename(seq_dir).split("_")[-1]
        base_csv = os.path.join(seq_dir, f"seq_{seq_idx}_aug_0.csv")
        base_df = pd.read_csv(base_csv)

        prev_aug_df = base_df.copy()
        prev_j = 0

        for j in AUG_SIZES:
            n_new = j - prev_j
            n_each = n_new // 2

            # Get available indices
            available_yes = list(set(yes_df.index) - used_yes_idx)
            available_no = list(set(no_df.index) - used_no_idx)

            if len(available_yes) < n_each or len(available_no) < n_each:
                raise ValueError(f"Not enough unused samples for augmentation size {j} in sequence {seq_idx}")

            # Sample without replacement
            sampled_yes_idx = np.random.choice(available_yes, n_each, replace=False)
            sampled_no_idx = np.random.choice(available_no, n_each, replace=False)

            # Mark as used
            used_yes_idx.update(sampled_yes_idx)
            used_no_idx.update(sampled_no_idx)

            # Get sampled rows
            sampled_yes = yes_df.loc[sampled_yes_idx]
            sampled_no = no_df.loc[sampled_no_idx]

            # Extend previous augmentation
            aug_df = pd.concat([prev_aug_df, sampled_yes, sampled_no], ignore_index=True)

            # Save
            out_csv = os.path.join(seq_dir, f"seq_{seq_idx}_aug_{j}.csv")
            aug_df.to_csv(out_csv, index=False)

            # Prepare for next step
            prev_aug_df = aug_df
            prev_j = j

# Upload all datasets collectively at the end using expected filenames
if do_upload:
    for seq_dir in seq_dirs:
        seq_idx = os.path.basename(seq_dir).split("_")[-1]
        # Base sequence
        base_csv = os.path.join(seq_dir, f"seq_{seq_idx}_aug_0.csv")
        files = {
            "train": base_csv,
            "dev": DEV_FILE,
            "dev-test": DEV_TEST_FILE,
            "test": TEST_FILE,
        }
        description = f"Augmented sequence {seq_idx} with 0 extra samples for experiment_008"
        upload_dataset(f"experiment_008_seq_{seq_idx}_aug_0", description, files)
        # Augmented sequences
        for j in AUG_SIZES:
            aug_csv = os.path.join(seq_dir, f"seq_{seq_idx}_aug_{j}.csv")
            files = {
                "train": aug_csv,
                "dev": DEV_FILE,
                "dev-test": DEV_TEST_FILE,
                "test": TEST_FILE,
            }
            description = f"Augmented sequence {seq_idx} with {j} extra samples for experiment_008"
            upload_dataset(f"experiment_008_seq_{seq_idx}_aug_{j}", description, files)

if do_make_configs:
    make_configs(
        artifact_base_name="experiment_008",
        total_train_samples=total_train_samples,
        train_model="roberta-base",
        batch_size=batch_size,
        load_best_model=load_best_model,
        num_seeds=num_seeds)

