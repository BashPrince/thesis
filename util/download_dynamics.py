import wandb
import argparse
import shutil
import os

parser = argparse.ArgumentParser(description="Download and move dynamics artifact.")
parser.add_argument('--dir', type=str, required=True, help="Directory to move the downloaded artifact.")
parser.add_argument('--artifact', type=str, required=True, help="Name of the artifact to download.")
args = parser.parse_args()

run = wandb.init()
artifact_name = f"redstag/thesis/{args.artifact}"
artifact = run.use_artifact(f"{artifact_name}", type='dynamics')
artifact_dir = artifact.download()

# Move the downloaded directory to the specified location
destination_dir = args.dir
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
shutil.move(artifact_dir, os.path.join(destination_dir, "training_dynamics"))

