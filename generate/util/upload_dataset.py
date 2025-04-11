import wandb
import argparse

def upload_dataset(
    dataset_name: str,
    description: str,
    files: dict,
    metadata: dict = None,
):
    with wandb.init(project="thesis", job_type="generate-data") as run:

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload dataset to Weights & Biases.")
    parser.add_argument("--train", required=True, help="Path to the training dataset file.")
    parser.add_argument("--dev", required=True, help="Path to the development dataset file.")
    parser.add_argument("--dev-test", required=True, help="Path to the development test dataset file.")
    parser.add_argument("--test", required=True, help="Path to the test dataset file.")
    parser.add_argument("--template", help="Path to the template file used to create the synthetic samples (optional).")
    parser.add_argument("--description", required=True, help="Description of the dataset.")
    parser.add_argument("--name", required=True, help="Name of the dataset.")

    args = parser.parse_args()

    files = {
        "train": args.train,
        "dev": args.dev,
        "dev-test": args.dev_test,
        "test": args.test,
    }

    if args.template:
        files["template"] = args.template

    upload_dataset(
        dataset_name=args.name,
        description=args.description,
        files=files,
    )
