import subprocess
import os

data_dir = "../data/CT24_checkworthy_english/shrink"


datasets = [
    {
        "name": fname.replace("_shrink", "").replace(".csv", ""),
        "path": os.path.join(data_dir, fname),
        "description": "\"CT24\""
    }
    for fname in os.listdir(data_dir)
    if fname.startswith("ct24")
]


for ds in datasets:
    cmd = [
        "python", "upload_dataset.py",
        "--train", ds["path"],
        "--dev", "../data/CT24_checkworthy_english/dev.csv",
        "--dev-test", "../data/CT24_checkworthy_english/dev-test.csv",
        "--test", "../data/CT24_checkworthy_english/test-combined.csv",
        "--name", ds["name"],
        "--description", ds["description"]
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)