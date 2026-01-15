import subprocess
import os

data_dir = "../data/CT24_checkworthy_english/subsets/synthetic/seq_4_append"


datasets = [
    {
        "name": "ct24_200_synth_seq_" + fname.replace(".csv", "").replace("subset_", ""),
        "path": os.path.join(data_dir, fname),
        "description": "\"200 CT24 samples augmented with increasing amounts of example prompted synthetic data (examples from same training data)\""
    }
    for fname in os.listdir(data_dir)
    if fname.startswith("subset")
]


for ds in datasets:
    cmd = [
        "python", "upload_dataset.py",
        "--train", ds["path"],
        "--dev", "../data/CT24_checkworthy_english/dev.csv",
        "--dev-test", "../data/CT24_checkworthy_english/dev-test.csv",
        "--test", "../data/CT24_checkworthy_english/test-combined-large.csv",
        "--name", ds["name"],
        "--description", ds["description"]
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)