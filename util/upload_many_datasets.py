import subprocess
import os

data_dir = "../data/general_claim"


datasets = [
    {
        "name": fname.replace("-", "_").replace(".csv", ""),
        "path": os.path.join(data_dir, fname),
        "description": f"General claim with {fname.split('_')[1]} == {fname.split('_')[2].split('.')[0]}"
    }
    for fname in os.listdir(data_dir)
    if fname.startswith("gc_")
]


for ds in datasets:
    cmd = [
        "python", "upload_dataset.py",
        "--train", ds["path"],
        "--dev", ds["path"],
        "--dev-test", ds["path"],
        "--test", ds["path"],
        "--name", ds["name"],
        "--description", ds["description"]
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
