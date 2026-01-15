import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

source_dir = '../data/CT24_checkworthy_english/subsets'
target_dir = '../data/CT24_checkworthy_english/subsets/synthetic'
file_prefix = 'subset'
num_augment = 1600

datasets = [fname for fname in os.listdir(source_dir) if fname.startswith(file_prefix)]
commands = []

for d in datasets:
    commands.append(
        ["python", "generate.py",
         "-n", "--num_samples",
         f"{num_augment // 2}", "--num_samples_per_prompt",
         "5", "--example_source",
         os.path.join(source_dir, d), "--template", "templates/neg_template.txt",
         "--out_file", os.path.join(target_dir, d.replace(".csv", "_neg.csv"))])
    commands.append(
        ["python", "generate.py",
         "--num_samples",
         f"{num_augment // 2}", "--num_samples_per_prompt",
         "5", "--example_source",
         os.path.join(source_dir, d), "--template", "templates/pos_template.txt",
         "--out_file", os.path.join(target_dir, d.replace(".csv", "_pos.csv"))])


num_sync = 5

def run_command(cmd):
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=num_sync) as executor:
        futures = [executor.submit(run_command, cmd) for cmd in commands]
        for future in as_completed(futures):
            rc = future.result()
            if rc != 0:
                print(f"Command failed with return code {rc}")

    # Concatenate _pos and _neg files for each dataset using pandas
    for d in datasets:
        base_name = d.replace(".csv", "")
        neg_file = os.path.join(target_dir, f"{base_name}_neg.csv")
        pos_file = os.path.join(target_dir, f"{base_name}_pos.csv")
        out_file = os.path.join(target_dir, d)
        dfs = []
        for fname in [neg_file, pos_file]:
            if os.path.exists(fname):
                dfs.append(pd.read_csv(fname))
        if dfs:
            pd.concat(dfs, ignore_index=True).to_csv(out_file, index=False)
            # Delete the _pos and _neg files
            for fname in [neg_file, pos_file]:
                if os.path.exists(fname):
                    os.remove(fname)

