import os

root_dir = "/home/stephen/dev/thesis/generate/sequences/experiment_003"

for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".csv") and "_train" in filename:
            old_path = os.path.join(dirpath, filename)
            new_filename = filename.replace("_train", "")
            new_path = os.path.join(dirpath, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
