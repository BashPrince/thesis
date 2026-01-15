import argparse
import os
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Subsample sequences from a CSV file.")
    parser.add_argument('--source', required=True, help='Path to source CSV file')
    parser.add_argument('--sizes', required=True, nargs='+', type=int, help='List of increasing integers for subset sizes')
    parser.add_argument('--out_dir', required=True, help='Output directory for subsets')
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    sizes = sorted(args.sizes)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    source_filename = os.path.basename(args.source).replace(".csv", "")

    # Shuffle once for reproducibility
    rng = np.random.default_rng(seed=42)
    df_yes = df[df['class_label'] == 'Yes'].sample(frac=1, random_state=42).reset_index(drop=True)
    df_no = df[df['class_label'] == 'No'].sample(frac=1, random_state=42).reset_index(drop=True)

    min_total = min(len(df_yes) + len(df_no), max(sizes))

    for i, size in enumerate(sizes):
        if size > min_total:
            raise ValueError(f"Requested size {size} exceeds available balanced rows ({min_total})")
        n_yes = size // 2
        n_no = size - n_yes
        # Adjust if not enough samples in either class
        n_yes = min(n_yes, len(df_yes))
        n_no = min(n_no, len(df_no))
        # If not enough for one class, fill with the other
        if n_yes + n_no < size:
            if len(df_yes) > n_yes:
                n_yes += size - (n_yes + n_no)
            elif len(df_no) > n_no:
                n_no += size - (n_yes + n_no)
        subset_df = pd.concat([df_yes.iloc[:n_yes], df_no.iloc[:n_no]], ignore_index=True)
        #subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)
        subset_df['Sentence_id'] = range(len(subset_df))
        out_path = os.path.join(out_dir, f"{source_filename}_{size}.csv")
        subset_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
