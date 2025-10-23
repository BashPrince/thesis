import argparse
import os
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Append random synthetic rows to sequence CSVs.")
    parser.add_argument('--sequence_dir', required=True, help='Directory containing sequence CSV files')
    parser.add_argument('--synth_file', required=True, help='CSV file with synthetic data')
    parser.add_argument('--num_synth', type=int, required=True, help='Number of synthetic rows to append')
    parser.add_argument('--out_dir', required=True, help='Output directory for modified CSVs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    synth_df = pd.read_csv(args.synth_file)
    if args.num_synth > len(synth_df):
        raise ValueError("num_synth exceeds number of rows in synth_file")

    selected_indices = np.random.choice(synth_df.index, args.num_synth, replace=False)
    synth_rows = synth_df.loc[selected_indices]

    for fname in os.listdir(args.sequence_dir):
        if fname.endswith('.csv'):
            seq_path = os.path.join(args.sequence_dir, fname)
            seq_df = pd.read_csv(seq_path)
            out_df = pd.concat([seq_df, synth_rows], ignore_index=True)
            out_path = os.path.join(args.out_dir, fname)
            out_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
