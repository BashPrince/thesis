import argparse
import os
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Append random synthetic rows to sequence CSVs.")
    parser.add_argument('--sequence_dir', required=True, help='Directory containing sequence CSV files')
    parser.add_argument('--append_file', required=True, help='CSV file with data to append')
    parser.add_argument('--num_append', type=int, required=True, help='Number of rows from append file to append')
    parser.add_argument('--out_dir', required=True, help='Output directory for modified CSVs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    append_df = pd.read_csv(args.append_file)
    if args.num_append > len(append_df):
        raise ValueError("num_append exceeds number of rows in append_file")

    selected_indices = np.random.choice(append_df.index, args.num_append, replace=False)
    append_rows = append_df.loc[selected_indices]

    for fname in os.listdir(args.sequence_dir):
        if fname.endswith('.csv'):
            seq_path = os.path.join(args.sequence_dir, fname)
            seq_df = pd.read_csv(seq_path)
            out_df = pd.concat([seq_df, append_rows], ignore_index=True)
            out_path = os.path.join(args.out_dir, fname)
            out_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
