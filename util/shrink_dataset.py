import argparse
import os
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Shrink dataset by halving size at each step.")
    parser.add_argument("--csv_path", type=str, help="Path to input CSV file")
    parser.add_argument("--min_size", type=int, help="Minimum dataset size")
    parser.add_argument("--max_size", type=int, help="Maximum dataset size")
    parser.add_argument("--num_sequences", type=int, help="Number of dataset sequences to create")
    parser.add_argument("--out_dir", type=str, help="Output directory for shrunken datasets")
    parser.add_argument("--out_name", type=str, help="Output name prefix")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    os.makedirs(args.out_dir, exist_ok=True)

    for sequence in range(args.num_sequences):
        # Randomly sample max_size rows for this variant
        if args.max_size > len(df):
            raise ValueError("max_size exceeds number of rows in CSV")
        variant_df = df.sample(n=args.max_size, random_state=sequence).reset_index(drop=True)
        current_size = args.max_size
        step = 0
        while current_size >= args.min_size:
            out_path = os.path.join(
                args.out_dir,
                f"{args.out_name}_seq_{sequence}_size_{current_size}.csv"
            )
            variant_df.iloc[:current_size].to_csv(out_path, index=False)
            # Halve the size for next step
            current_size = current_size // 2
            step += 1

if __name__ == "__main__":
    main()
