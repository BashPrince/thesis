import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Remove rows from first dataframe where comparison column value is present in second dataframe.")
    parser.add_argument("--df1", help="Path to the first dataframe (CSV)")
    parser.add_argument("--df2", help="Path to the second dataframe (CSV)")
    parser.add_argument("--compare", help="Column name to compare")
    parser.add_argument("--out", help="Path to save the resulting dataframe (CSV)")
    args = parser.parse_args()

    df1 = pd.read_csv(args.df1)
    df2 = pd.read_csv(args.df2)

    before_count = len(df1)
    df1_diff = df1[~df1[args.compare].isin(df2[args.compare])]
    after_count = len(df1_diff)
    removed_count = before_count - after_count

    print(f"Removed {removed_count} samples (from {before_count} to {after_count})")

    df1_diff.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
