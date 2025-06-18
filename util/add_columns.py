import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Add empty columns to a CSV file.")
    parser.add_argument("--input_csv", help="Path to input CSV file")
    parser.add_argument("--output_csv", help="Path to output CSV file")
    parser.add_argument("--columns", nargs="+", help="List of column names to add")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    for col in args.columns:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
        df[col] = "-"

    df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
