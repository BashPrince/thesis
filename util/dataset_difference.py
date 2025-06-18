import sys
import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Remove rows from source CSV that appear in one or more filter CSVs (by 'Text' column).")
    parser.add_argument("source_csv", help="Path to the source CSV file")
    parser.add_argument("filter_csvs", nargs="+", help="One or more filter CSV files")
    args = parser.parse_args()

    source_path = args.source_csv
    filter_paths = args.filter_csvs

    source = pd.read_csv(source_path)
    initial_count = len(source)

    for filter_path in filter_paths:
        filter_df = pd.read_csv(filter_path)
        source = source[~source["Text"].isin(filter_df["Text"])]

    filtered_count = len(source)
    output_path = os.path.join(os.path.dirname(source_path), "source_minus_filter.csv")
    #source.to_csv(output_path, index=False)
    print(f"Filtered out {initial_count - filtered_count} rows.")

if __name__ == "__main__":
    main()
