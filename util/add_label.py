import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Add a label column to a CSV file.")
    parser.add_argument("--input_csv", help="Path to input CSV file")
    parser.add_argument("--output_csv", help="Path to output CSV file")
    parser.add_argument("--label_value", help="Value to assign to the label column")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df["class_label"] = args.label_value
    df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
