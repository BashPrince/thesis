import pandas as pd
import argparse

def count_class_labels(csv_file_path):
    df = pd.read_csv(csv_file_path)
    yes_count = (df['class_label'] == 'Yes').sum()
    no_count = (df['class_label'] == 'No').sum()

    print(f"Rows with 'Yes' in 'class_label': {yes_count}")
    print(f"Rows with 'No' in 'class_label': {no_count}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Count class labels in a CSV file.")
    parser.add_argument("--csv_file_path", type=str, help="Path to the CSV file.")
    args = parser.parse_args()

    count_class_labels(args.csv_file_path)
