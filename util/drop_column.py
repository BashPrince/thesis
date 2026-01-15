import sys
import os
import pandas as pd

def drop_sentence_id_column(filepaths):
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        if "Sentence_id" in df.columns:
            df = df.drop(columns=["Sentence_id"])
        base, ext = os.path.splitext(filepath)
        new_filepath = f"{base}-wo-id{ext}"
        df.to_csv(new_filepath, index=False)
        print(f"Saved: {new_filepath}")

if __name__ == "__main__":
    # Usage: python drop_column.py file1.csv file2.csv ...
    if len(sys.argv) < 2:
        print("Usage: python drop_column.py <csv_file1> <csv_file2> ...")
        sys.exit(1)
    drop_sentence_id_column(sys.argv[1:])
