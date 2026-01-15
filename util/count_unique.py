import sys
import pandas as pd

def count_unique_text(csv_path):
    df = pd.read_csv(csv_path)
    unique_count = df['Text'].nunique()
    print(f'Unique elements in "Text" column: {unique_count}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_unique.py <path_to_csv>")
        sys.exit(1)
    count_unique_text(sys.argv[1])
