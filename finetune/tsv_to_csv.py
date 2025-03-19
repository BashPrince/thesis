import pandas as pd
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python tsv_to_csc.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tsv'):
            tsv_file_path = os.path.join(folder_path, file_name)
            csv_file_path = os.path.join(folder_path, file_name.replace('.tsv', '.csv'))

            try:
                csv_table = pd.read_table(tsv_file_path, sep='\t')
                csv_table.to_csv(csv_file_path, index=False)
                print(f"Converted {tsv_file_path} to {csv_file_path}")
            except Exception as e:
                print(f"Failed to convert {tsv_file_path}: {e}")