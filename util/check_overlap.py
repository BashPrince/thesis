import pandas as pd
import os

# Directory containing the first list of CSV files
dir1 = '../generate/sequences/experiment_001'  # <-- update this path as needed

# Get all CSV files in leaf directories
list1 = []
for root, dirs, files in os.walk(dir1):
    if not dirs:  # leaf directory
        list1.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])

# Example variables (replace with your actual lists and column)
list2 = ['../data/CT24_checkworthy_english/test-combined-wo-id.csv']
comparison_column = 'Text'

# Load all dataframes
dfs1 = [pd.read_csv(f) for f in list1]
dfs2 = [pd.read_csv(f) for f in list2]

# Combine all values from the comparison column in list2
values2 = set()
for df in dfs2:
    values2.update(df[comparison_column].dropna().unique())

# Check for overlap in list1
for idx, df in enumerate(dfs1):
    overlap = df[df[comparison_column].isin(values2)]
    if not overlap.empty:
        print(f"Overlap found in {list1[idx]}:")
        print(overlap)

print("Done.")