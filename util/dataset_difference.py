import sys
import pandas as pd
import os

if len(sys.argv) != 3:
    print("Usage: python dataset_difference.py <source_csv> <filter_csv>")
    sys.exit(1)

source_path = sys.argv[1]
filter_path = sys.argv[2]

source = pd.read_csv(source_path)
filter = pd.read_csv(filter_path)

initial_count = len(source)
filtered = source[~source["Text"].isin(filter["Text"])]
filtered_count = len(filtered)

output_path = os.path.join(os.path.dirname(source_path), "source_minus_filter.csv")
filtered.to_csv(output_path, index=False)
print(f"Filtered out {initial_count - filtered_count} rows.")
