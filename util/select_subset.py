import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Select subsets from a CSV file based on class labels.")
parser.add_argument('--file', type=str, required=True, help="Path to the input CSV file.")
parser.add_argument('--pos_count', type=int, required=True, help="Number of rows to select where class_label is 'Yes'.")
parser.add_argument('--neg_count', type=int, required=True, help="Number of rows to select where class_label is 'No'.")
args = parser.parse_args()

# Load the CSV file
df = pd.read_csv(args.file)

# Select subsets
pos_subset = df[df['class_label'] == 'Yes'].sample(n=args.pos_count, random_state=42)
neg_subset = df[df['class_label'] == 'No'].sample(n=args.neg_count, random_state=42)

# Combine subsets
subset = pd.concat([pos_subset, neg_subset])

# Shuffle the combined subset
#subset = subset.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the shuffled subset to a new file
output_file = args.file.replace('.csv', '_subset.csv')
subset.to_csv(output_file, index=False)

print(f"Subset saved to {output_file}")
