import pandas as pd

# Load the dataset
df = pd.read_csv('../../data/ClaimBuster_Datasets/datasets/high_quality_concat.csv')

# Filter rows where 'class_label' is 'No'
no_class_rows = df[df['class_label'] == 'No']

# Randomly sample 2764 rows to remove
rows_to_remove = no_class_rows.sample(n=2808, random_state=42)

# Drop the sampled rows
df = df.drop(rows_to_remove.index)

# Save the modified dataframe to a new file
df.to_csv('../../data/synthetic/high_quality_concat_remove_neg.csv', index=False)
