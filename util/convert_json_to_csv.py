import json
import pandas as pd

# Load the JSON file
with open('../data/ClaimBuster_Datasets/datasets/3xNCS.json', 'r') as file:
    data = json.load(file)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Rename columns
df = df.rename(columns={
    'sentence_id': 'Sentence_id',
    'text': 'Text',
    'label': 'class_label'
})

# Reorder columns
df = df[['Sentence_id', 'Text', 'class_label']]

# Convert labels
df['class_label'] = df['class_label'].map({1: 'Yes', 0: 'No'})

# Save to CSV
df.to_csv('../data/ClaimBuster_Datasets/datasets/3xNCS.csv', index=False)
