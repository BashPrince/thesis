import pandas as pd

# Load the CSV file
df = pd.read_csv('FactSpan_annotated.csv')

# Filter for rows where language is 'en'
df_en = df[df['language'] == 'en']

# Filter out rows where 'claim' contains any of the specified words (case-insensitive)
exclude_words = ['video', 'picture', 'image', 'photo', 'clip', 'audio']
pattern = '|'.join(exclude_words)
df_en = df_en[~df_en['claim'].str.contains(pattern, case=False, na=False)]

# Save the filtered DataFrame to a new CSV file
df_en.to_csv('FactSpan_en.csv', index=False)
