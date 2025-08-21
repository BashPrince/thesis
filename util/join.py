import pandas as pd

def prepare_and_save(real_csv_path, synth_csv_path, output_path, n_real, n_synth, random_state=None):
    # Load CSVs
    real_df = pd.read_csv(real_csv_path)
    synth_df = pd.read_csv(synth_csv_path)

    # Optionally undersample real and synthetic data
    if n_real:
        real_df_pos = (real_df[real_df['class_label'] == "Yes"]).sample(n=n_real // 2, random_state=random_state).reset_index(drop=True)
        real_df_neg = (real_df[real_df['class_label'] == "No"]).sample(n=n_real // 2, random_state=random_state).reset_index(drop=True)
        real_df = pd.concat([real_df_pos, real_df_neg], ignore_index=True)
    if n_synth:
        synth_df_pos = (synth_df[synth_df['class_label'] == "Yes"]).sample(n=n_synth // 2, random_state=random_state).reset_index(drop=True)
        synth_df_neg = (synth_df[synth_df['class_label'] == "No"]).sample(n=n_synth // 2, random_state=random_state).reset_index(drop=True)
        synth_df = pd.concat([synth_df_pos, synth_df_neg], ignore_index=True)

    # Add 'Sentence_id' to synth_df
    max_sentence_id = real_df['Sentence_id'].max() if len(real_df) > 0 else 0
    synth_df['Sentence_id'] = range(max_sentence_id + 1, max_sentence_id + 1 + len(synth_df))

    # Add 'context', 'properties' and 'violation' to real_df, leave empty
    real_df['context'] = ''
    real_df['properties'] = ''
    
    if 'violation' in synth_df.columns:
        real_df['violation'] = ''

    # Add 'synthetic' column
    real_df['synthetic'] = 'No'
    synth_df['synthetic'] = 'Yes'

    dfs = []
    if n_real:
        dfs.append(real_df)
    if n_synth:
        dfs.append(synth_df)

    # Concatenate, shuffle, and save
    combined_df = pd.concat(dfs, ignore_index=True)
    shuffled_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    shuffled_df.to_csv(output_path, index=False)

n_real = 9000
n_synth = 1000
real_csv_path = '../data/CT24_checkworthy_english/train.csv'
synth_csv_path = '../data/synthetic/synthetic_corona_1k.csv'
output_path = f'../data/synthetic/ct24_synthetic_corona_{n_real}_{n_synth}.csv'
prepare_and_save(real_csv_path, synth_csv_path, output_path, n_real=n_real, n_synth=n_synth, random_state=42)
