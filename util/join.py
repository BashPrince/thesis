import pandas as pd

def prepare_and_save(real_csv_path, synth_csv_path, output_path, n_real, n_synth, real_pos_ratio=None, synth_pos_ratio=None, random_state=None):
    # Load CSVs
    real_df = pd.read_csv(real_csv_path)
    synth_df = pd.read_csv(synth_csv_path)

    if n_real:
        if real_pos_ratio:
            real_df_pos = (real_df[real_df['class_label'] == "Yes"]).sample(n=int(n_real*real_pos_ratio), random_state=random_state).reset_index(drop=True)
            real_df_neg = (real_df[real_df['class_label'] == "No"]).sample(n=int(n_real*(1-real_pos_ratio)), random_state=random_state).reset_index(drop=True)
            real_df = pd.concat([real_df_pos, real_df_neg], ignore_index=True)
        else:
            real_df = real_df.sample(n=n_real, random_state=random_state).reset_index(drop=True)
    if n_synth:
        if synth_pos_ratio:
            synth_df_pos = (synth_df[synth_df['class_label'] == "Yes"]).sample(n=int(n_synth*synth_pos_ratio), random_state=random_state).reset_index(drop=True)
            synth_df_neg = (synth_df[synth_df['class_label'] == "No"]).sample(n=int(n_synth*(1-synth_pos_ratio)), random_state=random_state).reset_index(drop=True)
            synth_df = pd.concat([synth_df_pos, synth_df_neg], ignore_index=True)
        else:
            synth_df = synth_df.sample(n=n_synth, random_state=random_state).reset_index(drop=True)

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
        print(f"Using {len(real_df)} real samples ({len(real_df[real_df['class_label'] == 'Yes']) / len(real_df)} pos ratio)")
    if n_synth:
        dfs.append(synth_df)
        print(f"Using {len(synth_df)} synth samples ({len(synth_df[synth_df['class_label'] == 'Yes']) / len(synth_df)} pos ratio)")
    

    # Concatenate, shuffle, and save
    combined_df = pd.concat(dfs, ignore_index=True)
    shuffled_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    shuffled_df.to_csv(output_path, index=False)

n_steps = 5
dataset_size = 10000

for step in range(n_steps + 1):
    n_real = int(dataset_size * ((n_steps - step) / n_steps))
    n_synth = int(dataset_size * (step / n_steps))
    real_pos_ratio = None
    synth_pos_ratio = 0.23
    real_csv_path = '../data/CT24_checkworthy_english/topic_separation/train_wo_security.csv'
    synth_csv_path = '../data/synthetic/synthetic_security.csv'
    output_path = f'../data/synthetic/train_wo_security_synth_security_{n_steps - step}_{step}.csv'
    prepare_and_save(
        real_csv_path,
        synth_csv_path,
        output_path,
        n_real=n_real,
        n_synth=n_synth,
        real_pos_ratio=real_pos_ratio,
        synth_pos_ratio=synth_pos_ratio,
        random_state=42)
