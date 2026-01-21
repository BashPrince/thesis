import pandas as pd

append_sizes = [800, 1600]

for i in range(5):
    ex2 = pd.read_csv(f"./sequence_{i}/seq_{i}_aug_800.csv")
    append_source = pd.read_csv(f"../experiment_009/sequence_{i}/seq_{i}_aug_3200.csv")
    # Drop the 'context' column if it exists
    if 'context' in append_source.columns:
        append_source = append_source.drop(columns=['context'])
    append_source = append_source[append_source['topic'].notna() & (append_source['topic'] != '')]

    append_source_pos = append_source[append_source["class_label"] == "Yes"]
    append_source_neg = append_source[append_source["class_label"] == "No"]
    
    last_idx = 0
    for sz in append_sizes:
        next_idx = last_idx + sz//2
        append_pos = append_source_pos[last_idx:next_idx]
        append_neg = append_source_neg[last_idx:next_idx]
        append = pd.concat([append_pos, append_neg])
        ex2 = pd.concat([ex2, append], ignore_index=True)
        ex2.to_csv(f"./sequence_{i}/seq_{i}_aug_{sz*2}.csv", index=False)
        last_idx = next_idx

