import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Subsample positive and negative rows from a CSV.")
    parser.add_argument('--source', required=True, help='Path to source CSV file')
    parser.add_argument('--num_subsets', type=int, required=True, help='Number of non-overlapping subsets')
    parser.add_argument('--num_pos', type=int, required=True, help='Number of positive samples (class_label == Yes)')
    parser.add_argument('--num_neg', type=int, required=True, help='Number of negative samples (class_label == No)')
    parser.add_argument('--out_dir', required=True, help='Path to output directory')
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    pos = df[df['class_label'] == 'Yes'].sample(n=args.num_pos * args.num_subsets, random_state=42, ignore_index=True)
    neg = df[df['class_label'] == 'No'].sample(n=args.num_neg * args.num_subsets, random_state=42, ignore_index=True)

    df_remaining = df[~df['Sentence_id'].isin(pd.concat([pos['Sentence_id'], neg['Sentence_id']]))]

    for i in range(args.num_subsets):
        subset = pd.concat(
            [pos[i*args.num_pos:(i+1)*args.num_pos], neg[i*args.num_neg:(i+1)*args.num_neg]]
            ).sample(frac=1, random_state=42).reset_index(drop=True)
        out_name = os.path.join(args.out_dir, f'subset_{i}_size_{len(subset)}.csv')
        subset.to_csv(out_name, index=False)
    
    df_remaining.to_csv(os.path.join(args.out_dir, f'remaining_size_{len(df_remaining)}.csv'), index=False)

if __name__ == "__main__":
    main()
