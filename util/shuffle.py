import pandas as pd

def shuffle_csv(input_path, output_path):
    df = pd.read_csv(input_path)
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    df_shuffled.to_csv(output_path, index=False)

# Example usage:
shuffle_csv('../generate/data/pos_neg_prop_test_labelled.csv', '../generate/data/pos_neg_prop_test_labelled.csv')
