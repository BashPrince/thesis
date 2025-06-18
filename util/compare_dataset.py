import pandas as pd

def compare_text_column(csv1_path, csv2_path):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    if len(df1) != len(df2):
        print(f"Warning: CSV files have different number of rows ({len(df1)} vs {len(df2)})")

    min_len = min(len(df1), len(df2))
    for i in range(min_len):
        text1 = df1.loc[i, "Text"]
        text2 = df2.loc[i, "Text"]
        label1 = df1.loc[i, "class_label"]
        label2 = df2.loc[i, "class_label"]
        if text1 != text2 or label1 != label2:
            print(f"Row {i}: Different")

# Example usage:
compare_text_column('../ct24_synth_0_10k_train.csv', '../gc_synth_0_10k_train.csv')
