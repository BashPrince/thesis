import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate datasets.")
    parser.add_argument("--real_data", nargs='*', type=str, help="Path to the source dataset.")
    parser.add_argument("--gen_data", nargs='*', type=str, help="Path to the generated data.")
    parser.add_argument("--gen_data_labels", nargs='*', type=str, help="List of labels ('Yes' or 'No') for each file provided in --gen_data.")
    parser.add_argument("--out", type=str, required=True, help="Out file path.")

    args = parser.parse_args()

    if args.real_data is None:
        args.real_data = []
    if args.gen_data is None:
        args.gen_data = []
    if args.gen_data_labels is None:
        args.gen_data_labels = []

    # Check if the number of labels matches the number of generated data files
    if len(args.gen_data_labels) != len(args.gen_data):
        raise ValueError("The number of labels must match the number of generated data files.")
    # Check if the labels are valid
    for label in args.gen_data_labels:
        if label not in ['Yes', 'No']:
            raise ValueError("Labels must be either 'Yes' or 'No'.")
    
    if not args.real_data and not args.gen_data:
        raise ValueError("No data provided.")

    # Read the datasets
    real_dfs = [pd.read_csv(df) for df in args.real_data]
    gen_data_dfs = [pd.read_csv(df) for df in args.gen_data]

    # Drop unnecessary columns from the generated data
    if 'example_Text' in gen_data_dfs[0].columns:
        gen_data_dfs = [df.drop(columns=['example_Text']) for df in gen_data_dfs]

        # Add an example_Sentence_id column to the real data with None
        for df in real_dfs:
            df['example_Sentence_id'] = None

    # Add the Sentence_id column to the generated data starting from 1000000
    start_id = 1000000
    for df in gen_data_dfs:
        df['Sentence_id'] = range(start_id, start_id + len(df))
        start_id += len(df)

    # Add the class_label column to the generated data
    for df, label in zip(gen_data_dfs, args.gen_data_labels):
        df['class_label'] = label
    
    # Add the synthetic column to the real and generated data
    for df in gen_data_dfs:
        df['synthetic'] = 1

    for df in real_dfs:
        df['synthetic'] = 0

    # Concatenate the datasets and shuffle the rows
    combined_df = pd.concat(real_dfs + gen_data_dfs, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the result to the generated output file name
    combined_df.to_csv(args.out, index=False)