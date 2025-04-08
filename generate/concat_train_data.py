import os
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate generated data with the CheckThat dataset.")
    parser.add_argument("--checkthat_data", type=str, help="Path to the CheckThat dataset.")
    parser.add_argument("--gen_data", type=str, help="Path to the generated data.")

    args = parser.parse_args()

    # Read the datasets
    checkthat_df = pd.read_csv(args.checkthat_data)
    gen_data_df = pd.read_csv(args.gen_data)

    # Drop unnecessary columns from the generated data
    gen_data_df = gen_data_df.drop(columns=['example_Sentence_id', 'example_Text'])

    # Find the maximum Sentence_id in the CheckThat dataset
    max_sentence_id = checkthat_df['Sentence_id'].max()

    # Add the Sentence_id column to the generated data
    gen_data_df['Sentence_id'] = range(max_sentence_id + 1, max_sentence_id + 1 + len(gen_data_df))

    # Add the class_label column to the generated data
    gen_data_df['class_label'] = 'Yes'

    # Concatenate the datasets
    combined_df = pd.concat([checkthat_df, gen_data_df], ignore_index=True)

    # Generate the output file name by concatenating the input file names
    checkthat_filename = os.path.splitext(os.path.basename(args.checkthat_data))[0]
    gen_data_filename = os.path.splitext(os.path.basename(args.gen_data))[0]
    output_filename = f"../data/synthetic/{checkthat_filename}_{gen_data_filename}_concat.csv"

    # Save the result to the generated output file name
    combined_df.to_csv(output_filename, index=False)