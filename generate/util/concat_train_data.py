import os
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate datasets.")
    parser.add_argument("--source_data", type=str, help="Path to the source dataset.")
    parser.add_argument("--append_data", type=str, help="Path to the data to append.")
    parser.add_argument("--class_label", type=str, default="Yes", help="Class label to assign to the generated data.")

    args = parser.parse_args()

    # Read the datasets
    source_df = pd.read_csv(args.source_data)
    append_data_df = pd.read_csv(args.append_data)

    # Drop unnecessary columns from the generated data
    append_data_df = append_data_df.drop(columns=['example_Sentence_id', 'example_Text'])

    # Add the Sentence_id column to the generated data
    append_data_df['Sentence_id'] = -1

    # Add the class_label column to the generated data
    append_data_df['class_label'] = args.class_label

    # Concatenate the datasets
    combined_df = pd.concat([source_df, append_data_df], ignore_index=True)

    # Generate the output file name by concatenating the input file names
    source_filename = os.path.splitext(os.path.basename(args.source_data))[0]
    append_filename = os.path.splitext(os.path.basename(args.append_data))[0]
    output_filename = f"../../data/synthetic/{source_filename}_{append_filename}_concat.csv"

    # Save the result to the generated output file name
    combined_df.to_csv(output_filename, index=False)