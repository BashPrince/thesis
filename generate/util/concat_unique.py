import pandas as pd
import argparse

def load_and_process_csvs(input_files, output_file):
    # Load all CSV files into dataframes
    dataframes = [pd.read_csv(file) for file in input_files]
    
    # Concatenate all dataframes
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    
    # Remove duplicate rows based on 'Sentence_id'
    unique_df = concatenated_df.drop_duplicates(subset='Sentence_id')
    
    # Write the resulting dataframe to the output file
    unique_df.to_csv(output_file, index=False)

    # Count positive and negative class labels
    positive_count = unique_df[unique_df['class_label'] == 'Yes'].shape[0]
    negative_count = unique_df[unique_df['class_label'] == 'No'].shape[0]
    
    print(f"Number of positive class labels (Yes): {positive_count}")
    print(f"Number of negative class labels (No): {negative_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate CSV files and remove duplicates.")
    parser.add_argument(
        "input_files", 
        nargs='+', 
        help="Paths to the input CSV files."
    )
    parser.add_argument(
        "output_file", 
        help="Path to save the resulting CSV file."
    )
    args = parser.parse_args()
    
    load_and_process_csvs(args.input_files, args.output_file)
