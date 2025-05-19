import pandas as pd
import argparse
import os

def split_csv(input_file, ratio):
    # Load the CSV file
    df = pd.read_csv(input_file)
    
    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Parse the ratio
    train_ratio, dev_ratio = map(int, ratio.split(':'))
    total = train_ratio + dev_ratio
    train_size = int(len(df) * (train_ratio / total))
    
    # Split the DataFrame
    train_df = df[:train_size]
    dev_df = df[train_size:]
    
    # Generate output file names
    base, ext = os.path.splitext(input_file)
    train_file = f"{base}_train{ext}"
    dev_file = f"{base}_dev{ext}"
    
    # Save the splits
    train_df.to_csv(train_file, index=False)
    dev_df.to_csv(dev_file, index=False)
    print(f"Train split saved to: {train_file}")
    print(f"Dev split saved to: {dev_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a CSV file into train and dev sets.")
    parser.add_argument("--input_file", help="Path to the input CSV file.")
    parser.add_argument("--ratio", help="Split ratio in the form 'train:dev'.")
    args = parser.parse_args()
    
    split_csv(args.input_file, args.ratio)
