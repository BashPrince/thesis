import pandas as pd
import os

def process_csv(input_file):
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(input_file)
    
    # Filter rows where the "lang" column equals "en"
    df = df[df["lang"] == "en"]

    # Remove the column "lang"
    df = df.drop(columns=["lang"])
    
    # Rename columns
    df = df.rename(columns={"Unnamed: 0": "Sentence_id", "text": "Text", "label": "class_label"})

    # Convert "Sentence_id" column to integer
    df["Sentence_id"] = df["Sentence_id"].astype(int)
    
    # Convert "class_label" values from 0,1 to No, Yes
    df["class_label"] = df["class_label"].map({0: "No", 1: "Yes"})

    # Shuffle the dataframe rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the modified dataframe to a new file
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}-formatted{ext}"
    df.to_csv(output_file, index=False)
    print(f"Processed file saved to: {output_file}")

# Example usage
process_csv("../data/general_claim/general-claim.csv")
