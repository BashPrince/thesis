import pandas as pd
import os

def process_csv(input_file):
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(input_file)
    
    # Remove the columns "topic" and "tweet_url"
    df = df.drop(columns=["topic", "tweet_url"])
    
    # Rename columns
    df = df.rename(columns={"tweet_id": "Sentence_id", "tweet_text": "Text"})

    # Convert "Sentence_id" column to integer
    df["Sentence_id"] = df["Sentence_id"].astype(int)
    
    # Convert "class_label" values from 0,1 to No, Yes
    df["class_label"] = df["class_label"].map({0: "No", 1: "Yes"})
    
    # Save the modified dataframe to a new file
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}-formatted{ext}"
    df.to_csv(output_file, index=False)
    print(f"Processed file saved to: {output_file}")

# Example usage
process_csv("data/CT22_checkworthy_english/dev-test.csv")
process_csv("data/CT22_checkworthy_english/dev.csv")
process_csv("data/CT22_checkworthy_english/train.csv")
process_csv("data/CT22_checkworthy_english/test-gold.csv")
