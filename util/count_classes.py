import pandas as pd

def count_column_values(csv_file_path, column_name):
    """
    Loads a CSV file and counts the occurrences of unique values in a specified column.

    Args:
        csv_file_path (str): Path to the CSV file.
        column_name (str): Name of the column to analyze.

    Returns:
        dict: A dictionary with unique values as keys and their counts as values.
    """
    # Load the CSV file
    data = pd.read_csv(csv_file_path)
    
    # Count occurrences of unique values in the specified column
    value_counts = data[column_name].value_counts().to_dict()
    
    return value_counts

# Example usage
if __name__ == "__main__":
    csv_path = "../data/synthetic/high_quality_concat_fill.csv"  # Replace with your CSV file path
    column = "class_label"  # Replace with the column name you want to analyze
    counts = count_column_values(csv_path, column)
    print(counts)
