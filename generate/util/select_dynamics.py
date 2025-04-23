import pandas as pd
import json
import argparse

def load_jsonl_to_dataframe(filepath):
    """
    Load a .jsonl file into a pandas DataFrame.

    Args:
        filepath (str): Path to the .jsonl file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    with open(filepath, 'r') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)


def filter_dynamics(df, mode, metric, frac, count):
    """
    Select a fractional subset of the DataFrame based on a specified metric.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data to filter.
        mode (str): Selection mode, either 'top' to select the highest values 
                    or 'bottom' to select the lowest values based on the metric.
        metric (str): Column name in the DataFrame to use for sorting and selection.
        frac (float): Fraction of rows to select (0 < frac <= 1).
        count (int or None): Fixed number of elements to randomly sample from the 
                             selected subset. If None, all selected rows are returned.

    Returns:
        pd.DataFrame: Subset of the DataFrame based on the specified criteria.
    """
    if metric not in df.columns:
        raise ValueError(f"The DataFrame must contain a '{metric}' column.")
    if not (0 < frac <= 1):
        raise ValueError("The fraction must be between 0 and 1.")

    if mode == "top":
        subset = df.nlargest(int(len(df) * frac), metric)
    elif mode == "bottom":
        subset = df.nsmallest(int(len(df) * frac), metric)
    else:
        raise ValueError("Mode must be either 'top' or 'bottom'.")

    if count:
        # Randomly sample a fixed number of elements from the top subset
        return subset.sample(n=count, random_state=1)

    return subset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dynamics and training data.")
    parser.add_argument("--dynamics_data", type=str, required=True, help="Path to the dynamics .jsonl file")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data CSV file")
    parser.add_argument("--mode", type=str, required=True, help="Selection mode ('top' or 'bottom')")
    parser.add_argument("--metric", type=str, required=True, help="Metric to use for selection ('variability' or 'confidence')")
    parser.add_argument("--frac", type=float, required=True, help="Fraction of elements to select (0 < frac <= 1)")
    parser.add_argument("--sample_count", type=int, default=None, required=False, help="Select a fixed number of elements randomly sampled from the top/bottom subset")
    parser.add_argument("--label", type=str, default=None, required=False, help="Label to filter the training data (Yes or No)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the filtered train_subset DataFrame")
    args = parser.parse_args()

    # Load the dynamics DataFrame
    dynamics_df = load_jsonl_to_dataframe(args.dynamics_data)

    # Load the training data DataFrame
    train_df = pd.read_csv(args.train_data)

    # Filter the dynamics DataFrame based on the specified label if provided
    if args.label:
        if args.label not in ['Yes', 'No']:
            raise ValueError("Label must be either 'Yes' or 'No'.")
    
        train_filtered_label_df = train_df[train_df['class_label'] == args.label]
        dynamics_df = dynamics_df[dynamics_df['guid'].isin(train_filtered_label_df['Sentence_id'].values)]

    # Filter the dynamics
    dynamics_df = filter_dynamics(dynamics_df, args.mode, args.metric, args.frac, args.sample_count)

    # Filter train_df to include only rows with 'Sentence_id' matching 'guid' in dynamics_df
    train_subset_df = train_df[train_df['Sentence_id'].isin(dynamics_df['guid'])]

    # Count the number of positive ('Yes') and negative ('No') elements in the 'class_label' column
    positive_count = train_subset_df[train_subset_df['class_label'] == 'Yes'].shape[0]
    negative_count = train_subset_df[train_subset_df['class_label'] == 'No'].shape[0]

    # Print the counts
    print(f"Positive ('Yes') count: {positive_count}")
    print(f"Negative ('No') count: {negative_count}")

    # Save the filtered train_subset DataFrame to the specified path
    train_subset_df.to_csv(args.save_path, index=False)