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


def filter_dynamics(df, mode, metric, count, balance):
    """
    Select a fractional subset of the DataFrame based on a specified metric.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data to filter.
        mode (str): Selection mode, either 'top' to select the highest values 
                    or 'bottom' to select the lowest values based on the metric.
        metric (str): Column name in the DataFrame to use for sorting and selection.
        count (int or None): Fixed number of elements to randomly sample from the 
                             selected subset. If None, all selected rows are returned.
        balance (bool): If True, balance the dataset by selecting an equal number of
                        positive and negative samples based on the 'class_label' column.

    Returns:
        pd.DataFrame: Subset of the DataFrame based on the specified criteria.
    """

    if balance:
        df_yes = df[df['class_label'] == 'Yes']
        df_no = df[df['class_label'] == 'No']

        if mode == "top":
            df_yes = df_yes.nlargest(count // 2, metric)
            df_no = df_no.nlargest(count // 2, metric)
        elif mode == "bottom":
            df_yes = df_yes.nsmallest(count // 2, metric)
            df_no = df_no.nsmallest(count // 2, metric)

        subset = pd.concat([df_yes, df_no])
        subset = subset.sample(frac=1).reset_index(drop=True)
    else:
        if mode == "top":
            subset = df.nlargest(count, metric)
        elif mode == "bottom":
            subset = df.nsmallest(count, metric)

    return subset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dynamics and training data.")
    parser.add_argument("--dynamics_data", type=str, required=True, help="Path to the dynamics .jsonl file")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data CSV file")
    parser.add_argument("--mode", type=str, required=True, help="Selection mode ('top' or 'bottom')")
    parser.add_argument("--metric", type=str, required=True, help="Metric to use for selection ('variability' or 'confidence')")
    parser.add_argument("--sample_count", type=int, required=True, help="Select a fixed number of elements")
    parser.add_argument("--label", type=str, default=None, required=False, help="Label to filter the training data (Yes or No)")
    parser.add_argument("--synthetic", action='store_true', help="If only synthetic data should be selected")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the filtered train_subset DataFrame")
    parser.add_argument("--balance", action='store_true', help="Flag to balance the dataset")
    args = parser.parse_args()

    if args.label and args.balance:
        raise ValueError("Cannot use both --label and --balance options together. Please choose one.")
    
    if args.mode not in ['top', 'bottom']:
        raise ValueError("Mode must be either 'top' or 'bottom'.")
    
    # Load the dynamics DataFrame
    dynamics_df = load_jsonl_to_dataframe(args.dynamics_data)

    if args.metric not in dynamics_df.columns:
        raise ValueError(f"The DataFrame must contain a '{args.metric}' column.")

    # Load the training data DataFrame
    train_df = pd.read_csv(args.train_data)

    # Merge the labels and synthetic info (if present) into the dynamics DataFrame
    if args.synthetic:
        if 'synthetic' not in train_df.columns:
            raise ValueError("The dynamics DataFrame must contain a 'synthetic' column.")
        merge_columns = ['Sentence_id', 'class_label', 'synthetic']
    else:
        merge_columns = ['Sentence_id', 'class_label']
        
    labels_df = train_df[merge_columns].copy()
    labels_df.rename(columns={'Sentence_id': 'guid'}, inplace=True)
    dynamics_df = pd.merge(dynamics_df, labels_df, on='guid', how='left')

    # Filter the dynamics DataFrame based on the specified label if provided
    if args.label:
        if args.label not in ['Yes', 'No']:
            raise ValueError("Label must be either 'Yes' or 'No'.")
    
        dynamics_df = dynamics_df[dynamics_df['class_label'] == args.label]
    
    # If synthetic flag is set, filter the dynamics DataFrame to include only synthetic data
    if args.synthetic:
        dynamics_df = dynamics_df[dynamics_df['synthetic'] == 1]

    # Filter the dynamics
    dynamics_df = filter_dynamics(dynamics_df, args.mode, args.metric, args.sample_count, args.balance)

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