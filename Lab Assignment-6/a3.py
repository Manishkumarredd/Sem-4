import numpy as np
import pandas as pd

def compute_probabilities(y):
    """
    Computes the probability distribution of class labels.

    Parameters:
    y (array-like): Array of categorical class labels.

    Returns:
    np.array: Probability of each class.
    """
    unique_classes, counts = np.unique(y, return_counts=True)  # Count occurrences
    probabilities = counts / len(y)  # Compute probabilities
    return probabilities

def calculate_entropy(y):
    """
    Calculates entropy for a given dataset.

    Parameters:
    y (array-like): Array of categorical class labels.

    Returns:
    float: Entropy value.
    """
    probabilities = compute_probabilities(y)  # Get class probabilities
    entropy_value = -np.sum([p * np.log2(p) for p in probabilities if p > 0])  # Entropy formula
    return entropy_value

def compute_information_gain(data, target_column, feature_column):
    """
    Computes the Information Gain for a given feature.

    Parameters:
    data (DataFrame): The dataset.
    target_column (str): Name of the target column.
    feature_column (str): Name of the feature column.

    Returns:
    float: Information Gain value.
    """
    total_entropy = calculate_entropy(data[target_column])  # Compute dataset entropy
    feature_values, counts = np.unique(data[feature_column], return_counts=True)  # Get unique feature values

    weighted_entropy = 0
    for value, count in zip(feature_values, counts):
        subset = data[data[feature_column] == value]  # Subset data where feature == value
        subset_entropy = calculate_entropy(subset[target_column])  # Compute entropy of subset
        weighted_entropy += (count / len(data)) * subset_entropy  # Compute weighted entropy

    information_gain = total_entropy - weighted_entropy  # Compute IG
    return information_gain

def find_best_feature(data, target_column):
    """
    Identifies the best feature to split on based on Information Gain.

    Parameters:
    data (DataFrame): The dataset containing features and target.
    target_column (str): Name of the target column.

    Returns:
    str: Name of the best feature (root node).
    """
    feature_columns = [col for col in data.columns if col != target_column]  # Exclude target column
    info_gains = {feature: compute_information_gain(data, target_column, feature) for feature in feature_columns}  # Compute IG for each feature
    best_feature = max(info_gains, key=info_gains.get)  # Select feature with max IG
    return best_feature

def load_legal_dataset(file_path):
    """
    Loads the legal dataset and extracts features and labels.

    Parameters:
    file_path (str): Path to the dataset file.

    Returns:
    DataFrame: Features (X) and target labels (y).
    """
    try:
        # Load dataset
        data = pd.read_excel(file_path)

        # Print column names to check if "Label" exists
        print("Columns in dataset:", data.columns)

        # Handling possible column name mismatches
        possible_label_names = ["Label", "label", "Class", "Target"]
        label_column = None
        for col in data.columns:
            if col.strip() in possible_label_names:  # Checking for minor name variations
                label_column = col
                break

        # If label column not found, raise an error
        if label_column is None:
            raise ValueError("Error: Could not find the target column ('Label') in the dataset!")

        return data, label_column  # Return full dataset and detected label column

    except Exception as e:
        print("Error loading dataset:", e)
        return None, None

# Example usage
if __name__ == "__main__":
    file_path = "Judgment_Embeddings_InLegalBERT.xlsx"  # Dataset file path

    # Load dataset and find target column
    dataset, target_column = load_legal_dataset(file_path)

    if dataset is not None:
        # Select a small subset of features for performance (Decision Tree works best with fewer features)
        dataset_sample = dataset.iloc[:, :10]  # Use first 10 features for efficiency
        dataset_sample[target_column] = dataset[target_column]  # Keep target column

        # Identify best feature for root node
        best_root_feature = find_best_feature(dataset_sample, target_column)

        print("Best Feature (Root Node):", best_root_feature)
