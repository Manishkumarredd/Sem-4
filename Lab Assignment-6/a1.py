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
    Calculates entropy for the given class labels.

    Parameters:
    y (array-like): Array of categorical class labels.

    Returns:
    float: Entropy value.
    """
    probabilities = compute_probabilities(y)  # Get class probabilities
    entropy_value = -np.sum([p * np.log2(p) for p in probabilities if p > 0])  # Entropy formula
    return entropy_value

def load_legal_dataset(file_path):
    """
    Loads the legal dataset and extracts the target labels.

    Parameters:
    file_path (str): Path to the dataset file.

    Returns:
    Series: Target labels (y).
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

        return data[label_column]  # Extract target labels

    except Exception as e:
        print("Error loading dataset:", e)
        return None

# Example usage
if __name__ == "__main__":
    file_path = "Judgment_Embeddings_InLegalBERT.xlsx"  # Dataset file path

    # Load target labels
    y_labels = load_legal_dataset(file_path)

    if y_labels is not None:
        # Compute entropy
        entropy_result = calculate_entropy(y_labels)

        # Print the result
        print("Entropy of Label column:", entropy_result)
