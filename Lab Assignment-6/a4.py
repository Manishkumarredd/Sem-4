import numpy as np
import pandas as pd

def equal_width_binning(data, num_bins=4):
    """
    Performs equal-width binning on a continuous dataset.

    Parameters:
    data (array-like): Continuous numerical values.
    num_bins (int): Number of bins to divide into.

    Returns:
    np.array: Binned categorical labels.
    """
    bins = np.linspace(np.min(data), np.max(data), num_bins + 1)  # Create equal-width bin edges
    binned_data = np.digitize(data, bins, right=True)  # Assign bin numbers
    return binned_data

def equal_frequency_binning(data, num_bins=4):
    """
    Performs equal-frequency binning on a continuous dataset.

    Parameters:
    data (array-like): Continuous numerical values.
    num_bins (int): Number of bins to divide into.

    Returns:
    np.array: Binned categorical labels.
    """
    percentiles = np.linspace(0, 100, num_bins + 1)  # Create percentile-based bins
    bins = np.percentile(data, percentiles)  # Compute bin edges based on percentiles
    binned_data = np.digitize(data, bins, right=True)  # Assign bin numbers
    return binned_data

def bin_continuous_data(data, num_bins=4, binning_type="equal_width"):
    """
    Bins continuous data using specified binning method.

    Parameters:
    data (DataFrame): Continuous dataset.
    num_bins (int): Number of bins (default is 4).
    binning_type (str): "equal_width" or "equal_frequency" (default is "equal_width").

    Returns:
    DataFrame: Binned dataset.
    """
    binned_data = data.copy()
    for column in data.columns:
        if binning_type == "equal_width":
            binned_data[column] = equal_width_binning(data[column], num_bins)
        elif binning_type == "equal_frequency":
            binned_data[column] = equal_frequency_binning(data[column], num_bins)
        else:
            raise ValueError("Invalid binning type! Use 'equal_width' or 'equal_frequency'.")
    return binned_data

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

        # Extract features and target label
        X = data.drop(columns=[label_column])  # Features
        y = data[label_column]  # Target

        return X, y

    except Exception as e:
        print("Error loading dataset:", e)
        return None, None

# Example usage
if __name__ == "__main__":
    file_path = "Judgment_Embeddings_InLegalBERT.xlsx"  # Dataset file path

    # Load dataset
    X, y = load_legal_dataset(file_path)

    if X is not None:
        # Select a smaller feature subset for efficiency
        X_selected = X.iloc[:, :10]  # Use first 10 features for binning

        # Apply binning
        binned_data = bin_continuous_data(X_selected, num_bins=4, binning_type="equal_width")

        # Print preview of binned data
        print("Binned Dataset Preview:\n", binned_data.head())
