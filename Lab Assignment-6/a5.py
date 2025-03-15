import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

def train_decision_tree(X_train, y_train, max_depth=3):
    """
    Trains a Decision Tree classifier.

    Parameters:
    X_train (DataFrame): Training feature matrix.
    y_train (Series): Training target labels.
    max_depth (int): Maximum depth of the tree.

    Returns:
    DecisionTreeClassifier: Trained Decision Tree model.
    """
    model = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the Decision Tree model on the test set.

    Parameters:
    model (DecisionTreeClassifier): Trained Decision Tree model.
    X_test (DataFrame): Test feature matrix.
    y_test (Series): Test target labels.

    Returns:
    float: Accuracy score of the model.
    """
    y_pred = model.predict(X_test)  # Predict test labels
    accuracy = accuracy_score(y_test, y_pred)  # Compute accuracy
    return accuracy

# Example usage
if __name__ == "__main__":
    file_path = "Judgment_Embeddings_InLegalBERT.xlsx"  # Dataset file path

    # Load dataset
    X, y = load_legal_dataset(file_path)

    if X is not None:
        # Select a smaller subset of features for efficiency
        X_selected = X.iloc[:, :10]  # Using first 10 features

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        # Train Decision Tree
        dt_model = train_decision_tree(X_train, y_train, max_depth=5)

        # Evaluate model
        accuracy = evaluate_model(dt_model, X_test, y_test)

        print("Decision Tree Model Accuracy:", accuracy)
