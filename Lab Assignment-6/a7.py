import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def load_legal_dataset(file_path, selected_features):
    """
    Loads the legal dataset and extracts selected features and labels.

    Parameters:
    file_path (str): Path to the dataset file.
    selected_features (list): List of two feature names to be used.

    Returns:
    DataFrame, Series: Selected features (X) and target labels (y).
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

        # Extract selected features and target label
        X = data[selected_features]  # Use only the selected two features
        y = data[label_column]  # Target labels

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

def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary for a trained Decision Tree model.

    Parameters:
    model (DecisionTreeClassifier): Trained Decision Tree model.
    X (DataFrame): Feature matrix with two selected features.
    y (Series): Target labels.

    Returns:
    None
    """
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Predict class for each point in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor="k", cmap=plt.cm.Paired)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title("Decision Boundary of the Decision Tree")
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = "Judgment_Embeddings_InLegalBERT.xlsx"  # Dataset file path

    # Select two features for visualization
    selected_features = ["feature_0", "feature_1"]  # Modify if needed

    # Load dataset with selected features
    X, y = load_legal_dataset(file_path, selected_features)

    if X is not None:
        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Decision Tree
        dt_model = train_decision_tree(X_train, y_train, max_depth=3)

        # Plot Decision Boundary
        plot_decision_boundary(dt_model, X, y)
