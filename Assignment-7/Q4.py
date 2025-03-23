import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
def load_dataset(file_path):
    return pd.read_excel(file_path)

# Preprocessing data into X (features) and y (target)
def preprocess_data(df):
    X = df.drop(columns=["Label"]).values
    y = df["Label"].values
    return X, y

# Split dataset
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Hyperparameter tuning using RandomizedSearchCV
def tune_hyperparameters(model, param_grid, X_train, y_train, n_iter=20, cv=5):
    if param_grid:
        search = RandomizedSearchCV(model, param_distributions=param_grid,
                                    n_iter=n_iter, cv=cv, n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        return search.best_estimator_
    return model

# Train and evaluate a model
def train_and_evaluate(model, model_name, param_grid, X_train, X_test, y_train, y_test):
    model = tune_hyperparameters(model, param_grid, X_train, y_train)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\n{model_name} Results:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Classification Report (Train):\n", classification_report(y_train, y_train_pred))
    print("Classification Report (Test):\n", classification_report(y_test, y_test_pred))

    return model, train_acc, test_acc

# Define classifiers
def svm_classifier(X_train, X_test, y_train, y_test):
    model = SVC()
    param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    return train_and_evaluate(model, "SVM", param_grid, X_train, X_test, y_train, y_test)

def decision_tree_classifier(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    param_grid = {"max_depth": [5, 10, 20, None]}
    return train_and_evaluate(model, "Decision Tree", param_grid, X_train, X_test, y_train, y_test)

def random_forest_classifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
    return train_and_evaluate(model, "Random Forest", param_grid, X_train, X_test, y_train, y_test)

def xgboost_classifier(X_train, X_test, y_train, y_test):
    model = XGBClassifier()
    param_grid = {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.2]}
    return train_and_evaluate(model, "XGBoost", param_grid, X_train, X_test, y_train, y_test)

def adaboost_classifier(X_train, X_test, y_train, y_test):
    model = AdaBoostClassifier()
    param_grid = {"n_estimators": [50, 100]}
    return train_and_evaluate(model, "AdaBoost", param_grid, X_train, X_test, y_train, y_test)

def naive_bayes_classifier(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    return train_and_evaluate(model, "Naïve Bayes", None, X_train, X_test, y_train, y_test)

def mlp_classifier(X_train, X_test, y_train, y_test):
    model = MLPClassifier(max_iter=500)
    param_grid = {"hidden_layer_sizes": [(50,), (100,), (50, 50)], "activation": ["relu", "tanh"]}
    return train_and_evaluate(model, "MLP", param_grid, X_train, X_test, y_train, y_test)

# Main execution
if __name__ == "__main__":
    file_path = "Judgment_Embeddings_InLegalBERT.xlsx"
    df = load_dataset(file_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Run all classifiers
    svm_classifier(X_train, X_test, y_train, y_test)
    decision_tree_classifier(X_train, X_test, y_train, y_test)
    random_forest_classifier(X_train, X_test, y_train, y_test)
    xgboost_classifier(X_train, X_test, y_train, y_test)
    adaboost_classifier(X_train, X_test, y_train, y_test)
    naive_bayes_classifier(X_train, X_test, y_train, y_test)
    mlp_classifier(X_train, X_test, y_train, y_test)
