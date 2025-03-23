#1.SVM
import pandas as pd
import numpy as np
import sys

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Load dataset with error handling
def load_dataset(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)  # Load first sheet
        df.columns = df.columns.str.strip()  # Clean column names
        if 'Label' not in df.columns:
            raise ValueError("'Label' column not found in dataset.")
        return df
    except Exception as e:
        print("Error loading file:", e)
        sys.exit()

# Handle missing values
def preprocess_data(df):
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    X = df.drop(columns=['Label'])
    y = df['Label'].astype(int)  # Ensure Label is integer
    return X, y

# Train and evaluate SVM classifier
def train_svm(X_train, X_test, y_train, y_test):
    print("Training SVM classifier...")
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\nSVM Classifier Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Main function
def main():
    file_path = "Judgment_Embeddings_InLegalBERT.xlsx"
    df = load_dataset(file_path)
    X, y = preprocess_data(df)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train SVM classifier
    train_svm(X_train, X_test, y_train, y_test)

# Run the script
if __name__ == "__main__":
    main()
#2. Decision tree classifer
import pandas as pd
import numpy as np
import sys

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Load dataset with error handling
def load_dataset(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)  # Load first sheet
        df.columns = df.columns.str.strip()  # Clean column names
        if 'Label' not in df.columns:
            raise ValueError("'Label' column not found in dataset.")
        return df
    except Exception as e:
        print("Error loading file:", e)
        sys.exit()

# Handle missing values
def preprocess_data(df):
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    X = df.drop(columns=['Label'])
    y = df['Label'].astype(int)  # Ensure Label is integer
    return X, y

# Train and evaluate Decision Tree classifier
def train_decision_tree(X_train, X_test, y_train, y_test):
    print("Training Decision Tree classifier...")
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\nDecision Tree Classifier Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Main function
def main():
    file_path = "Judgment_Embeddings_InLegalBERT.xlsx"
    df = load_dataset(file_path)
    X, y = preprocess_data(df)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train Decision Tree classifier
    train_decision_tree(X_train, X_test, y_train, y_test)

# Run the script
if __name__ == "__main__":
    main()
#3.Random forest classifer
import pandas as pd
import numpy as np
import sys

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Load dataset with error handling
def load_dataset(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)  # Load first sheet
        df.columns = df.columns.str.strip()  # Clean column names
        if 'Label' not in df.columns:
            raise ValueError("'Label' column not found in dataset.")
        return df
    except Exception as e:
        print("Error loading file:", e)
        sys.exit()

# Handle missing values
def preprocess_data(df):
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    X = df.drop(columns=['Label'])
    y = df['Label'].astype(int)  # Ensure Label is integer
    return X, y

# Train and evaluate Random Forest classifier
def train_random_forest(X_train, X_test, y_train, y_test):
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\nRandom Forest Classifier Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Main function
def main():
    file_path = "Judgment_Embeddings_InLegalBERT.xlsx"
    df = load_dataset(file_path)
    X, y = preprocess_data(df)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train Random Forest classifier
    train_random_forest(X_train, X_test, y_train, y_test)

# Run the script
if __name__ == "__main__":
    main()
  #5.AdaBoost
import pandas as pd
import numpy as np
import sys

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Load dataset with error handling
def load_dataset(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)  # Load first sheet
        df.columns = df.columns.str.strip()  # Clean column names
        if 'Label' not in df.columns:
            raise ValueError("'Label' column not found in dataset.")
        return df
    except Exception as e:
        print("Error loading file:", e)
        sys.exit()

# Handle missing values
def preprocess_data(df):
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    X = df.drop(columns=['Label'])
    y = df['Label'].astype(int)  # Ensure Label is integer
    return X, y

# Train and evaluate AdaBoost classifier
def train_adaboost(X_train, X_test, y_train, y_test):
    print("Training AdaBoost classifier...")
    model = AdaBoostClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\nAdaBoost Classifier Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Main function
def main():
    file_path = "Judgment_Embeddings_InLegalBERT.xlsx"
    df = load_dataset(file_path)
    X, y = preprocess_data(df)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train AdaBoost classifier
    train_adaboost(X_train, X_test, y_train, y_test)

# Run the script
if __name__ == "__main__":
    main()
#6. XG Boost
import pandas as pd
import numpy as np
import sys

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Load dataset with error handling
def load_dataset(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)  # Load first sheet
        df.columns = df.columns.str.strip()  # Clean column names
        if 'Label' not in df.columns:
            raise ValueError("'Label' column not found in dataset.")
        return df
    except Exception as e:
        print("Error loading file:", e)
        sys.exit()

# Handle missing values
def preprocess_data(df):
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    X = df.drop(columns=['Label'])
    y = df['Label'].astype(int)  # Ensure Label is integer
    return X, y

# Train and evaluate XGBoost classifier
def train_xgboost(X_train, X_test, y_train, y_test):
    print("Training XGBoost classifier...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\nXGBoost Classifier Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Main function
def main():
    file_path = "Judgment_Embeddings_InLegalBERT.xlsx"
    df = load_dataset(file_path)
    X, y = preprocess_data(df)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train XGBoost classifier
    train_xgboost(X_train, X_test, y_train, y_test)

# Run the script
if __name__ == "__main__":
    main()
#7.Naïve-Bayes & MLP
import pandas as pd
import numpy as np
import sys

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Load dataset with error handling
def load_dataset(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)  # Load first sheet
        df.columns = df.columns.str.strip()  # Clean column names
        if 'Label' not in df.columns:
            raise ValueError("'Label' column not found in dataset.")
        return df
    except Exception as e:
        print("Error loading file:", e)
        sys.exit()

# Handle missing values
def preprocess_data(df):
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    X = df.drop(columns=['Label'])
    y = df['Label'].astype(int)  # Ensure Label is integer
    return X, y

# Train and evaluate Naïve Bayes classifier
def train_naive_bayes(X_train, X_test, y_train, y_test):
    print("Training Naïve Bayes classifier...")
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\nNaïve Bayes Classifier Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Main function
def main():
    file_path = "Judgment_Embeddings_InLegalBERT.xlsx"
    df = load_dataset(file_path)
    X, y = preprocess_data(df)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train Naïve Bayes classifier
    train_naive_bayes(X_train, X_test, y_train, y_test)

# Run the script
if __name__ == "__main__":
    main()

