import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from scipy.stats import randint

# Load dataset with error handling
file_path = "Judgment_Embeddings_InLegalBERT.xlsx"
try:
    df = pd.read_excel(file_path, sheet_name=0)  # Load first sheet
except Exception as e:
    print("Error loading file:", e)
    exit()

# Ensure column names are clean
df.columns = df.columns.str.strip()

# Check if 'Label' exists in the dataset
if 'Label' not in df.columns:
    print("Error: 'Label' column not found in dataset.")
    exit()

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[df.columns] = imputer.fit_transform(df)

# Define features and target
X = df.drop(columns=['Label'])  # Feature variables
y = df['Label']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define hyperparameter grid for Decision Tree
param_dist = {
    "max_depth": randint(3, 20),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 10),
    "criterion": ["gini", "entropy"]
}

# Initialize Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)

# Perform Randomized Search Cross Validation
random_search = RandomizedSearchCV(dt, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1, verbose=1)
random_search.fit(X_train, y_train)

# Get the best model
best_model = random_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print best parameters and results
print("\nBest Parameters:", random_search.best_params_)
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

