import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Handle missing values (fill with mean for simplicity)
df = df.fillna(df.mean())

# Define features and target
X = df.drop(columns=['Label'])  # All columns except 'Label'
y = df['Label']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier with better logging
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
