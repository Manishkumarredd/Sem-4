import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_name = "Judgment_Embeddings_InLegalBERT.xlsx"
df = pd.read_excel(file_name)

# Split features and target
X = df.drop(columns=["Label"]).values
y = df["Label"].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train MLP Classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),  # Single hidden layer with 100 neurons
    activation='relu',          # ReLU activation
    solver='adam',              # Optimizer
    max_iter=1000,              # Maximum iterations
    random_state=42,
    learning_rate_init=0.001    # Learning rate
)

mlp.fit(X_train, y_train)

# Predictions
y_train_pred = mlp.predict(X_train)
y_test_pred = mlp.predict(X_test)

# Compute accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nMLPClassifier Results")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report (Train Set):")
print(classification_report(y_train, y_train_pred))
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))
