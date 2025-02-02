import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Load data
file = pd.read_excel("/content/Lab Session Data.xlsx", sheet_name="Purchase data")

# Define customer classification
file["Customer Type"] = ["RICH" if amount > 200 else "POOR" for amount in file["Payment (Rs)"]]

# Select features and target
features = file[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
target = file["Customer Type"].values

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.5, random_state=42)

# Initialize and train classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, Y_train)

# Make predictions
y_pred = knn_classifier.predict(X_test)

# Display classification report
print("Classification Report:")
print(classification_report(Y_test, y_pred))
