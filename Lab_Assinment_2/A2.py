import pandas as pd
import numpy as np

# Load the data
file = pd.read_excel("/content/Lab Session Data.xlsx", sheet_name="Purchase data")

# Select relevant columns and rename them
purchase_data = file.iloc[:, :5].drop(columns=["Customer"])
purchase_data.columns = ["Candies", "Mangoes", "Milk_Packets", "Payment"]

# Prepare matrices
X = np.hstack((np.ones((purchase_data.shape[0], 1)), purchase_data[["Candies", "Mangoes", "Milk_Packets"]].values))
Y = purchase_data["Payment"].values

# Compute the Moore-Penrose pseudoinverse
X_pseudo_inverse = np.linalg.pinv(X)

# Compute model vector
model_vector = X_pseudo_inverse @ Y

# Display results
print("Model Vector (Intercept and Coefficients):")
print(model_vector)
