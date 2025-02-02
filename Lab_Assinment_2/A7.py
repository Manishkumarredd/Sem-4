import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load data
file = pd.read_excel("/content/Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Identify numeric columns
numeric_cols = file.select_dtypes(include=["int64", "float64"]).columns

# Plot before normalization
plt.figure(figsize=(12, 6))
file[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle("Before Normalization: Distribution of Numeric Attributes")
plt.show()

# Initialize scalers
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Create a copy for normalization
normalized_data = file.copy()

# Apply appropriate scaling
for col in numeric_cols:
    if file[col].skew() < 1:  # Normal distribution → Standard Scaling
        normalized_data[col] = standard_scaler.fit_transform(file[[col]])
    else:  # Skewed distribution → Min-Max Scaling
        normalized_data[col] = minmax_scaler.fit_transform(file[[col]])

print("\nNormalization Completed!")

# Plot after normalization
plt.figure(figsize=(12, 6))
normalized_data[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle("After Normalization: Distribution of Numeric Attributes")
plt.show()