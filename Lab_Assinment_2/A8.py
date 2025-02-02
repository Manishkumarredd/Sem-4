import pandas as pd
import numpy as np

# Load data
file = pd.read_excel("/content/Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Select first two rows
first_two_vectors = file.iloc[:2]

# Identify binary columns
binary_cols = [
    col
    for col in file.columns
    if file[col].dropna().apply(lambda x: x in (0, 1, 0.0, 1.0)).all()
    and file[col].dropna().nunique() <= 2  # Check for at most 2 unique values
]

binary_data = first_two_vectors[binary_cols]
vector_1 = binary_data.iloc[0].values
vector_2 = binary_data.iloc[1].values

# Compute similarity measures
f11 = np.sum((vector_1 == 1) & (vector_2 == 1))  # Both 1
f00 = np.sum((vector_1 == 0) & (vector_2 == 0))  # Both 0
f10 = np.sum((vector_1 == 1) & (vector_2 == 0))  # (1,0) mismatch
f01 = np.sum((vector_1 == 0) & (vector_2 == 1))  # (0,1) mismatch

JC = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0  # Avoid division by zero
SMC = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0

# Print results
print("\nBinary Attributes Considered:", binary_cols)
print("\nJaccard Coefficient (JC):", round(JC, 4))
print("Simple Matching Coefficient (SMC):", round(SMC, 4))

if JC < SMC:
    print("\n✅ SMC is higher than JC because it considers both matches (1,1 and 0,0).")
    print("✅ JC is useful when we only care about 1s (e.g., features presence).")
else:
    print("\n✅ JC and SMC values are close, meaning the feature vectors are similar.")
