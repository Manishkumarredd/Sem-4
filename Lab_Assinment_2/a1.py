import pandas as pd
import numpy as np

# Load the data
file = pd.read_excel("/content/Lab Session Data.xlsx", sheet_name="Purchase data")

# Extract matrices
A = file.loc[:, ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = file[['Payment (Rs)']].values

# Compute properties
dim = A.shape[1]
num_vectors = A.shape[0]
rank_A = np.linalg.matrix_rank(A)

# Compute the Moore-Penrose pseudoinverse using SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)
pseudo_inv_A = Vt.T @ np.diag(1 / S) @ U.T

# Compute the cost per unit
cost = pseudo_inv_A @ C

# Display results
print(f"A = {A}")
print(f"C = {C}")
print(f"Dimensionality of vector space: {dim}")
print(f"Number of vectors in vector space: {num_vectors}")
print(f"Rank of matrix A: {rank_A}")
print(f"Pseudoinverse of A: {pseudo_inv_A}")
print(f"Cost per unit of each product: {cost.flatten()}")
