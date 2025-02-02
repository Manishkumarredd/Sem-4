import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file = pd.read_excel("/content/Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Display first few rows
print(file.head())

# Data types of each column
print("\nData Types of Each Column:\n", file.dtypes)

# Identify categorical columns
categorical_cols = file.select_dtypes(include=["object"]).columns
print("\nCategorical Columns:\n", categorical_cols)

# Identify numeric columns
numeric_cols = file.select_dtypes(include=["int64", "float64"]).columns
print("\nRange of Numeric Variables:\n", file[numeric_cols].describe().loc[["min", "max"]])

# Missing values count
missing_values = file.isnull().sum()
print("\nMissing Values in Each Attribute:\n", missing_values)

# Boxplot for outlier detection
plt.figure(figsize=(12, 6))
file[numeric_cols].boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot to Detect Outliers")
plt.show()

# Compute mean and standard deviation
mean_values = file[numeric_cols].mean()
std_values = file[numeric_cols].std()
print("\nMean of Numeric Variables:\n", mean_values)
print("\nStandard Deviation of Numeric Variables:\n", std_values)