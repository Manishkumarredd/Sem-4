import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file = pd.read_excel("/content/Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Check for missing values
print("\nMissing Values Before Imputation:\n", file.isnull().sum())

# Identify numeric and categorical columns
numeric_cols = file.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = file.select_dtypes(include=["object"]).columns
print("\nNumeric Columns:", numeric_cols)
print("\nCategorical Columns:", categorical_cols)

# Boxplot for outlier detection
plt.figure(figsize=(12, 6))
file[numeric_cols].boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot to Detect Outliers")
plt.show()

# Handle missing values in numeric columns
for col in numeric_cols:
    if file[col].isnull().sum() > 0:  # Check if there are missing values
        if file[col].skew() < 1:  # No strong skewness, assume no outliers
            file[col].fillna(file[col].mean(), inplace=True)  # Use Mean
        else:
            file[col].fillna(file[col].median(), inplace=True)  # Use Median for outliers

print("\nMissing Values After Numeric Imputation:\n", file.isnull().sum())

# Handle missing values in categorical columns
for col in categorical_cols:
    if file[col].isnull().sum() > 0:
        file[col].fillna(file[col].mode()[0], inplace=True)  # Use Mode

print("\nMissing Values After Categorical Imputation:\n", file.isnull().sum())

# Save imputed data to file
file.to_excel("Imputed_Thyroid_Data.xlsx", index=False)
