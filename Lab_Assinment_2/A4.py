import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file = pd.read_excel("/content/Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

# Extract column D
column_D = file.iloc[:, 3]
print(f"D = {column_D}")

# Compute mean and variance
mean_value = statistics.mean(column_D)
variance_value = statistics.variance(column_D)
print(f"The mean of column D is = {mean_value}")
print(f"The variance of column D is = {variance_value}")

# Convert date column and extract weekdays
file["Date"] = pd.to_datetime(file["Date"])
file["Weekday"] = file["Date"].dt.weekday

# Analyze Wednesdays
wednesdays = file[file['Weekday'] == 2]
wednesday_mean = wednesdays['Price'].mean()
print(f"The sample mean for all Wednesdays in the dataset is = {wednesday_mean}")

# Analyze April data
file['Month'] = file["Date"].dt.month
April_mean = statistics.mean(file[file["Month"] == 4]['Price'])
print(f"The sample mean for April in the dataset is = {April_mean}")

# Probability calculations
loss_count = (file['Chg%'] < 0).sum()
probability_loss = loss_count / len(file)
print(f"The probability of making a loss in the stock is {probability_loss}")

profit_count = (wednesdays['Chg%'] > 0).sum()
probability_wed_profit = profit_count / len(file)
print(f"The probability of making a profit in the stock on Wednesday is {probability_wed_profit}")

# Conditional probability on Wednesdays
conditional_prob_wed = profit_count / len(wednesdays)
print(f"The conditional probability of making a profit given that today is Wednesday = {conditional_prob_wed}")

# Visualization
sns.scatterplot(x="Weekday", y="Chg%", data=file, hue="Weekday", palette="hls")
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("Chg% Distribution against the Day of the Week")
plt.show()