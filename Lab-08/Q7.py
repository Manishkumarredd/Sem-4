#A7:Comparison with Pseudo-Inverse
# Adding bias term to X
X_with_bias = np.c_[np.ones(X_customer.shape[0]), X_customer]
# Pseudo-inverse solution
pseudo_weights = np.linalg.pinv(X_with_bias) @ y_customer
# Predict with pseudo-inverse
pseudo_predictions = (X_with_bias @ pseudo_weights > 0.5).astype(int)

print("\nPseudo-inverse results:")
print("Pseudo-inverse weights:", pseudo_weights)
print("Predictions:", pseudo_predictions)
print("Actual:    ", y_customer)
print("Accuracy:  ", np.mean(pseudo_predictions == y_customer))

# After running both perceptron and pseudo-inverse:
print("\nComparison Summary")
print("Method\t\t\tAccuracy\tWeights")
print(f"Perceptron\t\t{np.mean(predictions == y_customer):.4f}\t{trained_weights}")
print(f"Pseudo-Inverse\t\t{np.mean(pseudo_predictions == y_customer):.4f}\t{pseudo_weights}")
