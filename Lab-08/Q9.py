#A9:XOR with Backpropagation
# XOR Gate Data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])  # XOR labels

# Initial weights and parameters
initial_weights = np.array([10, 0.2, -0.75])
learning_rate = 0.05
activation = 'step'   # Step activation

def train_perceptron(X, y, weights, activation='step', max_epochs=1000, convergence_error=0.002):
    epoch_errors = []
    for epoch in range(max_epochs):
        total_error = 0
        for xi, target in zip(X, y):
            # Forward pass
            summation = np.dot(xi, weights[1:]) + weights[0]
            output = 1 if summation > 0 else 0  # Step activation

            # Error calculation
            error = target - output
            total_error += error ** 2

            # Weight updates
            weights[0] += learning_rate * error  # Bias update
            weights[1:] += learning_rate * error * xi

        epoch_errors.append(total_error)
        if total_error <= convergence_error:
            break

    return weights, epoch_errors, epoch + 1

# Train on XOR
trained_weights, errors, epochs_needed = train_perceptron(
    X_xor, y_xor, initial_weights.copy(), activation=activation
)

# Results
print(f"Results (XOR Gate with Step Activation)")
print(f"Initial weights: {initial_weights}")
print(f"Final weights:   {trained_weights}")
print(f"Epochs needed:   {epochs_needed} (of max 1000)")
print(f"Final error:     {errors[-1]:.4f} (Target ≤ {0.002})")
print("\nNote: XOR is not linearly separable - single-layer perceptron cannot learn this pattern.")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(errors)+1), errors, marker='o', color='red')
plt.axhline(y=0.002, color='gray', linestyle='--', label='Convergence Threshold')
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('XOR Learning (Fails - Single-Layer Perceptron Limitation)')
plt.legend()
plt.grid(True)
plt.show()
