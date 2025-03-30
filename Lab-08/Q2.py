#A2:Perceptron Implementation for AND Gate
def train_perceptron(X, y, weights, learning_rate=0.05, activation='step', max_epochs=1000, convergence_error=0.002):
    epoch_errors = []
    for epoch in range(max_epochs):
        total_error = 0
        for xi, target in zip(X, y):
            # Forward pass
            summation = summation_unit(xi, weights)
            output = activation_unit(summation, activation)
            # Error calculation
            error = comparator_unit(target, output)
            total_error += error ** 2
            # Weight updates
            weights[0] += learning_rate * error  # bias update
            for i in range(len(xi)):
                weights[i+1] += learning_rate * error * xi[i]
        epoch_errors.append(total_error)

        # Check for convergence
        if total_error <= convergence_error:
            break

    return weights, epoch_errors, epoch + 1

# AND Gate data
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Initial weights as given
initial_weights = np.array([10, 0.2, -0.75])

# Train with step activation
trained_weights, errors, epochs_needed = train_perceptron(X_and, y_and, initial_weights.copy())

print(f"Epochs needed for convergence with step activation: {epochs_needed}")

# Plot epochs vs error
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(errors)+1), errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Training Error vs Epochs (Step Activation)')
plt.grid(True)
plt.show()
