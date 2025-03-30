#A5:XOR Gate Implementation
# XOR Gate Data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])  # XOR labels

# Initial weights
initial_weights = np.array([10, 0.2, -0.75])
learning_rate = 0.05

# Activation functions to test (A3 requirements)
activation_functions = ['step', 'bipolar_step', 'sigmoid', 'relu']

def train_perceptron(X, y, weights, activation='step', max_epochs=1000, convergence_error=0.002):
    epoch_errors = []
    for epoch in range(max_epochs):
        total_error = 0
        for xi, target in zip(X, y):
            # Forward pass
            summation = np.dot(xi, weights[1:]) + weights[0]
            if activation == 'step':
                output = 1 if summation > 0 else 0
            elif activation == 'bipolar_step':
                output = 1 if summation > 0 else (-1 if summation < 0 else 0)
            elif activation == 'sigmoid':
                output = 1 / (1 + np.exp(-summation))
            elif activation == 'relu':
                output = max(0, summation)

            # Error and weight update
            error = target - output
            total_error += error ** 2
            weights[0] += learning_rate * error  # Bias update
            weights[1:] += learning_rate * error * xi

        epoch_errors.append(total_error)
        if total_error <= convergence_error:
            break
    return weights, epoch_errors, epoch + 1

# Run experiments for XOR
print("XOR Gate Results (Single-Layer Perceptron)")
for activation in activation_functions:
    _, errors, epochs = train_perceptron(X_xor, y_xor, initial_weights.copy(), activation=activation)
    print(f"Activation: {activation:12s} | Epochs: {epochs:4d} | Final Error: {errors[-1]:.4f}")
    plt.plot(range(1, len(errors)+1), errors, label=activation)

plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('XOR Learning (Fails - Not Linearly Separable)')
plt.legend()
plt.grid(True)
plt.show()
