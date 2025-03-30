#A8:Neural Network with Backpropagation for AND Gate
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_neural_network(X, y, hidden_units=2, learning_rate=0.05, max_epochs=1000, convergence_error=0.002):
    # Initialize weights randomly
    input_size = X.shape[1]
    input_hidden_weights = np.random.rand(input_size, hidden_units) * 0.1 - 0.05
    hidden_output_weights = np.random.rand(hidden_units) * 0.1 - 0.05
    hidden_bias = np.random.rand(hidden_units) * 0.1 - 0.05
    output_bias = np.random.rand() * 0.1 - 0.05

    epoch_errors = []

    for epoch in range(max_epochs):
        total_error = 0

        for xi, target in zip(X, y):
            # Forward pass
            hidden_layer_input = np.dot(xi, input_hidden_weights) + hidden_bias
            hidden_layer_output = sigmoid(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, hidden_output_weights) + output_bias
            predicted_output = sigmoid(output_layer_input)

            # Error calculation
            error = target - predicted_output
            total_error += error ** 2

            # Backward pass
            # Output layer
            d_output = error * sigmoid_derivative(predicted_output)

            # Hidden layer
            d_hidden = d_output * hidden_output_weights * sigmoid_derivative(hidden_layer_output)

            # Update weights
            hidden_output_weights += learning_rate * d_output * hidden_layer_output
            output_bias += learning_rate * d_output

            input_hidden_weights += learning_rate * np.outer(xi, d_hidden)
            hidden_bias += learning_rate * d_hidden

        epoch_errors.append(total_error)

        if total_error <= convergence_error:
            break

    return input_hidden_weights, hidden_output_weights, epoch_errors, epoch + 1

# Train neural network for AND gate
input_hidden_weights, hidden_output_weights, nn_errors, nn_epochs = train_neural_network(X_and, y_and)

print(f"\nNeural network converged in {nn_epochs} epochs")
