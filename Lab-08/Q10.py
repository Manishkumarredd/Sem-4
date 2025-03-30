#A10:Two Output Nodes Implementation
def train_perceptron_two_output(X, y, weights, learning_rate=0.05, max_epochs=1000, convergence_error=0.002):
    """
    Perceptron with two output nodes
    Maps 0 to [1, 0] and 1 to [0, 1]
    """
    # Initialize weights for two outputs
    if weights.shape != (3, 2):  # 2 bias + 2 input weights for each output
        weights = np.random.rand(3, 2) * 0.1 - 0.05

    epoch_errors = []

    for epoch in range(max_epochs):
        total_error = 0

        for xi, target in zip(X, y):
            # Convert target to two outputs
            target_vec = np.array([1, 0]) if target == 0 else np.array([0, 1])

            # Forward pass for both outputs
            summation1 = summation_unit(xi, weights[:, 0])
            output1 = activation_unit(summation1, 'sigmoid')

            summation2 = summation_unit(xi, weights[:, 1])
            output2 = activation_unit(summation2, 'sigmoid')

            predicted_vec = np.array([output1, output2])

            # Error calculation
            error = target_vec - predicted_vec
            total_error += np.sum(error ** 2)

            # Weight updates for both outputs
            for i in range(2):
                weights[0, i] += learning_rate * error[i]  # bias update
                for j in range(len(xi)):
                    weights[j+1, i] += learning_rate * error[i] * xi[j]

        epoch_errors.append(total_error)

        if total_error <= convergence_error:
            break

    return weights, epoch_errors, epoch + 1

# Train two-output perceptron for AND gate
two_output_weights, two_output_errors, two_output_epochs = train_perceptron_two_output(X_and, y_and, np.random.rand(3, 2))

print(f"\nTwo-output perceptron converged in {two_output_epochs} epochs")
