#A6:Customer Data Classification
# Customer data
X_customer = np.array([
    [20, 6, 2],
    [16, 3, 6],
    [27, 6, 2],
    [19, 1, 2],
    [24, 4, 2],
    [22, 1, 5],
    [15, 4, 2],
    [18, 4, 2],
    [21, 1, 4],
    [16, 2, 4]
])
y_customer = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])  # 1=Yes, 0=No

def summation_unit(inputs, weights):
    return np.dot(inputs, weights[1:]) + weights[0]

def activation_unit(y, activation_type='sigmoid'):
    if activation_type == 'sigmoid':
        return 1 / (1 + np.exp(-y))
    elif activation_type == 'step':
        return 1 if y > 0 else 0

def train_perceptron(X, y, weights, activation='sigmoid', learning_rate=0.1, max_epochs=1000, convergence_error=0.002):
    epoch_errors = []
    for epoch in range(max_epochs):
        total_error = 0
        for xi, target in zip(X, y):
            # Forward pass
            summation = summation_unit(xi, weights)
            output = activation_unit(summation, activation)

            # Error calculation
            error = target - output
            total_error += error ** 2

            # Weight updates
            weights[0] += learning_rate * error  # bias update
            for i in range(len(xi)):
                weights[i+1] += learning_rate * error * xi[i]

        epoch_errors.append(total_error)
        if total_error <= convergence_error:
            break

    return weights, epoch_errors, epoch + 1

# Initialize weights randomly (3 features + 1 bias)
customer_weights = np.random.rand(4)

# Train with sigmoid activation and learning rate=0.1
trained_weights, errors, epochs = train_perceptron(
    X_customer,
    y_customer,
    customer_weights,
    activation='sigmoid',
    learning_rate=0.1
)

print(f"\nCustomer classification converged in {epochs} epochs")

# Make predictions
predictions = []
for xi in X_customer:
    summation = summation_unit(xi, trained_weights)
    output = activation_unit(summation, 'sigmoid')
    predictions.append(1 if output > 0.5 else 0)

print("Predictions:", predictions)
print("Actual:     ", y_customer)
print("Accuracy:   ", np.mean(predictions == y_customer))
