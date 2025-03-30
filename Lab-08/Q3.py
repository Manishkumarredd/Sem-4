#A3:Different Activation Functions
activation_functions = ['step', 'bipolar_step', 'sigmoid', 'relu']
epochs_results = {}

for activation in activation_functions:
    _, _, epochs = train_perceptron(X_and, y_and, initial_weights.copy(), activation=activation)
    epochs_results[activation] = epochs
    print(f"Epochs needed with {activation} activation: {epochs}")
