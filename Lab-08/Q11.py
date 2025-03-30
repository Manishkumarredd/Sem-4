# A11: MLPClassifier for AND and XOR
# AND gate
mlp_and = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd',
                       learning_rate_init=0.05, max_iter=1000, tol=0.002)
mlp_and.fit(X_and, y_and)
print("\nMLPClassifier for AND gate:")
print("Training set accuracy:", mlp_and.score(X_and, y_and))

# XOR gate
mlp_xor = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd',
                       learning_rate_init=0.05, max_iter=1000, tol=0.002)
mlp_xor.fit(X_xor, y_xor)
print("\nMLPClassifier for XOR gate:")
print("Training set accuracy:", mlp_xor.score(X_xor, y_xor))
