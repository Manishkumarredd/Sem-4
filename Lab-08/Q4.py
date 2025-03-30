#A4: Varying Learning Rates
learning_rates = np.arange(0.1, 1.1, 0.1)
epochs_vs_lr = []
for lr in learning_rates:
    _, _, epochs = train_perceptron(X_and, y_and, initial_weights.copy(), learning_rate=lr)
    epochs_vs_lr.append(epochs)
    print(f"Learning rate {lr:.1f} - Epochs needed: {epochs}")
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, epochs_vs_lr, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Epochs to Converge')
plt.title('Learning Rate vs Epochs to Converge')
plt.grid(True)
plt.show()
