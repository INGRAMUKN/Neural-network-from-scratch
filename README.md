# Neural-network-from-scratch
I've created a neural network using nothing but NumPy and a lot of caffeine. It's flexible, it's modular, and most importantly, it actually works (most of the time)!
-------------------------------------------------------------------
Presentation of the project:
A neural network is a machine learning model that draws inspiration from the human brain. It uses interconnected nodes, or neurons, in a layered structure that mirrors how our brains work. I was fascinated by this concept and curious about how it could be put into practice. So, I started digging into the details, learning about the different steps neural networks go through. I studied the math behind them and decided to build one from scratch to put my knowledge to the test.
-------------------------------------------------------------------
The goal of this code is to minimize the cost as much as possible the closer to 0 the better
-------------------------------------------------------------------
- Forward prop
- Backprop
- Gradient descent
- Sigmoid activation (because why make life easy with ReLU)
- Cross-entropy loss, to punish your network's mistakes
--------------------------------------------------------------------
Here is an AI generated data example to check out how the project works: 

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])
layer_dims = [2, 4, 1]

epochs = 10000
learning_rate = 0.1

params, cost_history = train(X, Y, layer_dims, epochs, learning_rate)

# Plot the cost history
plt.plot(cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Training Cost over Epochs')
plt.show()

# Make predictions
Y_pred, _ = forward_prop(X, params)
Y_pred = (Y_pred > 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(Y_pred == Y)
print(f"Accuracy: {accuracy * 100:.2f}%")
everything else is in NeuralNetwork.py file, thank you.
