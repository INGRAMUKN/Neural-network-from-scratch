Basic Neural Network from Scratch
----------------------------------
This project implements a basic neural network using Python from scratch. The network includes functions for parameter initialization, forward propagation, cost computation, backpropagation, and parameter updates.

Features:
---------
Parameter Initialization: Initialize weights and biases for the neural network.

Forward Propagation: Compute the output of the network using the sigmoid activation function.

Cost Function: Measure the performance of the model using cross-entropy loss.

Backpropagation: Compute gradients for weights and biases to update parameters.

Training Loop: Train the network over multiple epochs and track the cost.

Installation:
--------------
Ensure you have Python installed. This project requires the following libraries:

numpy,
matplotlib

Example of how to use this code:
-------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Example usage
X = np.random.randn(3, 100)  # 3 features, 100 samples
Y = np.random.randint(0, 2, (1, 100))  # Binary labels

layer_dims = [3, 5, 1]  # 3 input neurons, 5 hidden neurons, 1 output neuron
epochs = 1000
learning_rate = 0.01

params, cost_history = train(X, Y, layer_dims, epochs, learning_rate)

# Plot the cost history
plt.plot(cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost History')
plt.show()

Contributing:
--------------
Feel free to open issues or submit pull requests if you find bugs or want to add new features.

