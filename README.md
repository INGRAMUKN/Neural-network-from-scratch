# Neural-network-from-scratch
I've created a neural network using nothing but NumPy and a lot of caffeine. It's flexible, it's modular, and most importantly, it actually works (most of the time)!
-------------------------------------------------------------------
Presentation of the project:
A neural network is a machine learning model that draws inspiration from the human brain. It uses interconnected nodes, or neurons, in a layered structure that mirrors how our brains work. I was fascinated by this concept and curious about how it could be put into practice. So, I started digging into the details, learning about the different steps neural networks go through. I studied the math behind them and decided to build one from scratch to put my knowledge to the test.
-------------------------------------------------------------------
- Forward prop? Check.
- Backprop? You bet.
- Gradient descent? Of course!
- Sigmoid activation (because why make life easy with ReLU)
- Cross-entropy loss, for when you really want to punish your network's mistakes
--------------------------------------------------------------------
Here is an AI generated data example to check out how the project works: 

# Example data and parameters (replace with actual data and parameters)
X = np.random.randn(10, 100)  # Example input data (10 features, 100 samples)
Y = np.random.randn(1, 100)   # Example target data (1 output, 100 samples)
layer_dims = [10, 5, 1]       # 3-layer network: 10 -> 5 -> 1
epochs = 1000                 # Number of training epochs
learning_rate = 0.01          # Learning rate

# Train the model
params, cost_history = train(X, Y, layer_dims, epochs, learning_rate)

everything else is in NeuralNetwork.py file, thank you.
