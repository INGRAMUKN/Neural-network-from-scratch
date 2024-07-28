#importing libraries 
import numpy as np 
import matplotlib.pyplot as plt

# Function to initialize parameters (weights, biases)
# layer_dims holds the dimensions of each layer
def init_params(layer_dims):
    np.random.seed(3)
    params = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        params['b'+str(l)] = np.zeros((layer_dims[l], 1))
        
    return params

# Sigmoid activation function to remove linearity in the model
# We define Z (linear hypothesis) = W*X + b where X is the input, W is the weight 
# and b is the bias
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z  # We cache Z for backpropagation phase
    return A, cache 

# Forward propagation function - this function will take values from previous
# layers as an input to the next layers (FORWARD propagation)
def forward_prop(X, params):
    A = X  # Input to first layer i.e., training data
    caches = []
    L = len(params) // 2
    for l in range(1, L+1):
        A_prev = A
        # Linear Hypothesis
        Z = np.dot(params['W'+str(l)], A_prev) + params['b'+str(l)] 
        # Sigmoid activation on linear hypothesis
        A, activation_cache = sigmoid(Z)
        # Storing the linear and activation cache
        cache = ((A_prev, params['W'+str(l)], params['b'+str(l)]), activation_cache)
        caches.append(cache)
    
    return A, caches

# Cost function - this function measures how well the neural network predictions 
# fit the actual target, using cross-entropy loss
def cost_function(A, Y):
    m = Y.shape[1]
    cost = (-1/m) * np.sum(np.multiply(np.log(A), Y) + np.multiply(np.log(1-A), 1-Y))
    cost = np.squeeze(cost)  # To ensure cost is a scalar
    return cost

# Backpropagation for a single layer
def one_layer_backward(dA, cache):
    linear_cache, activation_cache = cache
    Z = activation_cache
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    
    dZ = dA * sigmoid_derivative(Z)  # Derivative of the sigmoid function
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

# Derivative of the sigmoid function
def sigmoid_derivative(Z):
    A, _ = sigmoid(Z)
    return A * (1 - A)

# Backpropagation algorithm
def backprop(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    
    current_cache = caches[L-1]
    grads['dA'+str(L)], grads['dW'+str(L)], grads['db'+str(L)] = one_layer_backward(dAL, current_cache)
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = one_layer_backward(grads["dA" + str(l+2)], current_cache)
        grads["dA" + str(l+1)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
        
    return grads

# Updating the parameters of the neural network 
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(L):
        parameters['W'+str(l+1)] -= learning_rate * grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] -= learning_rate * grads['db'+str(l+1)]
        
    return parameters

# Training loop
def train(X, Y, layer_dims, epochs, lr):
    params = init_params(layer_dims)
    cost_history = []
    
    for i in range(epochs):
        Y_hat, caches = forward_prop(X, params)
        cost = cost_function(Y_hat, Y)
        cost_history.append(cost)
        grads = backprop(Y_hat, Y, caches)
        params = update_parameters(params, grads, lr)
        
        if i % 100 == 0:
            print(f"Cost after epoch {i}: {cost}")
        
    return params, cost_history

