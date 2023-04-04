import numpy as np
import math
from icecream import ic
from data import get_training_data

ic.configureOutput(includeContext=True)

class Layer:

    def __init__(self, n_inputs, n_neurons) -> None:
        # Initialize weights and biases
        self.n_inputs = n_inputs
        np.random.seed(42)
        self.weights = np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros((1, n_neurons))
        ic(self.weights, self.biases)

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = np.array(inputs)

        # Flatten inputs
        self.inputs = np.reshape(self.inputs, (-1, self.n_inputs))
        ic((self.inputs))

        self.output = np.array([])
        # calculate matrix multiplication of input, weight and bias one input at a time
        for input in self.inputs:
            # Calculate output values from input ones, weights and biases
            result = np.matmul(input, self.weights.T) + self.biases
            
            # if self.output is empty, directly write away result, otherwise concactenate
            if self.output.shape[0] == 0:
                self.output = result
            else:
                self.output = np.concatenate((self.output, result), axis=0)
        ic(self.output)

        return None

# Softmax activation
class SoftmaxActivation:

    def __init__(self):
        pass

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # calculate denominator of softmax function
        denominator = np.sum(np.exp(inputs), axis=1, keepdims=True)

        # calculate softmax probabilities
        self.output = np.exp(self.inputs)/denominator
        
        return None

# Cross Entropy Loss
class CrossEntropyLoss:

    def __init__(self):
        pass
    
    def forward(self, y_pred, y_true):
        # Remember input values
        self.y_pred = y_pred
        self.y_true = y_true

        # number of samples
        N = len(y_pred)
        
        # calculate cross-entropy loss
        self.loss = -np.sum(y_true * np.log2(y_pred))/N

# create training data
X, y = get_training_data()

# initialize layer with 9 inputs and 2 outputs
layer = Layer(9, 2)

# initialize softmax activation
softmax_activation = SoftmaxActivation()

# initialize cross entropy loss
cross_entropy_loss = CrossEntropyLoss()

# make forward pass dense layer
layer.forward(X)

# make forward pass softmax activation
softmax_activation.forward(layer.output)
ic(softmax_activation.output)

# calculate loss
cross_entropy_loss.forward(softmax_activation.output, y)
ic(cross_entropy_loss.loss)
