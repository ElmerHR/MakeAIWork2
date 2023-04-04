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

        # print result
        for i, y in enumerate(y_true):
            print(f"true_val = {y}, predicted_val = {np.round(y_pred[i], 3)}, loss = {(-np.sum(y*np.log2(y_pred[i]))):.2f}")
        
        return self

# creating combined softmax activation and cross entropy loss class because of easier backward gradient
class SoftmaxActivationCrossEntropyLoss:

    def __init__(self) :
        self.activation = SoftmaxActivation()
        self.loss = CrossEntropyLoss()
    
    def forward(self, inputs, y_true):
        # forward activation pass
        self.activation.forward(inputs)
        self.output = self.activation.output

        # forward cross entropy loss pass
        return self.loss.forward(self.output, y_true)



# create training data
X, y = get_training_data()

# initialize layer with 9 inputs and 2 outputs
layer = Layer(9, 2)

softmaxactivation_crossentropyloss = SoftmaxActivationCrossEntropyLoss()

# forward pass
layer.forward(X)
loss = softmaxactivation_crossentropyloss.forward(layer.output, y)
ic(loss.loss)
