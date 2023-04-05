import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from data import get_training_data, get_test_data

ic.configureOutput(includeContext=True)

# Goal: programming a neural net using one dense layer and a softmax activation \
# function to succesfully classify a dataset with crosses and circles

# Dense layer
class Layer:

    def __init__(self, n_inputs, n_neurons) -> None:
        # Initialize weights and biases
        self.n_inputs = n_inputs
        np.random.seed(42)
        self.weights = np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros((1, n_neurons))
        # ic(self.weights, self.biases)

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = np.array(inputs)

        # Flatten inputs
        self.inputs = np.reshape(self.inputs, (-1, self.n_inputs))
        # ic((self.inputs))

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
        # ic(self.output)

        return None
    
    # Backward pass
    def backward(self, dvalues):
        # d/dx xi*wi + b = wi
        # d/dw xi*wi + b = xi
        # d/db xi*wi + b = 1
        # since this partial derivate is part of the chainrule we need to multiply this \
        # derivative with the gradient of the previous layer (dinputs)

        # ic(dvalues, self.weights, self.biases, self.inputs)
        self.dinputs = np.dot(dvalues, self.weights)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # self.dweights needs to be calculated one input at a time
        self.dweights = np.array([])
        for i, input in enumerate(self.inputs):
            # ic(input.reshape(len(self.inputs), -1).shape, dvalues[i].reshape(1, -1).shape)
            result = np.matmul(input.reshape(len(self.inputs), -1), dvalues[i].reshape(1, -1)).reshape(1, len(self.inputs), -1)
            # ic(result)

            if self.dweights.shape[0] == 0:
                self.dweights = result
            else:
                self.dweights = np.concatenate((self.dweights, result), axis=0)

        # average dweights for all input samples
        self.dweights = np.sum(self.dweights, axis=0)/len(self.inputs)
        # ic(self.dinputs, self.dweights, self.dbiases)


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
    
    def forward(self, y_pred, y_true, logging):
        # Remember input values
        self.y_pred = y_pred
        self.y_true = y_true

        # number of samples
        N = len(y_pred)
        
        # calculate cross-entropy loss
        self.loss = -np.sum(y_true * np.log2(y_pred))/N

        # print result
        if logging == True:
            for i, y in enumerate(y_true):
                print(f"true_val = {y}, predicted_val = {np.round(y_pred[i], 3)}, loss = {(-np.sum(y*np.log2(y_pred[i]))):.2f}")
        
        return self

# creating combined softmax activation and cross entropy loss \
# class because of easier backward gradient
class SoftmaxActivationCrossEntropyLoss:

    def __init__(self) :
        self.activation = SoftmaxActivation()
        self.loss = CrossEntropyLoss()
    
    def forward(self, inputs, y_true, logging=False):
        # forward activation pass
        self.activation.forward(inputs)
        self.output = self.activation.output

        # forward cross entropy loss pass
        return self.loss.forward(self.output, y_true, logging)

    def backward(self, y_pred, y_true):
        # calculate gradient (from literature derivate equals (y_pred - y_true)
        # ic(y_pred, y_true)
        self.dinputs = y_pred - y_true
        # ic(self.dinputs)

class Optimizer:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    # update weights with learning_rate and gradients
    def update_params(self, layer):
        # ic(layer.weights, layer.dweights, layer.dweights.T)
        # ic(layer.biases, layer.dbiases)
        layer.weights += -self.learning_rate * layer.dweights.T
        layer.biases += -self.learning_rate * layer.dbiases

class SaveLoss:
    # initialize empty lists
    def __init__(self):
        self.train_loss = []
        self.test_loss = []
    
    # save losses per epoch
    def save_loss(self, loss_train, loss_test):
        self.train_loss.append(loss_train)
        self.test_loss.append(loss_test)
    
    # generate plot of loss function
    def show_plot(self, epochs):
        fig, ax = plt.subplots(figsize=(15, 6))

        ax.plot(range(epochs), self.train_loss, 'red', label='train loss')
        ax.plot(range(epochs), self.test_loss, 'blue', label='test loss')
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        ax.set_title("Loss function")
        ax.legend()
        plt.show()


# create training data
X, y = get_training_data()
X_test, y_test = get_test_data()

# initialize layer with 9 inputs and 2 outputs
layer = Layer(9, 9)

layer2 = Layer(9, 2)

# initialize combined softmax activation and cross entropy loss function
softmaxactivation_crossentropyloss = SoftmaxActivationCrossEntropyLoss()

# initialize loss saving
save_losses = SaveLoss()

# define nummber of epochs
epochs = 8000
for epoch in range(epochs):
    print(f"####### EPOCH {epoch} #######")
    # forward pass train
    layer.forward(X)
    layer2.forward(layer.output)
    loss = softmaxactivation_crossentropyloss.forward(layer2.output, y, True)
    ic(loss.loss)

    # backward pass train
    softmaxactivation_crossentropyloss.backward(softmaxactivation_crossentropyloss.output, y)
    layer2.backward(softmaxactivation_crossentropyloss.dinputs)
    layer.backward(layer2.dinputs)

    # forward pass test
    layer.forward(X_test)
    layer2.forward(layer.output)
    loss_test = softmaxactivation_crossentropyloss.forward(layer2.output, y_test)
    ic(loss_test.loss)
    save_losses.save_loss(loss.loss, loss_test.loss)

    # update weights and biases with optimizer
    optimizer = Optimizer(0.001)
    optimizer.update_params(layer)
    optimizer.update_params(layer2)

# make prediction
print("######## PREDICTION ########")
# forward pass
layer.forward(X_test)
layer2.forward(layer.output)
loss = softmaxactivation_crossentropyloss.forward(layer2.output, y_test, True)
ic(loss.loss)

# plotting loss functions of train and test data
save_losses.show_plot(epochs)

# ask you can see in the output of the plot, the train and test loss are very similar
# I guess it means that the neural net does not have enough data available to start overfitting

# In the prediction of the test dataset is also becomes apparant that the nn has problems with \
# giving a high confidence on test sample 2 (0.669 prob of it being a cross (cross being true label))
# looking at the sample it make sense
# img06 = {
#         "img": [[1, 0, 1],
#                 [0, 0, 0],
#                 [1, 0, 1]],
#         "label": [1, 0]
#         }
# Although it is labeled as a cross, with some imagination you could also see it as a circle
#
# #### UPDATE ####
# Adding a second dense layer, even without an activation function in between, already improves the performance
# For sample 2, the prob improves to 0.963 from 0.669!!
