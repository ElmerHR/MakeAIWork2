"""Naive implementation of a neural net
"""
import numpy as np
import math
from icecream import ic

ic.configureOutput(includeContext=True)
# prepare data

# create crosses and dots data
img01 = {
        "img": [[1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]],
        "label": [1, 0]
        }  
        
img02 = {
        "img": [[1, 0, 1],
                [0, 1, 0],
                [0, 0, 1]],
        "label": [1, 0]
        }

img03 = {
        "img": [[0, 0, 1],
                [0, 1, 0],
                [1, 0, 1]],
        "label": [1, 0]
        }

img04 = {
        "img": [[1, 0, 0],
                [0, 1, 0],
                [1, 0, 1]],
        "label": [1, 0]
        }

img05 = {
        "img": [[1, 0, 1],
                [0, 1, 0],
                [1, 0, 0]],
        "label": [1, 0]
        } 

img06 = {
        "img": [[1, 0, 1],
                [0, 0, 0],
                [1, 0, 1]],
        "label": [1, 0]
        }

img07 = {
        "img": [[0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]],
        "label": [0, 1]
        }

img08 = {
        "img": [[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]],
        "label": [0, 1]
        }

img09 = {
        "img": [[1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]],
        "label": [0, 1]
        }

img10 = {
        "img": [[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]],
        "label": [0, 1]
        }

img11 = {
        "img": [[1, 1, 1],
                [1, 0, 1],
                [0, 1, 0]],
        "label": [0, 1]
        }

img12 = {
        "img": [[1, 1, 0],
                [1, 0, 1],
                [1, 1, 1]],
        "label": [0, 1]
        }


img13 = {
        "img": [[1, 1, 1],
                [1, 0, 1],
                [0, 1, 1]],
        "label": [0, 1]
        }

img14 = {
        "img": [[1, 1, 1],
                [1, 1, 1],
                [0, 1, 1]],
        "label": [0, 1]
        }

# create training and test_data
training_data = [img01, img02, img03, img04, img07, img08, img09, img10, img11]
test_data = [img05, img06, img12, img13, img14]

def softmax(input):
    """Calculate softmax probabilities

    Args:
        input (Numpy array): result of matmul first pass
    """
    # calculate denominator of softmax function
    denominator = np.sum(math.e**input)
    # calculate softmax probabilities
    softmax_output = np.array([(math.e**input/denominator)[0]])
    return softmax_output

def cross_entropy_loss(yHat, y):
    # get number of samples
    N = yHat.shape[0]
    # calculate cross-entropy loss
    cel = -np.sum(y * np.log2(yHat))/N
    return cel


def update_weights_and_bias(W, B):
    # generate random int between 0 and 9
    r1 = np.random.randint(0, 9)
    r2 = np.random.randint(0, 9)
    # generate random float between 0 and 1
    r_float1 = np.random.random_sample()
    r_float2 = np.random.random_sample()
    r_float3 = np.random.random_sample()*5
    r_float4 = np.random.random_sample()*5
    # substitute random index in weight and bias array with random float from above
    W[0][r1] = r_float1
    W[1][r2] = r_float2
    B[0][0] = r_float3
    B[0][1] = r_float4
    return None

def validate(x_test, yTest):
    output = np.array([])
    for input in x_test:
        logits = []
        # cast to numpy array
        input = np.array(input["img"])
        # flatten input to feed to nn
        input = np.reshape(input, (1, -1))
        # perform matrix multiplication
        logits = np.matmul(input, W.T) + B
        # calculate softmax probabilities
        softmax_prob = softmax(logits)
        if output.shape[0] == 0:
            output = softmax_prob
        else:
            output = np.concatenate((output, softmax_prob), axis=0)
    # ic(output)

    cel = cross_entropy_loss(output, yTest)
    ic(cel)
    return None

# initialize weights at 1
W = np.stack((np.ones_like(np.reshape(np.array(training_data[0]["img"]), (1, -1)), dtype="float")[0], np.ones_like(np.reshape(np.array(training_data[0]["img"]), (1, -1)), dtype="float")[0]), axis=0)
B = np.array([[0.0, 0.0]])
# remember weights and bias
rememberW = W.copy()
rememberB = B.copy()
# define y train data
yTrain = np.array([y["label"] for y in training_data])

epochs = 1000
cel_prev = 999999999999999
for epoch in range(epochs):
    print(f"####### EPOCH {epoch} ########")
    output = np.array([])
    # loop over all training data
    for input in training_data:
        logits = []
        # cast to numpy array
        input = np.array(input["img"])
        # flatten input to feed to nn
        input = np.reshape(input, (1, -1))
        # perform matrix multiplication
        logits = np.matmul(input, W.T) + B
        # calculate softmax probabilities
        softmax_prob = softmax(logits)
        if output.shape[0] == 0:
            output = softmax_prob
        else:
            output = np.concatenate((output, softmax_prob), axis=0)
    # calculate cross entropy loss
    cel = cross_entropy_loss(output, yTrain)
    ic(cel, cel_prev)
    # if cel is higher than previous cel than we don't want to remember the weight and bias changes and we reset them to the previous best weights and biases
    if cel > cel_prev:
        W = rememberW.copy()
        B = rememberB.copy()
    # if cel is lower, we want to update the prev cel as best cel and remember the weights and biases
    else:
        cel_prev = cel.copy()
        ic(W, B)
        rememberW = W.copy()
        rememberB = B.copy()
    # make new weights and biases
    update_weights_and_bias(W, B)

# validate performance with test data
yTest = np.array([y["label"] for y in test_data])
print(f"####### VALIDATION ########")
validate(test_data, yTest)
