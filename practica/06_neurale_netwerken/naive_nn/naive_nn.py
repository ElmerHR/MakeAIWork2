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


# def backpropagation():


# initialize weights at 1
W = np.stack((np.ones_like(np.reshape(np.array(training_data[0]["img"]), (1, -1)))[0], np.ones_like(np.reshape(np.array(training_data[0]["img"]), (1, -1)))[0]), axis=0)
B = np.array([[0, 0]])
yTrain = np.array([y["label"] for y in training_data])

output = np.array([])
# loop over all training data
for input in training_data:
    logits = []
    # cast to numpy array
    input = np.array(input["img"])
    # flatten input to feed to nn
    input = np.reshape(input, (1, -1))
    logits = np.matmul(input, W.T) + B
    softmax_prob = softmax(logits)
    if output.shape[0] == 0:
        output = softmax_prob
    else:
        output = np.concatenate((output, softmax_prob), axis=0)
ic(output)

cel = cross_entropy_loss(output, yTrain)
ic(cel)
    
