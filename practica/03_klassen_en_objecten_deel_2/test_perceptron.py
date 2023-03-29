#!/usr/bin/env python

from perceptron import Perceptron
import itertools
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

possibleOutcomes = [0, 1]
xTrain = np.array(
    [element for element in itertools.product(possibleOutcomes, possibleOutcomes)]
)
logging.debug(f"====== andPerceptron ======")
logging.debug(f"xTrain : {xTrain}")

yTrain = np.array([0, 0, 0, 1])

andPerceptron = Perceptron()
andPerceptron.train(xTrain, yTrain, epochs=100, learningRate=0.1)
testInput = np.array([1, 1])
logging.debug(f"testInput : {testInput}")

prediction = andPerceptron.predict(testInput)
logging.info(f"Predicted y value : {prediction}")

# OPDDRACHT
# Maak nu zelf het object orPerceptron
orPerceptron = Perceptron()

yTrain = np.array([0, 1, 1, 1])

orPerceptron.train(xTrain, yTrain, epochs=100, learningRate=0.1)
testInput = np.array([0, 1])
testInput2 = np.array([[0, 0]])
logging.debug(f"====== orPerceptron ======")
logging.debug(f"testInput : {testInput}")

prediction = orPerceptron.predict(testInput)
logging.info(f"Predicted y value : {prediction}")

logging.debug(f"testInput2 : {testInput2}")

prediction2 = orPerceptron.predict(testInput2)
logging.info(f"Predicted y value : {prediction2}")
