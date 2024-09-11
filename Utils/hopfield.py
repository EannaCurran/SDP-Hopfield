import numpy as np
import random
import math
from numpy.random import choice

# Redifine the convergance criteria.

class HopfieldNetworkCut:

    def __init__(self, data, iterations, activationThreshold):
        self.numberOfNeurons = data.shape[0]
        self.weights = data.copy()
        self.iterations = iterations
        A = np.random.randint(0, 2, self.numberOfNeurons)
        A[A == 0] = -1
        self.activation = A
        self.activationRound = A
        self.nonConvergenceCount = 0
        self.activationThreshold = activationThreshold
        self.activationHistory = []

    def train(self):
        for iterate in range(0, self.iterations):
            newIndex = dict()
            ''' Asynchronously  update
            update_order = list(range(0,self.numberOfNeurons))
            random.shuffle(update_order)
            oldActivation = self.activation.copy()
            for index in update_order:
                score = np.dot(self.weights[index], self.activation)
                self.activation[index] = 1 if score > 0 else -1
            '''
            newActivation = sigmoid_all(self.weights.dot(self.activation))

            newActivationRound = newActivation.copy()
            newIndex['output'] = newActivation.copy()
            newActivationRound[newActivationRound > 0.5] = 1
            newActivationRound[newActivationRound <= 0.5] = -1

            self.activationHistory.append(newIndex)

            if np.array_equal(newActivationRound, self.activationRound) and iterate != 0:
                self.nonConvergenceCount = iterate + 1
                self.activation = self.activationRound
                return

            self.activation = newActivation
            self.activationRound = newActivationRound

        self.activation = self.activationRound
        self.nonConvergenceCount = -1

    def get_partition(self):
        return self.activation, self.nonConvergenceCount

    def get_weights(self):
        return self.weights

    def back_propagate(self, error):

        for i in reversed(range(len(self.activationHistory)-1)):

            layer = self.activationHistory[i]
            activations = self.activationHistory[i+1]['output']
            delta = self.transfer_derivative(activations) * error
            delta_re = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activationHistory[i]['output']
            current_activations = current_activations.reshape(current_activations.shape[0], -1)
            layer['weight error'] = np.clip(np.dot(current_activations, delta_re), -1, 1)

            error = np.dot(delta, self.weights[i].T)

    def get_history(self):
        return self.activationHistory

    def transfer_derivative(self, values):
        return values * (1-values)


class HopfieldNetworkClique:

    def __init__(self, data, dummyNode, iterations, activationThreshold):
        self.numberOfNeurons = data.shape[0]
        self.weights = data.copy()
        self.iterations = iterations
        A = np.random.randint(0, 2, self.numberOfNeurons)
        self.activation = A
        self.activationProb = A.copy().astype(float)
        self.nonConvergenceCount = 0
        self.activationThreshold = activationThreshold
        self.dummyNode = dummyNode
        self.activationHistory = []

    def train(self):
        for iterate in range(0, self.iterations):

            newIndex = dict()
            update_order = list(range(0, self.numberOfNeurons))
            random.shuffle(update_order)
            newActivation = self.activation.copy()
            newActivationProb = self.activationProb.copy()

            for index in update_order:
                score = np.dot(self.weights[index], newActivation) + self.dummyNode[index]
                score = sigmoid(score)
                newActivationProb[index] = score
                newActivation[index] = 1 if score > 0.5 else 0

            newIndex['output'] = newActivationProb
            newIndex['weights'] = self.weights
            newIndex['bias'] = self.dummyNode
            self.activationHistory.append(newIndex)

            if np.array_equal(newActivation, self.activation) and iterate != 0:
                self.nonConvergenceCount = iterate + 1
                return
            self.activation = newActivation
            self.activationProb = newActivationProb
        self.nonConvergenceCount = -1

    def get_partition(self):
        return self.activation, self.nonConvergenceCount

    def get_weights(self):
        return self.weights

    def get_history(self):
        return self.activationHistory

    def back_propagate(self, error):

        for i in reversed(range(len(self.activationHistory) - 1)):

            layer = self.activationHistory[i]
            activations = self.activationHistory[i + 1]['output'].copy()
            delta = self.transfer_derivative(activations).copy() * error
            delta_re = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activationHistory[i]['output'].copy()
            current_activations = current_activations.reshape(current_activations.shape[0], -1)
            layer['weight error'] = np.clip(np.dot(current_activations, delta_re),-1,1)
            layer['bias error'] = np.clip(delta,-1,1)
            error = np.dot(delta, self.weights[i].copy().T)


    def transfer_derivative(self, values):
        return values * (1-values)

    def relu(self, values):
        return np.maximum(0, values)

def sigmoid_all(x):
    for n in range(0, len(x)):
        x[n] = 1 / (1 + math.exp(-x[n]))
    return x


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class HopfieldNetworkColour:

    def __init__(self, data, iterations, activationThreshold, k, G):

        self.numberOfNeurons = data.shape[0]
        self.weights = data.copy()
        self.iterations = iterations
        self.activation = np.full((self.numberOfNeurons, k), (1/k))
        self.nonConvergenceCount = 0
        self.activationThreshold = activationThreshold
        self.k = k
        self.graph = G
        self.activationHistory = []

    def train(self):

        for iterate in range(0, self.iterations):

            update_order = list(range(0, self.numberOfNeurons))
            random.shuffle(update_order)
            newActivation = self.activation.copy()
            newIndex = dict()
            indexOutputs = []

            for index in update_order:
                for colourIndex in range(0, self.k):
                    newColourActivation = np.dot(self.weights[index], newActivation[:, colourIndex])
                    newActivation[index, colourIndex] = newColourActivation

                newActivation[index] = softmax_vector(newActivation[index])
                colourIndex = choice(range(len(newActivation[index])), p=newActivation[index])
                newIndexActivation = [1 if n == colourIndex else 0 for n in range(0, self.k)]
                newActivation[index] = newIndexActivation
                indexOutputs.append(newActivation[index])

            newIndex['weights'] = self.weights
            newIndex['output'] = indexOutputs

            self.activationHistory.append(indexOutputs)
            if np.array_equal(newActivation, self.activation):
                self.nonConvergenceCount = iterate + 1
                return

            self.activation = newActivation
        self.nonConvergenceCount = iterate + 1

    def get_colouring(self):
        return self.activation

    def get_weights(self):
        return self.weights

def softmax_vector(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)
