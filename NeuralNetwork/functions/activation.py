import numpy as np
from NeuralNetwork import trace


@trace()
def relu(x, prime=False):
    if prime:
        return 1. * (x > 0)
    return x * (x > 0)


@trace()
def tanh(x, prime=False):
    if prime:
        return 1 - np.tanh(x)**2
    return np.tanh(x)


@trace()
def sigmoid(x, prime=False):
    s = 1 / (1 + np.exp(-x))
    if prime:
        return s * (1 - s)
    return s


@trace()
def softmax(x, prime=False):
    e = np.exp(x - np.max(x))
    s = e / np.sum(e, axis=-1, keepdims=True)
    if prime:
        return np.diagflat(s) - np.dot(s, s.T)
    return s
