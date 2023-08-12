import numpy as np


def relu(x, prime=False):
    if prime:
        return 1. * (x > 0)
    return x * (x > 0)


def tanh(x, prime=False):
    if prime:
        return 1 - np.tanh(x)**2
    return np.tanh(x)


def sigmoid(x, prime=False):
    s = 1 / (1 + np.exp(-x))
    if prime:
        return s * (1 - s)
    return s
