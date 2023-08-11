import numpy as np


def tanh(x, prime=False):
    if prime:
        return 1 - np.tanh(x)**2
    return np.tanh(x)


def mean_squared_error(y_true, y_pred, prime=False):
    if prime:
        return 2 * (y_pred - y_true) / y_true.size
    return np.mean(np.power(y_true - y_pred, 2))
