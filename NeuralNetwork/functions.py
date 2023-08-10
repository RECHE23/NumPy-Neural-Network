import numpy as np


def tanh(x, prime=False):
    return 1-np.tanh(x)**2 if prime else np.tanh(x)


def mse(y_true, y_pred, prime=False):
    return 2*(y_pred-y_true)/y_true.size if prime else np.mean(np.power(y_true-y_pred, 2))
