import numpy as np
from NeuralNetwork import trace


@trace()
def mean_squared_error(y_true, y_pred, prime=False):
    y_pred = y_pred.squeeze()
    if prime:
        return 2 * (y_pred - y_true) / y_true.size
    return np.mean(np.power(y_true - y_pred, 2), axis=-1)


@trace()
def categorical_cross_entropy(y_true, y_pred, prime=False):
    y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
    if prime:
        return y_pred_clipped - y_true
    return -np.sum(y_true * np.log(y_pred_clipped), axis=-1)
