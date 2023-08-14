import numpy as np
from activation_functions import softmax


def convert_data(data, to=None):
    if to == "one_hot":
        idx = np.argmax(data, axis=-1).squeeze()
        data = np.zeros(data.shape).squeeze()
        data[np.arange(data.shape[0]), idx] = 1
    elif to == "binary":
        data = np.where(data >= 0.5, 1, 0).squeeze()
    elif to == "labels":
        data = np.argmax(data, axis=-1).squeeze()
    elif to == "probability":
        data = softmax(data)
    return data
