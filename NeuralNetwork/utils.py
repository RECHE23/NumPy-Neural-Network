import numpy as np
from activation_functions import softmax


def convert_data(data, to=None):
    if to == "one_hot":
        idx = np.argmax(data, axis=-1)
        data = np.zeros(data.shape)
        data[np.arange(data.shape[0]), idx] = 1
    elif to == "binary":
        data = np.where(data >= 0.5, 1, 0)
    elif to == "labels":
        data = np.argmax(data, axis=-1)
    elif to == "probability":
        data = softmax(data)
    return data.squeeze()


def batch_iterator(inputs, targets, batch_size, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, inputs.shape[0])
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt], targets[excerpt]
