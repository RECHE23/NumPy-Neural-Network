import numpy as np
from NeuralNetwork.tools import trace


@trace()
def convert_targets(targets, to=None):
    if (to is None and targets.ndim == 1) or to == "categorical":
        # Equivalent to keras.tools.to_categorical:
        targets = np.eye(len(set(targets)))[targets]
    elif to == "one_hot":
        # Converts targets' vectors to one hot vectors:
        idx = np.argmax(targets, axis=-1)
        targets = np.zeros(targets.shape)
        targets[np.arange(targets.shape[0]), idx] = 1
    elif to == "binary":
        # Converts a probability vector to a binary vector using a threshold of 0.5:
        targets = np.where(targets >= 0.5, 1, 0)
    elif to == "labels" and targets.ndim != 1:
        # Converts targets' vectors to labels:
        targets = np.argmax(targets, axis=-1)
    elif to == "probability":
        # Converts targets' vectors to probability distributions:
        from NeuralNetwork.functions.activation import softmax
        targets = softmax(targets)
    return targets


def pair(args):
    assert isinstance(args, tuple) or isinstance(args, int)
    if isinstance(args, tuple):
        assert len(args) == 2
    else:
        args = (args, args)
    return args
