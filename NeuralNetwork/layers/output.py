import numpy as np
from . import Layer


class OutputLayer(Layer):
    def __init__(self, activation_function, loss_function):
        self.activation_function = activation_function
        self.loss_function = loss_function
        super().__init__()

    def _forward_propagation(self, input_data):
        self.output = self.activation_function(self.input)

    def _backward_propagation(self, upstream_gradients, learning_rate, y_true):
        self.retrograde = self.loss_function(y_true, self.output, prime=True)

    def loss(self, y_true, y_pred, prime=False):
        return np.sum(self.loss_function(y_true, y_pred, prime))
