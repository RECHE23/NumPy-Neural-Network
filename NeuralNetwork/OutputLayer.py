import numpy as np
from Layer import Layer


class OutputLayer(Layer):
    def __init__(self, activation_function, loss_function):
        self.activation_function = activation_function
        self.loss_function = loss_function
        super().__init__()

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate, y_true):
        return self.loss_function(y_true, self.output, prime=True)

    def loss(self, y_true, y_pred, prime=False):
        return np.sum(self.loss_function(y_true, y_pred, prime))
