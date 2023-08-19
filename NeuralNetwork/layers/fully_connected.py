import numpy as np
from . import Layer
from NeuralNetwork.tools import trace


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        # Xavier initialization:
        a = np.sqrt(6/(input_size + output_size))
        self.weights = np.random.uniform(-a, a, (input_size, output_size))
        self.bias = np.zeros((output_size, ))
        super().__init__()

    @trace()
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.einsum("ij,jk", self.input, self.weights, optimize=True) + self.bias
        return self.output

    @trace()
    def backward_propagation(self, output_error, learning_rate, y_true):
        input_error = np.einsum("ij,kj", output_error, self.weights, optimize=True)
        weights_error = np.einsum("ji,jk", self.input, output_error, optimize=True)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(output_error, axis=0)
        return input_error
