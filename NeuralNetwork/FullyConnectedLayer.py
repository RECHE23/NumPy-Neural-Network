from Layer import Layer
import numpy as np


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)
        super().__init__()

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.einsum("ij,jk", self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.einsum("ij,kj", output_error, self.weights)
        weights_error = np.einsum("ji,jk", self.input, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
