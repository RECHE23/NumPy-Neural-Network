import numpy as np
from . import Layer


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size, *args, **kwargs):
        # Xavier initialization:
        a = np.sqrt(6/(input_size + output_size))
        self.weights = np.random.uniform(-a, a, (input_size, output_size))
        self.bias = np.zeros((output_size, ))
        super().__init__(*args, **kwargs)

    def _forward_propagation(self, input_data):
        self.output = np.einsum("ij,jk", self.input, self.weights, optimize=True) + self.bias

    def _backward_propagation(self, upstream_gradients, y_true):
        self.retrograde = np.einsum("ij,kj", upstream_gradients, self.weights, optimize=True)
        weights_error = np.einsum("ji,jk", self.input, upstream_gradients, optimize=True)

        self.optimizer.update([self.weights, self.bias], [weights_error, np.sum(upstream_gradients, axis=0)])
