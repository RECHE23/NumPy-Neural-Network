import numpy as np
from scipy import signal
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from . import Layer
from NeuralNetwork.tools import trace


class ConvolutionalLayer(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        self._kernels_gradients = None
        self._input_gradients = None
        self._output_gradients = None

        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        # Xavier initialization:
        a = np.sqrt(6 / (input_height * input_width + kernel_size * kernel_size))
        self.kernels = np.random.uniform(-a, a, self.kernels_shape)
        self.biases = np.zeros(self.output_shape)
        super().__init__()

    @trace()
    def forward_propagation(self, input_data):
        n_samples = input_data.shape[0]
        self.input = input_data
        self.output = np.repeat(np.expand_dims(self.biases, axis=0), n_samples, axis=0)

        with ThreadPoolExecutor() as executor:
            executor.map(self._forward_propagation_helper, product(range(n_samples), range(self.depth), range(self.input_depth)))

        return self.output

    def _forward_propagation_helper(self, args):
        i, j, k = args
        self.output[i, j] += signal.correlate2d(self.input[i, k], self.kernels[j, k], "valid")

    @trace()
    def backward_propagation(self, output_gradients, learning_rate, y_true):
        n_samples = output_gradients.shape[0]
        self._kernels_gradients = np.zeros((n_samples,) + self.kernels_shape)
        self._input_gradients = np.zeros((n_samples,) + self.input_shape)
        self._output_gradients = output_gradients

        with ThreadPoolExecutor() as executor:
            executor.map(self._backward_propagation_helper, product(range(n_samples), range(self.depth), range(self.input_depth)))

        self.kernels -= learning_rate * np.sum(self._kernels_gradients, axis=0)
        self.biases -= learning_rate * np.sum(self._output_gradients, axis=0)

        return self._input_gradients

    def _backward_propagation_helper(self, args):
        i, j, k = args
        self._kernels_gradients[i, j, k] = signal.correlate2d(self.input[i, k], self._output_gradients[i, j], "valid")
        self._input_gradients[i, k] += signal.convolve2d(self._output_gradients[i, j], self.kernels[j, k], "full")
