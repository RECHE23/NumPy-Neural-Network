import numpy as np
from . import Layer
from NeuralNetwork.functions import correlate2d, convolve2d, parallel_iterator
from NeuralNetwork.tools import trace


class Convolutional2DLayer(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        self.kernels_gradients = None
        self.input_shape = input_depth, input_height, input_width = input_shape
        # Xavier initialization:
        a = np.sqrt(6 / (input_height * input_width + kernel_size * kernel_size))
        self.kernels = np.random.uniform(-a, a, (depth, input_depth, kernel_size, kernel_size))
        self.biases = np.zeros((depth, input_height - kernel_size + 1, input_width - kernel_size + 1))
        super().__init__()

    @trace()
    def _forward_propagation(self, input_data):
        n_samples = input_data.shape[0]
        self.output = np.repeat(np.expand_dims(self.biases, axis=0), n_samples, axis=0)

        parallel_iterator(self._forward_propagation_helper,
                          range(n_samples), range(self.kernels.shape[0]), range(self.input_shape[0]))

    def _forward_propagation_helper(self, args):
        sample, kernel_layer, input_layer = args
        self.output[sample, kernel_layer] += \
            correlate2d(self.input[sample, input_layer], self.kernels[kernel_layer, input_layer], "valid")

    @trace()
    def _backward_propagation(self, upstream_gradients, learning_rate, y_true):
        n_samples = upstream_gradients.shape[0]
        self.kernels_gradients = np.empty((n_samples,) + self.kernels.shape)
        self.retrograde = np.zeros((n_samples,) + self.input_shape)

        parallel_iterator(self._backward_propagation_helper,
                          range(n_samples), range(self.kernels.shape[0]), range(self.input_shape[0]))

        self.kernels -= learning_rate * np.sum(self.kernels_gradients, axis=0)
        self.biases -= learning_rate * np.sum(upstream_gradients, axis=0)

    def _backward_propagation_helper(self, args):
        sample, kernel_layer, input_layer = args
        self.kernels_gradients[sample, kernel_layer, input_layer] = \
            correlate2d(self.input[sample, input_layer], self.upstream_gradients[sample, kernel_layer], "valid")
        self.retrograde[sample, input_layer] += \
            convolve2d(self.upstream_gradients[sample, kernel_layer], self.kernels[kernel_layer, input_layer], "full")
