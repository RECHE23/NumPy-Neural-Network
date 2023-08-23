import numpy as np
from . import Layer
from neural_network.functions import correlate2d, convolve2d, parallel_iterator, pair


class Convolutional2DLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, input_shape, *args, **kwargs):
        self.kernels_gradients = None
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_shape = pair(kernel_size)
        self.input_shape = input_height, input_width = input_shape
        # Xavier initialization:
        a = np.sqrt(6 / (np.prod(self.input_shape) + np.prod(self.kernel_shape)))
        self.kernels = np.random.uniform(-a, a, (output_channels, input_channels, *self.kernel_shape))
        self.output_shape = (input_height - self.kernel_shape[0] + 1, input_width - self.kernel_shape[1] + 1)
        self.biases = np.zeros((output_channels, ))
        super().__init__(*args, **kwargs)

    def _forward_propagation(self, input_data):
        biases_tiled = np.tile(self.biases[:, np.newaxis, np.newaxis], (1, *self.output_shape))
        self.output = np.repeat(np.expand_dims(biases_tiled, axis=0), self.n_samples, axis=0)

        parallel_iterator(self._forward_propagation_helper,
                          range(self.n_samples), range(self.output_channels), range(self.input_channels))

    def _forward_propagation_helper(self, args):
        sample, kernel_layer, input_layer = args
        input_data = self.input[sample, input_layer]
        kernel = self.kernels[kernel_layer, input_layer]

        self.output[sample, kernel_layer] += correlate2d(input_data, kernel, "valid")

    def _backward_propagation(self, upstream_gradients, y_true):
        self.kernels_gradients = np.empty((self.n_samples, *self.kernels.shape))
        self.retrograde = np.zeros((self.n_samples, self.input_channels, *self.input_shape))

        parallel_iterator(self._backward_propagation_helper,
                          range(self.n_samples), range(self.output_channels), range(self.input_channels))

        self.optimizer.update([self.kernels, self.biases], [np.sum(self.kernels_gradients, axis=0), np.sum(upstream_gradients, axis=(0, 2, 3))])

    def _backward_propagation_helper(self, args):
        sample, kernel_layer, input_layer = args
        input_data = self.input[sample, input_layer]
        upstream_gradient = self.upstream_gradients[sample, kernel_layer]
        kernel = self.kernels[kernel_layer, input_layer]

        self.kernels_gradients[sample, kernel_layer, input_layer] = correlate2d(input_data, upstream_gradient, "valid")
        self.retrograde[sample, input_layer] += convolve2d(upstream_gradient, kernel, "full")
