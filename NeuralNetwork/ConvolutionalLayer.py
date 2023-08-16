import numpy as np
from scipy import signal
from utils import trace
from Layer import Layer


class ConvolutionalLayer(Layer):
    def __init__(self, input_shape, kernel_size, depth):
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

        for i in range(n_samples):
            for j in range(self.depth):
                for k in range(self.input_depth):
                    self.output[i, j] += signal.correlate2d(self.input[i, k], self.kernels[j, k], "valid")

        return self.output

    @trace()
    def backward_propagation(self, output_gradients, learning_rate, y_true):
        n_samples = output_gradients.shape[0]
        kernels_gradients = np.zeros((n_samples,) + self.kernels_shape)
        input_gradients = np.zeros((n_samples,) + self.input_shape)

        for i in range(n_samples):
            for j in range(self.depth):
                for k in range(self.input_depth):
                    kernels_gradients[i, j, k] = signal.correlate2d(self.input[i, k], output_gradients[i, j], "valid")
                    input_gradients[i, k] += signal.convolve2d(output_gradients[i, j], self.kernels[j, k], "full")

        self.kernels -= learning_rate * np.sum(kernels_gradients, axis=0)
        self.biases -= learning_rate * np.sum(output_gradients, axis=0)

        return input_gradients
