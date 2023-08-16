import numpy as np
from utils import trace
from Layer import Layer


class ReshapeLayer(Layer):
    def __init__(self, output_shape, dtype='float32'):
        self.dtype = dtype
        self.input_shape = None
        self.output_shape = output_shape
        super().__init__()

    @trace()
    def forward_propagation(self, input_data):
        self.input = input_data
        if not self.input_shape:
            self.input_shape = input_data.shape[1:]
        self.output = np.reshape(input_data.astype(self.dtype), (-1, ) + self.output_shape)
        return self.output

    @trace()
    def backward_propagation(self, output_gradient, learning_rate, y_true):
        return np.reshape(output_gradient, (-1, ) + self.input_shape)
