import numpy as np
from . import Layer


class ReshapeLayer(Layer):
    def __init__(self, output_shape, dtype='float32', *args, **kwargs):
        self.dtype = dtype
        self.input_shape = None
        self.output_shape = output_shape
        super().__init__(*args, **kwargs)

    def _forward_propagation(self, input_data):
        if not self.input_shape:
            self.input_shape = input_data.shape[1:]
        self.output = np.reshape(input_data.astype(self.dtype), (-1, ) + self.output_shape)

    def _backward_propagation(self, upstream_gradients, y_true):
        self.retrograde = np.reshape(upstream_gradients, (-1,) + self.input_shape)
