import numpy as np
from utils import trace
from Layer import Layer


class NormalizationLayer(Layer):
    def __init__(self, dtype='float32', norm='minmax'):
        self.dtype = dtype
        self.shape = None
        self.metric = norm
        self.norm = None
        super().__init__()

    @trace()
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data.reshape(input_data.shape[0], -1)
        self.output = self.output.astype(self.dtype)
        if not (self.shape and self.norm):
            self.shape = input_data.shape[1:]
            if self.metric == 'minmax':
                self.norm = np.max(self.output) - np.min(self.output)
            else:
                self.norm = np.linalg.norm(self.output, ord=self.metric)
        self.output -= np.min(self.output)
        self.output /= self.norm
        return self.output

    @trace()
    def backward_propagation(self, output_error, learning_rate, y_true):
        return output_error * self.norm
