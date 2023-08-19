import numpy as np
from . import Layer
from NeuralNetwork.tools import trace


class NormalizationLayer(Layer):
    def __init__(self, norm='minmax', dtype='float32', samples=None):
        self.metric = norm
        self.dtype = dtype
        self.samples = samples
        self.norm = None
        if samples is not None:
            self._evaluate_norm(samples)
        super().__init__()

    @trace()
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data.astype(self.dtype)
        if not self.norm:
            self._evaluate_norm(self.output)
        self.output /= self.norm
        return self.output

    @trace()
    def backward_propagation(self, upstream_gradients, learning_rate, y_true):
        self.retrograde = upstream_gradients * self.norm
        return self.retrograde

    def _evaluate_norm(self, samples):
        samples = samples.astype(self.dtype)
        if self.metric == 'minmax':
            self.norm = np.max(samples) - np.min(samples)
        else:
            self.norm = np.linalg.norm(samples, ord=self.metric)
