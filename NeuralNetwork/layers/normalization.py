import numpy as np
from . import Layer


class NormalizationLayer(Layer):
    def __init__(self, norm='minmax', dtype='float32', samples=None, *args, **kwargs):
        self.metric = norm
        self.dtype = dtype
        self.samples = samples
        self.norm = None
        if samples is not None:
            self._evaluate_norm(samples)
        super().__init__(*args, **kwargs)

    def _forward_propagation(self, input_data):
        self.output = input_data.astype(self.dtype)
        if not self.norm:
            self._evaluate_norm(self.output)
        self.output /= self.norm

    def _backward_propagation(self, upstream_gradients, y_true):
        self.retrograde = upstream_gradients * self.norm

    def _evaluate_norm(self, samples):
        samples = samples.astype(self.dtype)
        if self.metric == 'minmax':
            self.norm = np.max(samples) - np.min(samples)
        else:
            self.norm = np.linalg.norm(samples, ord=self.metric)
