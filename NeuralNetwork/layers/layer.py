from abc import abstractmethod
from NeuralNetwork.tools import trace
import itertools


class Layer:
    id_iter = itertools.count()

    def __init__(self):
        self.id = next(self.id_iter)
        self.input = None
        self.output = None
        self.retrograde = None
        self.upstream_gradients = None
        self.n_samples = None

    def __call__(self, *args, **kwargs):
        return self.forward_propagation(*args, **kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return f"{self.__class__.__name__}(Layer #{self.id})"

    @trace()
    def forward_propagation(self, input_data):
        self.input = input_data
        self.n_samples = input_data.shape[0]
        self._forward_propagation(input_data)
        self.n_samples = None
        return self.output

    @abstractmethod
    def _forward_propagation(self, input_data):
        raise NotImplementedError

    @trace()
    def backward_propagation(self, upstream_gradients, learning_rate, y_true):
        self.upstream_gradients = upstream_gradients
        self.n_samples = None if upstream_gradients is None else upstream_gradients.shape[0]
        self._backward_propagation(upstream_gradients, learning_rate, y_true)
        self.n_samples = None
        return self.retrograde

    @abstractmethod
    def _backward_propagation(self, upstream_gradients, learning_rate, y_true):
        raise NotImplementedError
