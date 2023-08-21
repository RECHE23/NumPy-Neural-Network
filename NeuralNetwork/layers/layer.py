from abc import abstractmethod


class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.retrograde = None

    @abstractmethod
    def forward_propagation(self, input_data):
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, upstream_gradients, learning_rate, y_true):
        raise NotImplementedError
