from . import Layer


class ActivationLayer(Layer):
    def __init__(self, activation_function):
        self.activation_function = activation_function
        super().__init__()

    def _forward_propagation(self, input_data):
        self.output = self.activation_function(self.input)

    def _backward_propagation(self, upstream_gradients, learning_rate, y_true):
        self.retrograde = self.activation_function(self.input, prime=True) * upstream_gradients
