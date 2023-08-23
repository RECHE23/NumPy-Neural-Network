from . import Layer


class ActivationLayer(Layer):
    def __init__(self, activation_function, *args, **kwargs):
        self.activation_function = activation_function
        super().__init__(*args, **kwargs)

    def _forward_propagation(self, input_data):
        self.output = self.activation_function(self.input)

    def _backward_propagation(self, upstream_gradients, y_true):
        self.retrograde = self.activation_function(self.input, prime=True) * upstream_gradients