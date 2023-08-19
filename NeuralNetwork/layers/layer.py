class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.retrograde = None

    def forward_propagation(self, input_data):
        raise NotImplementedError

    def backward_propagation(self, upstream_gradients, learning_rate, y_true):
        raise NotImplementedError
