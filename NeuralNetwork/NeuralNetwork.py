import numpy as np
from OutputLayer import OutputLayer
from utils import convert_data


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        assert sum(isinstance(layer, OutputLayer) for layer in self.layers) <= 1

    def predict(self, samples, to=None):
        assert isinstance(self.layers[-1], OutputLayer)

        predictions = samples
        for layer in self.layers:
            predictions = layer.forward_propagation(predictions)

        return convert_data(predictions, to=to)

    def fit(self, samples, labels, epochs=100, learning_rate=0.05):
        assert isinstance(self.layers[-1], OutputLayer)

        for epoch in range(1, epochs + 1):
            error = 0
            for i, propagated_samples in enumerate(samples):
                # Forward propagation:
                for layer in self.layers:
                    propagated_samples = layer.forward_propagation(propagated_samples)

                # Compute the total loss:
                error += self.layers[-1].loss(labels[i], propagated_samples)

                # Backward propagation:
                error_grad = None
                for layer in reversed(self.layers):
                    error_grad = layer.backward_propagation(error_grad, learning_rate, labels[i])

            # Evaluate the average error on all samples:
            error /= len(samples)
            print(f"Epoch {epoch:4d} of {epochs:<4d} \t Error = {error:.6f}")
