import numpy as np

from OutputLayer import OutputLayer


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        assert sum(isinstance(layer, OutputLayer) for layer in self.layers) <= 1

    def predict(self, samples):
        assert isinstance(self.layers[-1], OutputLayer)

        predictions = np.empty((len(samples), self.layers[-1].output.shape[1]))

        for i, propagated_sample in enumerate(samples):
            for layer in self.layers:
                propagated_sample = layer.forward_propagation(propagated_sample)
            predictions[i] = propagated_sample

        return predictions

    def fit(self, samples, labels, epochs=100, learning_rate=0.1):
        assert isinstance(self.layers[-1], OutputLayer)

        for epoch in range(1, epochs + 1):
            error = 0
            for i, propagated_sample in enumerate(samples):
                # Forward propagation:
                for layer in self.layers:
                    propagated_sample = layer.forward_propagation(propagated_sample)

                # Compute the total loss:
                error += self.layers[-1].loss(labels[i], propagated_sample)

                # Backward propagation:
                error_grad = None
                for layer in reversed(self.layers):
                    error_grad = layer.backward_propagation(error_grad, learning_rate, labels[i])

            # Evaluate the average error on all samples:
            error /= len(samples)
            print(f"Epoch {epoch:4d} of {epochs:<4d} \t Error = {error:.6f}")
