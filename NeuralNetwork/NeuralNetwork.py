from utils import convert_targets, batch_iterator
from OutputLayer import OutputLayer


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

        return convert_targets(predictions, to=to)

    def fit(self, samples, targets, epochs=100, learning_rate=0.05, batch_size=1, shuffle=False):
        assert isinstance(self.layers[-1], OutputLayer)

        # Converts targets to a one hot encoding if necessary:
        targets = convert_targets(targets)

        for epoch in range(1, epochs + 1):
            error = 0
            for batch_samples, batch_labels in batch_iterator(samples, targets, batch_size, shuffle):
                # Forward propagation:
                for layer in self.layers:
                    batch_samples = layer.forward_propagation(batch_samples)

                # Compute the total loss:
                error += self.layers[-1].loss(batch_labels, batch_samples)

                # Backward propagation:
                error_grad = None
                for layer in reversed(self.layers):
                    error_grad = layer.backward_propagation(error_grad, learning_rate, batch_labels)

            # Evaluate the average error on all samples:
            error /= len(samples)
            print(f"Epoch {epoch:4d} of {epochs:<4d} \t Error = {error:.6f}")
