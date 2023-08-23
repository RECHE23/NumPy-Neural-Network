import numpy as np
from tools import trace
from functions import convert_targets, accuracy_score, f1_score
from layers import OutputLayer


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    @trace()
    def add(self, layer):
        self.layers.append(layer)
        assert sum(isinstance(layer, OutputLayer) for layer in self.layers) <= 1

    @trace()
    def predict(self, samples, to=None):
        assert isinstance(self.layers[-1], OutputLayer)

        predictions = samples
        for layer in self.layers:
            predictions = layer.forward_propagation(predictions)

        return convert_targets(predictions, to=to)

    @trace()
    def fit(self, samples, targets, epochs=100, batch_size=1, shuffle=False):
        assert samples.shape[0] == targets.shape[0]
        assert isinstance(self.layers[-1], OutputLayer)

        # Converts targets to a one hot encoding if necessary:
        targets = convert_targets(targets)

        for epoch in range(1, epochs + 1):
            error = 0
            for batch_samples, batch_labels in self._batch_iterator(samples, targets, batch_size, shuffle):
                # Forward propagation:
                for layer in self.layers:
                    batch_samples = layer.forward_propagation(batch_samples)

                # Backward propagation:
                error_grad = None
                for layer in reversed(self.layers):
                    error_grad = layer.backward_propagation(error_grad, batch_labels)
                    layer.optimizer.next_epoch()

                # Compute the total loss:
                error += self.layers[-1].loss(batch_labels, batch_samples)

            # Evaluate the average error on all samples:
            error /= len(samples)
            predictions = self.predict(samples)
            accuracy = accuracy_score(predictions, targets)
            f1 = f1_score(predictions, targets)
            print(f"Epoch {epoch:4d} of {epochs:<4d} \t Error = {error:.6f} \t Train set accuracy = {accuracy:.2%} \t F1 Score = {f1:.2%}")

    @staticmethod
    def _batch_iterator(samples, targets, batch_size, shuffle=False):
        if shuffle:
            indices = np.arange(samples.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, samples.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, samples.shape[0])
            if shuffle:
                excerpt = indices[start_idx:end_idx]
            else:
                excerpt = slice(start_idx, end_idx)
            yield samples[excerpt], targets[excerpt]
