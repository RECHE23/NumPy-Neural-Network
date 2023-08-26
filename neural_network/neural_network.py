import numpy as np
from typing import Tuple, Generator, List
from .tools import trace
from .functions import convert_targets
from .layers import OutputLayer, Layer


class NeuralNetwork:
    """
    A customizable neural network model.

    Attributes:
    -----------
    layers : list
        A list to store the layers of the neural network.

    Methods:
    --------
    add(layer: Layer) -> None:
        Add a layer to the neural network.

    predict(samples: np.ndarray, to: str = None) -> np.ndarray:
        Make predictions using the neural network.

    fit(samples: np.ndarray, targets: np.ndarray, epochs: int = 100, batch_size: int = 1, shuffle: bool = False) -> None:
        Train the neural network.

    Attributes (private):
    ---------------------
    _batch_iterator(samples: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool = False) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        Generate batches of samples and targets.

    Special Methods:
    ----------------
    __init__():
        Initialize a neural network model.

    __str__():
        Return a string representation of the neural network.

    __repr__():
        Return a detailed string representation of the neural network.

    __call__(samples: np.ndarray, to: str = None) -> np.ndarray:
        Make predictions using the neural network by calling it as a function.
    """

    def __init__(self):
        """
        Initialize a neural network model.

        Attributes:
        -----------
        layers : list
            List to store the layers of the neural network.
        """
        self.layers: List[Layer] = []

    def __str__(self):
        """
        Return a string representation of the neural network.

        Returns:
        --------
        str_representation : str
            String representation of the neural network.
        """
        layers_info = "\n".join([f"Layer {i}: {repr(layer)}" for i, layer in enumerate(self.layers)])
        return f"NeuralNetwork:\n{layers_info}"

    def __repr__(self):
        """
        Return a detailed string representation of the neural network.

        Returns:
        --------
        str_representation : str
            Detailed string representation of the neural network.
        """
        return f"NeuralNetwork(layers={self.layers})"

    def __call__(self, samples: np.ndarray, to: str = None) -> np.ndarray:
        """
        Make predictions using the neural network by calling it as a function.

        Parameters:
        -----------
        samples : np.ndarray
            Input samples for prediction.
        to : str or None, optional
            Target format to convert predictions to. Default is None.

        Returns:
        --------
        predictions : np.ndarray
            Predictions made by the neural network.
        """
        return self.predict(samples, to=to)

    @trace()
    def add(self, layer: Layer) -> None:
        """
        Add a layer to the neural network.

        Parameters:
        -----------
        layer : Layer
            The layer to be added to the network.
        """
        self.layers.append(layer)
        assert sum(isinstance(layer, OutputLayer) for layer in self.layers) <= 1

    @trace()
    def predict(self, samples: np.ndarray, to: str = None) -> np.ndarray:
        """
        Make predictions using the neural network.

        Parameters:
        -----------
        samples : np.ndarray
            Input samples for prediction.
        to : str or None, optional
            Target format to convert predictions to. Default is None.

        Returns:
        --------
        predictions : np.ndarray
            Predictions made by the neural network.
        """
        assert isinstance(self.layers[-1], OutputLayer)

        predictions = samples
        for layer in self.layers:
            predictions = layer.forward_propagation(predictions)

        return convert_targets(predictions, to=to)

    @trace()
    def fit(self, samples: np.ndarray, targets: np.ndarray, epochs: int = 100, batch_size: int = 1, shuffle: bool = False) -> None:
        """
        Train the neural network.

        Parameters:
        -----------
        samples : np.ndarray
            Input samples for training.
        targets : np.ndarray
            Target labels for training.
        epochs : int, optional
            Number of training epochs. Default is 100.
        batch_size : int, optional
            Batch size for training. Default is 1.
        shuffle : bool, optional
            Whether to shuffle the data before each epoch. Default is False.
        """
        assert samples.shape[0] == targets.shape[0]
        assert isinstance(self.layers[-1], OutputLayer)

        for layer in self.layers:
            layer.is_training(True)

        # Converts targets to a one-hot encoding if necessary:
        targets = convert_targets(targets)

        for epoch in range(1, epochs + 1):
            error = 0
            accuracy = 0
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
                accuracy += np.sum(np.argmax(batch_samples, axis=-1) == convert_targets(batch_labels, to="labels"))

            # Evaluate the average error on all samples:
            error /= len(samples)
            accuracy /= len(samples)
            print(f"Epoch {epoch:4d} of {epochs:<4d} \t Error = {error:.6f} \t On training accuracy = {accuracy:.2%}")

        for layer in self.layers:
            layer.is_training(False)

    @staticmethod
    def _batch_iterator(samples: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool = False) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate batches of samples and targets.

        Parameters:
        -----------
        samples : np.ndarray
            Input samples for training.
        targets : np.ndarray
            Target labels for training.
        batch_size : int
            Batch size.
        shuffle : bool, optional
            Whether to shuffle the data before each epoch. Default is False.

        Yields:
        -------
        batch_samples : np.ndarray
            Batch of input samples.
        batch_targets : np.ndarray
            Batch of target labels.
        """
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
