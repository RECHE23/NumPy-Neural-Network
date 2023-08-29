import numpy as np
from typing import Tuple, Iterator, List, Callable, Optional
from .tools import trace
from .functions import convert_targets
from .layers import OutputLayer, Layer
from .callbacks import ProgressCallback


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

    fit(samples: np.ndarray, targets: np.ndarray, epochs: int = 100, batch_size: int = 1, shuffle: bool = False, callbacks: List[Callable] = None) -> None:
        Train the neural network.

    Private Attributes:
    -------------------
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
        self.callbacks: List[Callable] = [ProgressCallback()]
        self._is_training: bool = False

    def __str__(self):
        """
        Return a string representation of the neural network.

        Returns:
        --------
        str_representation : str
            String representation of the neural network.
        """
        layers_info = "\n".join([f"Layer {i}: {layer}" for i, layer in enumerate(self.layers)])
        return f"NeuralNetwork:\n{layers_info}\nLearnable parameters count: {self.parameters_count}"

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

    @property
    def parameters_count(self):
        """
        Calculate the total number of learnable parameters in the network.

        Returns:
        --------
        count : int
            Total number of learnable parameters.
        """
        return sum(layer.parameters_count for layer in self.layers)

    @trace()
    def add(self, layer: Layer) -> None:
        """
        Add a layer to the neural network.

        Parameters:
        -----------
        layer : Layer
            The layer to be added to the network.
        """
        # Check compatibility of layer shapes:
        if self.layers and layer.input_shape:
            assert self.layers[-1].output_shape == layer.input_shape, "Previous layer's output shape does not match the new layer's input shape"
        self.layers.append(layer)
        assert sum(isinstance(layer, OutputLayer) for layer in self.layers) <= 1, "Only one output layer can be added."

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
        assert self.layers, "No layers in the neural network. Add layers before making predictions."
        assert isinstance(self.layers[-1], OutputLayer), "An output layer has to be added before using fit and predict."
        assert samples.shape[1:] == self.layers[0].input_shape[1:], "Input sample shape does not match the network's input layer shape"

        predictions = samples
        for layer in self.layers:
            predictions = layer.forward(predictions)

        return convert_targets(predictions, to=to)

    @trace()
    def fit(self, samples: np.ndarray, targets: np.ndarray, epochs: int = 100, batch_size: int = 1,
            shuffle: bool = False, callbacks: List[Callable] = None) -> None:
        """
        Train the neural network.

        Parameters:
        -----------
        samples : np.ndarray
            Input samples for training.
        targets : np.ndarray
            Target targets for training.
        epochs : int, optional
            Number of training epochs. Default is 100.
        batch_size : int, optional
            Batch size for training. Default is 1.
        shuffle : bool, optional
            Whether to shuffle the data before each epoch. Default is False.
        callbacks : list of callable functions, optional
            List of callback functions to be called after each batch and epoch.
        """
        assert isinstance(self.layers[-1], OutputLayer), "An output layer has to be added before using fit and predict."
        assert samples.shape[0] == targets.shape[0], "The length of the samples doesn't match the length of the targets."
        if self.layers[0].input_shape is not None:
            assert samples.shape[1:] == self.layers[0].input_shape[1:], "Input sample shape does not match the network's input layer shape"

        # Set additional callbacks if provided:
        self.callbacks += callbacks if callbacks else []

        self.is_training(True)

        # Converts targets to a one-hot encoding if necessary:
        targets = convert_targets(targets)

        for epoch in range(1, epochs + 1):
            epoch_info = (epoch, epochs)

            # Call on epoch begin callbacks:
            self.call_callbacks(epoch_info, None, samples, targets, status="epoch_begin")

            for batch_info, batch_samples, batch_targets in self._batch_iterator(samples, targets, batch_size, shuffle):

                # Call on batch begin callbacks:
                self.call_callbacks(epoch_info, batch_info, batch_samples, batch_targets, status="batch_begin")

                # Forward propagation:
                for layer in self.layers:
                    batch_samples = layer.forward(batch_samples)

                # Backward propagation:
                error_grad = None
                for layer in reversed(self.layers):
                    error_grad = layer.backward(error_grad, batch_targets)
                    layer.optimizer.next_epoch()
                    if hasattr(layer, 'weights'):
                        assert not np.any(np.isnan(layer.weights)), "NaN values detected in layer weights"
                        assert not np.any(np.isinf(layer.weights)), "Infinity values detected in layer weights"
                    if hasattr(layer, 'bias'):
                        assert not np.any(np.isnan(layer.bias)), "NaN values detected in layer weights"
                        assert not np.any(np.isinf(layer.bias)), "Infinity values detected in layer weights"

                # Call on batch end callbacks:
                self.call_callbacks(epoch_info, batch_info, batch_samples, batch_targets, status="batch_end")

            # Call on epoch end callbacks:
            self.call_callbacks(epoch_info, None, samples, targets, status="epoch_end")

        self.is_training(False)

    def call_callbacks(self, epoch_info, batch_info, batch_samples, batch_targets, status):
        """
        Call registered callbacks.

        Parameters:
        -----------
        epoch_info : Tuple[int, int]
            Current epoch number and total number of epochs.
        batch_info : Tuple[int, int]
            Current batch number and total number of batches.
        batch_samples : np.ndarray
            Batch of input samples.
        batch_targets : np.ndarray
            Batch of target labels.
        status : str
            Current callback status ("batch_begin", "batch_end", "epoch_begin", "epoch_end").
        """
        for callback in self.callbacks:
            callback(self, epoch_info, batch_info, batch_samples, batch_targets, status)

    def is_training(self, value: Optional[bool] = None) -> bool:
        """
        Get or set the training status of the neural network.

        Parameters:
        -----------
        value : bool, optional
            If provided, set the training status to this value. If None, return the current training status.

        Returns:
        --------
        training_status : bool
            Current training status of the neural network.
        """
        if value is not None:
            self._is_training = value
            for layer in self.layers:
                layer.is_training(value)
        return self._is_training

    @staticmethod
    def _batch_iterator(samples: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool = False) \
            -> Iterator[Tuple[Tuple[int, int], np.ndarray, np.ndarray]]:
        """
        Generate batches of samples and targets.

        Parameters:
        -----------
        samples : np.ndarray
            Input samples for training.
        targets : np.ndarray
            Target targets for training.
        batch_size : int
            Batch size.
        shuffle : bool, optional
            Whether to shuffle the data before each epoch. Default is False.

        Yields:
        -------
        batch_info : Tuple[int, int]
            A tuple containing the current batch number and the total number of batches.
        batch_samples : np.ndarray
            Batch of input samples.
        batch_targets : np.ndarray
            Batch of target labels.
        """
        assert batch_size <= samples.shape[0], "Batch size cannot be larger than the number of samples"

        if shuffle:
            indices = np.arange(samples.shape[0])
            np.random.shuffle(indices)

        # Generate ranges for batch start indices:
        start_indices_range = range(0, samples.shape[0], batch_size)
        total_batches = len(start_indices_range)

        # Iterate over batches:
        for batch_index, start_idx in enumerate(start_indices_range):
            end_idx = min(start_idx + batch_size, samples.shape[0])
            if shuffle:
                excerpt = indices[start_idx:end_idx]
            else:
                excerpt = slice(start_idx, end_idx)

            # Yield batch information, samples, and targets:
            yield (batch_index + 1, total_batches), samples[excerpt], targets[excerpt]
