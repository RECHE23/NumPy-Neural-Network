import numpy as np
from typing import Tuple, Iterator, List, Callable, Optional, Dict, Any, Union
from .tools import trace
from .functions import convert_targets
from .callbacks import *
from .modules import *
from .optimizers import *


class NeuralNetwork:
    """
    A customizable neural network model.

    Attributes:
    -----------
    layers : list
        A list to store the layers of the neural network.

    Methods:
    --------
    add(layer: Module) -> None:
        Add a layer to the neural network.

    predict(samples: np.ndarray, to: str = None) -> np.ndarray:
        Make predictions using the neural network.

    fit(samples: np.ndarray, targets: np.ndarray, epochs: int = 100, batch_size: int = 1, shuffle: bool = False, callbacks: List[Callable] = None) -> None:
        Train the neural network.

    Private Attributes:
    -------------------
    _batch_iterator(samples: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool = False) -> Iterator[Tuple[Tuple[int, int], np.ndarray, np.ndarray]]:
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

    def __init__(self, *layers, verbose: bool = True):
        """
        Initialize a neural network model.

        Attributes:
        -----------
        layers : list
            List to store the layers of the neural network.
        """
        self.layers: List[Module] = []
        self.callbacks: List[Callable] = [BaseCallback()]
        self._is_training: bool = False
        self._verbose: bool = verbose

        for layer in layers:
            self.add(layer)

        if self._verbose:
            self.callbacks.append(ProgressCallback())

    def __str__(self):
        """
        Return a string representation of the neural network.

        Returns:
        --------
        str_representation : str
            String representation of the neural network.
        """
        layers_info = "\n".join([f" ({i}) {layer}" for i, layer in enumerate(self.layers)])
        return f"NeuralNetwork:\n{layers_info}\nLearnable parameters count: {self.parameters_count}\n\n"

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
    def add(self, layer: Module) -> None:
        """
        Add a layer to the neural network.

        Parameters:
        -----------
        layer : Module
            The layer to be added to the network.
        """
        # Check compatibility of layer shapes:
        if self.layers and layer.input_shape:
            assert self.layers[-1].output_shape == layer.input_shape, "Previous layer's output shape does not match the new layer's input shape"
        self.layers.append(layer)
        assert sum(isinstance(layer, OutputLayer) for layer in self.layers) <= 1, "Only one output layer can be added."

    @trace()
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the neural network.

        Parameters:
        -----------
        input_data : np.ndarray, shape (n_samples, ...)
            The input data to propagate through the layer.

        Returns:
        --------
        np.ndarray, shape (n_samples, ...)
            The output of the layer after applying its operations.
        """
        predictions = input_data

        for layer in self.layers:
            predictions = layer.forward(predictions)

        return predictions

    @trace()
    def backward(self, upstream_gradients: Optional[np.ndarray] = None, y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform backward propagation through the neural network.

        Parameters:
        -----------
        upstream_gradients : np.ndarray, shape (n_samples, ...)
            Gradients received from the subsequent layer during backward propagation.
        y_true : np.ndarray, shape (n_samples, ...)
            The true target values corresponding to the input data.

        Returns:
        --------
        np.ndarray, shape (n_samples, ...)
            Gradients propagated backward through the layer.
        """
        error_grad = upstream_gradients

        for layer in reversed(self.layers):
            error_grad = layer.backward(error_grad, y_true)

        return error_grad

    @trace()
    def predict(self, samples: np.ndarray, batch_size: int = 64, to: str = None) -> np.ndarray:
        """
        Make predictions using the neural network.

        Parameters:
        -----------
        samples : np.ndarray
            Input samples for prediction.
        batch_size : int, optional
            Batch size for training. Default is 64.
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

        predictions = np.zeros((samples.shape[0], *self.layers[-1].output_shape[1:]))

        batch_size = min(batch_size, samples.shape[0])

        for batch_info, batch_samples in self._batch_iterator(samples, batch_size):
            first_index = (batch_info[0] - 1) * batch_size
            last_index = (batch_info[0] - 1) * batch_size + batch_samples.shape[0]
            predictions[first_index:last_index] = self.forward(batch_samples)

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
            List of callback functions to be called before and after each batch and epoch.
        """
        assert isinstance(self.layers[-1], OutputLayer), "An output layer has to be added before using fit and predict."
        assert samples.shape[0] == targets.shape[0], "The length of the samples doesn't match the length of the targets."
        if self.layers[0].input_shape is not None:
            assert samples.shape[1:] == self.layers[0].input_shape[1:], "Input sample shape does not match the network's input layer shape"

        # Set additional callbacks if provided:
        self.callbacks += callbacks if callbacks else []

        # Converts targets to a one-hot encoding if necessary:
        targets = convert_targets(targets)

        for epoch in range(1, epochs + 1):
            epoch_info = (epoch, epochs)

            # Call on epoch begin callbacks:
            self.call_callbacks(epoch_info, None, samples, targets, status="epoch_begin")

            for batch_info, batch_samples, batch_targets in self._batch_iterator(samples, batch_size, targets, shuffle):

                # Call on batch begin callbacks:
                self.call_callbacks(epoch_info, batch_info, batch_samples, batch_targets, status="batch_begin")

                # Forward propagation:
                batch_predictions = self.forward(batch_samples)

                # Backward propagation:
                self.backward(y_true=batch_targets)

                # Call on batch end callbacks:
                self.call_callbacks(epoch_info, batch_info, batch_predictions, batch_targets, status="batch_end")

            # Call on epoch end callbacks:
            self.call_callbacks(epoch_info, None, samples, targets, status="epoch_end")

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

    @property
    def state(self) -> Dict[str, Any]:
        """
        Return the current state of the neural network.

        Returns:
        --------
        state : Dict[str, Any]
            Dictionary containing the current state of the neural network.
        """
        return {
            "class_name": self.__class__.__name__,
            "layers_state": [layer.state for layer in self.layers]
        }

    @state.setter
    def state(self, value) -> None:
        """
        Set the state of the neural network.

        Parameters:
        -----------
        value : Dict[str, Any]
            Dictionary containing the state to be set.

        Raises:
        ------
        AssertionError
            If the provided class name does not match the current class name.
        """
        assert value["class_name"] == self.__class__.__name__

        for class_name, layer_state in value["layers_state"]:
            layer = globals()[class_name](**layer_state)
            layer.state = layer_state
            self.layers.append(layer)

    def save_state(self, filepath: str) -> None:
        """
        Save the current state (parameters) of the neural network to a file.

        Parameters:
        -----------
        filepath : str
            The path to the file where the state will be saved.
        """
        np.savez(filepath, **self.state)

    def load_state(self, filepath: str) -> None:
        """
        Load the saved state (parameters) of the neural network from a file.

        Parameters:
        -----------
        filepath : str
            The path to the file containing the saved state.
        """
        self.state = np.load(filepath, allow_pickle=True)
        assert self.state["class_name"] == self.__class__.__name__

    @staticmethod
    def _batch_iterator(samples: np.ndarray, batch_size: int, targets: Optional[np.ndarray] = None, shuffle: bool = False) \
            -> Iterator[Union[Tuple[Tuple[int, int], np.ndarray, np.ndarray], Tuple[Tuple[int, int], np.ndarray]]]:
        """
        Generate batches of samples and targets.

        Parameters:
        -----------
        samples : np.ndarray
            Input samples for training.
        batch_size : int
            Batch size.
        targets : np.ndarray, optional
            Target targets for training.
        shuffle : bool, optional
            Whether to shuffle the data before each epoch. Default is False.

        Yields:
        -------
        batch_info : Tuple[int, int]
            A tuple containing the current batch number and the total number of batches.
        batch_samples : np.ndarray
            Batch of input samples.
        batch_targets : np.ndarray, optional
            Batch of target labels if targets are provided.
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

            if targets is not None:
                # Yield batch information, samples, and targets:
                yield (batch_index + 1, total_batches), samples[excerpt], targets[excerpt]
            else:
                # Yield batch information and samples:
                yield (batch_index + 1, total_batches), samples[excerpt]
