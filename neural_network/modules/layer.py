from typing import Optional, Tuple, Dict, Any
from abc import abstractmethod
import numpy as np
from neural_network.tools import trace
from neural_network.optimizers import Optimizer, Adam


class Layer:
    """
    Base class for creating neural network layers.

    This class defines the fundamental structure and behavior of a neural network layer.
    It provides methods for both forward and backward propagation, allowing for the
    building of complex neural network architectures.

    Attributes:
    -----------
    input : np.ndarray or None
        The input data passed to the layer during forward propagation.
    output : np.ndarray or None
        The output data after applying the layer's operations.
    retrograde : np.ndarray or None
        The gradients propagated backward through the layer during backward propagation.
    upstream_gradients : np.ndarray or None
        The gradients received from the subsequent layer during backward propagation.
    n_samples : int or None
        The number of samples in the input data. Used for handling batch operations.
    optimizer : Optimizer
        The optimizer to use for gradient descent.
    _optimizer_instance : Optimizer or None
        The optimizer instance to update the layer's parameters during training.
    _is_training : bool
        Flag indicating whether the layer is in training mode.

    Methods:
    --------
    forward(input_data: np.ndarray) -> np.ndarray:
        Perform forward propagation through the layer.
    backward(upstream_gradients: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        Perform backward propagation through the layer.

    """

    def __init__(self, optimizer: Optional[Optimizer] = None, *args, **kwargs):
        """
        Initialize a neural network layer.

        Parameters:
        -----------
        optimizer : Optimizer or None, optional
            The optimizer to use for updating the layer's parameters during training.
            If not provided, the default optimizer Adam with learning rate 1e-3 and
            decay 1e-4 will be used.
        """
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None
        self.retrograde: Optional[np.ndarray] = None
        self.upstream_gradients: Optional[np.ndarray] = None
        self.n_samples: Optional[int] = None
        self._optimizer_instance: Optional[Optimizer] = optimizer
        self._is_training: bool = False

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        Make the layer callable for forward propagation.

        Returns:
        --------
        np.ndarray
            Output of the layer after applying its operations.
        """
        return self.forward(*args, **kwargs)

    def __str__(self) -> str:
        """
        Convert the layer to a string representation.

        Returns:
        --------
        str
            String representation of the layer.
        """
        return repr(self)

    def __repr__(self) -> str:
        """
        Convert the layer to a detailed string representation.

        Returns:
        --------
        str
            Detailed string representation of the layer.
        """
        return f"{self.__class__.__name__}()"

    @property
    @abstractmethod
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the current state of the layer.

        Returns:
        --------
        Tuple[str, Dict[str, Any]]
            A tuple containing the class name and a dictionary of the layer's state.
        """
        raise NotImplementedError

    @state.setter
    @abstractmethod
    def state(self, value) -> None:
        """
        Set the state of the layer.

        Parameters:
        -----------
        value : Tuple[str, Dict[str, Any]]
            A tuple containing the class name and a dictionary of the layer's state.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters_count(self) -> int:
        """
        Get the total number of learnable parameters in the layer.

        Returns:
        --------
        int
            Total number of learnable parameters.
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Get the input shape of the layer's data.

        Returns:
        --------
        Tuple[int, ...]
            Input shape of the layer's data.
        """
        return self.input.shape if self.input is not None else None

    @property
    @abstractmethod
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape (batch_size, ...) of the layer's data.

        Returns:
        --------
        Tuple[int, ...]
            Output shape of the layer's data.
        """
        raise NotImplementedError

    def is_training(self, value: Optional[bool] = None) -> bool:
        """
        Get or set the training mode of the layer.

        Parameters:
        -----------
        value : bool, optional
            If provided, set the training mode of the layer.
            If not provided, return the current training mode.

        Returns:
        --------
        bool
            The current training mode of the layer. If 'value' is provided, the updated training mode.
        """
        if value is not None:
            self._is_training = value
        return self._is_training

    @property
    def optimizer(self) -> Optimizer:
        """
        Get the optimizer instance for updating the layer's parameters during training.

        Returns:
        --------
        Optimizer
            The optimizer instance.
        """
        if self._optimizer_instance is None:
            self._optimizer_instance = Adam(learning_rate=1e-3, decay=1e-4)
        return self._optimizer_instance

    @optimizer.setter
    def optimizer(self, value: Optimizer) -> None:
        """
        Set the optimizer instance for updating the layer's parameters during training.

        Parameters:
        -----------
        value : Optimizer
            The optimizer instance.
        """
        self._optimizer_instance = value

    @trace()
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the layer.

        Parameters:
        -----------
        input_data : np.ndarray, shape (n_samples, ...)
            The input data to propagate through the layer.

        Returns:
        --------
        np.ndarray, shape (n_samples, ...)
            The output of the layer after applying its operations.
        """
        self.input = input_data
        self.n_samples = input_data.shape[0]
        self._forward_propagation(input_data)
        self.n_samples = None
        return self.output

    @abstractmethod
    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Abstract method to implement the actual forward propagation.

        Parameters:
        -----------
        input_data : np.ndarray, shape (n_samples, ...)
            The input data to propagate through the layer.
        """
        raise NotImplementedError

    @trace()
    def backward(self, upstream_gradients: Optional[np.ndarray], y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform backward propagation through the layer.

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
        assert self.input is not None, "You need a forward propagation before a backward propagation..."
        self.upstream_gradients = upstream_gradients
        self.n_samples = None if upstream_gradients is None else upstream_gradients.shape[0]
        self._backward_propagation(upstream_gradients, y_true)
        self.n_samples = None
        return self.retrograde

    @abstractmethod
    def _backward_propagation(self, upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> None:
        """
        Abstract method to implement the actual backward propagation.

        Parameters:
        -----------
        upstream_gradients : np.ndarray, shape (n_samples, ...)
            Gradients received from the subsequent layer during backward propagation.
        y_true : np.ndarray, shape (n_samples, ...)
            The true target values corresponding to the input data.
        """
        raise NotImplementedError
