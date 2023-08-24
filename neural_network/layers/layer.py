from typing import Optional
from abc import abstractmethod
import numpy as np
from neural_network.tools import trace
from neural_network.optimizers import *


class Layer:
    """
    Base class for creating neural network layers.

    This class defines the fundamental structure and behavior of a neural network layer.
    It provides methods for both forward and backward propagation, allowing for the
    building of complex neural network architectures.

    Parameters:
    -----------
    optimizer : Optimizer, optional
        The optimizer to use for updating the layer's parameters during training.
        If not provided, the default optimizer Adam with learning rate 1e-3 and
        decay 1e-4 will be used.

    Attributes:
    -----------
    input : array-like
        The input data passed to the layer during forward propagation.
    output : array-like
        The output data after applying the layer's operations.
    retrograde : array-like
        The gradients propagated backward through the layer during backward propagation.
    upstream_gradients : array-like
        The gradients received from the subsequent layer during backward propagation.
    n_samples : int or None
        The number of samples in the input data. Used for handling batch operations.

    Methods:
    --------
    forward_propagation(input_data) -> np.ndarray
        Perform forward propagation through the layer.
    backward_propagation(upstream_gradients, y_true) -> np.ndarray
        Perform backward propagation through the layer.

    """

    def __init__(self, optimizer: Optional[Optimizer] = None):
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None
        self.retrograde: Optional[np.ndarray] = None
        self.upstream_gradients: Optional[np.ndarray] = None
        self.n_samples: Optional[int] = None
        self.optimizer = optimizer if optimizer is not None else Adam(learning_rate=1e-3, decay=1e-4)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward_propagation(*args, **kwargs)

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @trace()
    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the layer.

        Parameters:
        -----------
        input_data : array-like, shape (n_samples, ...)
            The input data to propagate through the layer.

        Returns:
        --------
        output : array-like, shape (n_samples, ...)
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
        input_data : array-like, shape (n_samples, ...)
            The input data to propagate through the layer.
        """
        raise NotImplementedError

    @trace()
    def backward_propagation(self, upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> np.ndarray:
        """
        Perform backward propagation through the layer.

        Parameters:
        -----------
        upstream_gradients : array-like, shape (n_samples, ...)
            Gradients received from the subsequent layer during backward propagation.
        y_true : array-like, shape (n_samples, ...)
            The true target values corresponding to the input data.

        Returns:
        --------
        retrograde : array-like, shape (n_samples, ...)
            Gradients propagated backward through the layer.
        """
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
        upstream_gradients : array-like, shape (n_samples, ...)
            Gradients received from the subsequent layer during backward propagation.
        y_true : array-like, shape (n_samples, ...)
            The true target values corresponding to the input data.
        """
        raise NotImplementedError
