from typing import Callable
import numpy as np
from . import Layer
from neural_network.functions.activation import activation_functions


class ActivationLayer(Layer):
    """
    A layer implementing activation functions for neural network architectures.

    Parameters:
    -----------
    activation_function : str
        The activation function to be applied during forward and backward propagation.

    Attributes:
    -----------
    activation_function_name: str
        The name of the activation function. Default is "relu".
    activation_function : callable
        The activation function.
    output : np.ndarray
        The output of the activation layer after forward propagation.
    retrograde : np.ndarray
        The gradient used for backward propagation.

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Compute the output of the activation layer using the given input data.
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Compute the retrograde gradient for the activation layer.
    """

    def __init__(self, activation_function: str = "relu", *args, **kwargs):
        """
        Initialize the ActivationLayer with the given activation function.

        Parameters:
        -----------
        activation_function : callable
            The activation function to be applied during forward and backward propagation. Default is "relu".
        """
        self.activation_function_name: str = activation_function.lower().strip()
        self.activation_function: Callable = activation_functions[activation_function]
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(activation_function={self.activation_function_name})"

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Compute the output of the activation layer using the given input data.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data for the activation layer.
        """
        self.output = self.activation_function(input_data)

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Compute the retrograde gradient for the activation layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            The gradient coming from the subsequent layer.
        y_true : np.ndarray
            The true labels used for calculating the retrograde gradient.
        """
        self.retrograde = self.activation_function(self.input, prime=True) * upstream_gradients
