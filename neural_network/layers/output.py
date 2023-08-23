from typing import Union, Callable
import numpy as np
from . import Layer


class OutputLayer(Layer):
    """
    An output layer for neural network architectures.

    Parameters:
    -----------
    activation_function : callable
        Activation function applied to the layer's output.
    loss_function : callable
        Loss function to calculate the loss between predicted and true values.
    *args, **kwargs:
        Additional arguments to pass to the base class.

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Compute the output of the output layer using the given input data.
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Compute the retrograde gradients for the output layer.
    loss(y_true: np.ndarray, y_pred: np.ndarray, prime: bool = False) -> float:
        Compute the loss between true and predicted values.

    Attributes:
    -----------
    activation_function : callable
        Activation function applied to the layer's output.
    loss_function : callable
        Loss function to calculate the loss between predicted and true values.
    """

    def __init__(self, activation_function: Callable, loss_function: Callable, *args, **kwargs):
        """
        Initialize the OutputLayer with the given activation and loss functions.

        Parameters:
        -----------
        activation_function : callable
            Activation function applied to the layer's output.
        loss_function : callable
            Loss function to calculate the loss between predicted and true values.
        *args, **kwargs:
            Additional arguments to pass to the base class.
        """
        self.activation_function: Callable = activation_function
        self.loss_function: Callable = loss_function
        super().__init__(*args, **kwargs)

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Compute the output of the output layer using the given input data.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data for the output layer.
        """
        self.output = self.activation_function(self.input)

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Compute the retrograde gradients for the output layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Upstream gradients coming from the subsequent layer.
        y_true : np.ndarray
            The true labels used for calculating the retrograde gradient.
        """
        self.retrograde = self.loss_function(y_true, self.output, prime=True)

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray, prime: bool = False) -> Union[float, np.ndarray]:
        """
        Compute the loss between true and predicted values.

        Parameters:
        -----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted values.
        prime : bool, optional
            If True, compute the derivative of the loss with respect to y_pred. Default is False.

        Returns:
        --------
        loss : float
            The computed loss value.
        """
        return np.sum(self.loss_function(y_true, y_pred, prime))
