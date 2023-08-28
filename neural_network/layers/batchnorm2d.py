from typing import Tuple, Optional
import numpy as np
from . import Layer


class BatchNorm2d(Layer):
    """
    A 2D Batch Normalization layer for neural networks.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.

    eps : float, optional
        A small constant to avoid division by zero, by default 1e-05.

    momentum : float, optional
        The momentum value for updating running statistics during training, by default 0.1.

    *args, **kwargs
        Additional arguments and keyword arguments for the base class.

    Attributes:
    ----------
    in_channels : int
        Number of input channels.

    epsilon : float
        A small constant to avoid division by zero.

    momentum : float
        The momentum value for updating running statistics during training.

    gamma : np.ndarray
        Scaling parameter for normalization.

    beta : np.ndarray
        Bias parameter for normalization.

    running_mean : np.ndarray
        Exponential moving average of mean during training.

    running_var : np.ndarray
        Exponential moving average of variance during training.

    Methods:
    -------
    __repr__() -> str
        Return a string representation of the batch normalization layer.

    output_shape() -> Tuple[int, ...]
        Get the output shape of the layer.

    _forward_propagation(input_data: np.ndarray) -> None
        Perform the forward propagation step.

    _backward_propagation(upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> None
        Perform the backward propagation step.
    """

    def __init__(self, num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = num_features
        self.epsilon = eps
        self.momentum = momentum
        self.affine = affine

        if self.affine:
            self.gamma = np.ones((1, self.input_channels, 1, 1))
            self.beta = np.zeros((1, self.input_channels, 1, 1))

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def __repr__(self) -> str:
        """
        Return a string representation of the batch normalization layer.
        """
        return f"{self.__class__.__name__}({self.input_channels}, eps={self.epsilon}, momentum={self.momentum})"

    @property
    def parameters_count(self) -> int:
        c = np.prod(self.running_mean.shape) + np.prod(self.running_var.shape)
        if self.affine:
            c += np.prod(self.gamma.shape) + np.prod(self.beta.shape)
        return c

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape of the layer.

        Returns:
        ----------
        output_shape : tuple
            The shape of the layer's output data.
        """
        return self.input.shape

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform the forward propagation step.

        Parameters:
        ----------
        input_data : np.ndarray
            The input data.
        """
        if self.is_training():
            # Compute mean and variance over spatial dimensions (H x W)
            mean = np.nanmean(input_data, axis=(0, 2, 3))
            var = np.nanvar(input_data, axis=(0, 2, 3))

            # Update running statistics with exponential moving average
            self.running_mean += self.momentum * (mean - self.running_mean)
            self.running_var += self.momentum * (var - self.running_var)

            self.mean = mean.reshape((1, self.input_channels, 1, 1))
            self.var = var.reshape((1, self.input_channels, 1, 1))
        else:
            self.mean = self.running_mean[None, :, None, None]
            self.var = self.running_var[None, :, None, None]

        # Normalize input data
        x_normalized = (input_data - self.mean) / np.sqrt(self.var + self.epsilon)
        self.output = self.gamma * x_normalized + self.beta if self.affine else x_normalized

    def _backward_propagation(self, upstream_gradients: Optional[np.ndarray], y_true: Optional[np.ndarray] = None) -> None:
        """
        Perform the backward propagation step.

        Parameters:
        ----------
        upstream_gradients : np.ndarray or None
            Gradients flowing in from the layer above.

        y_true : np.ndarray
            The true labels for the data.
        """
        m = self.input.shape[0] * self.input.shape[2] * self.input.shape[3]

        x_minus_mean = self.input - self.mean
        sqrt_var_eps = np.sqrt(self.var + self.epsilon)

        x_normalized = x_minus_mean / sqrt_var_eps

        # Compute gradients of gamma and beta
        dgamma = np.sum(upstream_gradients * x_normalized, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(upstream_gradients, axis=(0, 2, 3), keepdims=True)

        dx_normalized = upstream_gradients * self.gamma if self.affine else upstream_gradients

        # Compute gradients of mean and variance
        dvar = np.sum(dx_normalized * x_minus_mean * -0.5 * (self.var + self.epsilon) ** (-1.5), axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dx_normalized * -1.0 / sqrt_var_eps, axis=(0, 2, 3), keepdims=True) + dvar * np.sum(-2.0 * x_minus_mean, axis=(0, 2, 3), keepdims=True) / m

        # Compute retrograde (backward) gradients for input
        self.retrograde = dx_normalized / sqrt_var_eps
        self.retrograde += dvar * 2.0 * x_minus_mean / m
        self.retrograde += dmean / m

        # Update gamma and beta using optimizer
        if self.affine:
            self.optimizer.update([self.gamma, self.beta], [dgamma, dbeta])
