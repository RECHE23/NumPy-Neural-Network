from typing import Tuple, Optional, Dict, Any
import numpy as np
from . import Module


class BatchNorm2d(Module):
    """
    A 2D Batch Normalization layer for neural networks.

    Parameters:
    ----------
    num_features : int
        Number of input channels.

    eps : float, optional
        A small constant to avoid division by zero, by default 1e-05.

    momentum : float, optional
        The momentum value for updating running statistics during training, by default 0.1.

    *args, **kwargs
        Additional arguments and keyword arguments for the base class.

    Attributes:
    ----------
    num_features : int
        Number of input channels.

    eps : float
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
        self.num_features: int
        self.eps: float
        self.momentum: float
        self.affine: bool
        self.gamma: np.ndarray
        self.beta: np.ndarray
        self.running_mean: np.ndarray
        self.running_var: np.ndarray

        super().__init__(*args, **kwargs)
        self.state = {
            "num_features": num_features,
            "eps": eps,
            "momentum": momentum,
            "affine": affine,
        }

    def __repr__(self) -> str:
        """
        Return a string representation of the batch normalization layer.
        """
        return f"{self.__class__.__name__}({self.num_features}, eps={self.eps}, momentum={self.momentum})"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the current state of the batch normalization layer.
        """
        state = self.__class__.__name__, {
            "num_features": self.num_features,
            "eps": self.eps,
            "momentum": self.momentum,
            "affine": self.affine,
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }

        if self.affine:
            state[1]["gamma"] = self.gamma
            state[1]["beta"] = self.beta

        return state

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """
        Set the state of the batch normalization layer.
        """
        assert value["num_features"] > 0, "Number of features must be greater than 0"
        assert value["eps"] > 0, "Epsilon must be positive for numerical stability"
        assert 0 <= value["momentum"] < 1, "Momentum must be in the range [0, 1)"

        self.num_features = value["num_features"]
        self.eps = value["eps"]
        self.momentum = value["momentum"]
        self.affine = value["affine"]

        if self.affine:
            self.gamma = value.get("gamma", np.ones((1, self.num_features, 1, 1)))
            self.beta = value.get("beta", np.zeros((1, self.num_features, 1, 1)))

        self.running_mean = value.get("running_mean", np.zeros(self.num_features))
        self.running_var = value.get("running_var", np.ones(self.num_features))

    @property
    def parameters_count(self) -> int:
        """
        Get the total number of parameters in the module.
        """
        c = np.prod(self.running_mean.shape) + np.prod(self.running_var.shape)
        if self.affine:
            c += np.prod(self.gamma.shape) + np.prod(self.beta.shape)
        return c

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape (batch_size, out_channels, output_height, output_width)  of the layer's data.

        Returns:
        --------
        Tuple[int, ...]
            Output shape of the layer's data.
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
        assert len(input_data.shape) == 4, "Input data must have shape (batch, num_features, height, width)"

        if self.is_training():
            # Compute mean and variance over spatial dimensions (H x W)
            mean = np.nanmean(input_data, axis=(0, 2, 3))
            var = np.nanvar(input_data, axis=(0, 2, 3))

            # Update running statistics with exponential moving average
            self.running_mean += self.momentum * (mean - self.running_mean)
            self.running_var += self.momentum * (var - self.running_var)

            self.mean = mean.reshape((1, self.num_features, 1, 1))
            self.var = var.reshape((1, self.num_features, 1, 1))
        else:
            self.mean = self.running_mean[None, :, None, None]
            self.var = self.running_var[None, :, None, None]

        # Normalize input data
        sqrt_var_eps = np.sqrt(self.var + self.eps)
        x_normalized = (input_data - self.mean) / sqrt_var_eps

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
        assert len(upstream_gradients.shape) == 4, "Upstream gradients must have shape (batch, num_features, height, width)"

        m = self.input.shape[0] * self.input.shape[2] * self.input.shape[3]

        x_minus_mean = self.input - self.mean
        sqrt_var_eps = np.sqrt(self.var + self.eps)

        dx_normalized = upstream_gradients * self.gamma if self.affine else upstream_gradients

        # Compute gradients of mean and variance
        dvar = np.sum(dx_normalized * x_minus_mean * -0.5 * (self.var + self.eps) ** (-1.5), axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dx_normalized * -1.0 / sqrt_var_eps, axis=(0, 2, 3), keepdims=True) + dvar * np.sum(-2.0 * x_minus_mean, axis=(0, 2, 3), keepdims=True) / m

        # Compute retrograde (backward) gradients for input
        self.retrograde = dx_normalized / sqrt_var_eps
        self.retrograde += dvar * 2.0 * x_minus_mean / m
        self.retrograde += dmean / m

        # Compute gradients of gamma and beta and update gamma and beta using optimizer
        if self.affine:
            x_normalized = x_minus_mean / sqrt_var_eps
            dgamma = np.sum(upstream_gradients * x_normalized, axis=(0, 2, 3), keepdims=True)
            dbeta = np.sum(upstream_gradients, axis=(0, 2, 3), keepdims=True)
            self.optimizer.update([self.gamma, self.beta], [dgamma, dbeta])
