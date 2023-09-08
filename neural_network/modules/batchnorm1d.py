from typing import Any, Dict, Optional, Tuple

import numpy as np
from . import Module


class BatchNorm1d(Module):
    """
    A 1D Batch Normalization layer for neural networks.

    Parameters:
    ----------
    num_features : int
        Number of input features.

    eps : float, optional
        A small constant to avoid division by zero, by default 1e-05.

    momentum : float, optional
        The momentum value for updating running statistics during training, by default 0.1.

    *args, **kwargs
        Additional arguments and keyword arguments for the base class.

    Attributes:
    ----------
    num_features : int
        Number of input features.

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

    _forward_propagation(input_data: np.ndarray) -> None
        Perform the forward propagation step.

    _backward_propagation(upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> None
        Perform the backward propagation step.
    """
    num_features: int
    eps: float
    momentum: float
    affine: bool
    gamma: np.ndarray
    beta: np.ndarray
    running_mean: np.ndarray
    running_var: np.ndarray

    def __init__(self, num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine=True, *args, **kwargs):
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
            self.gamma = value.get("gamma", np.ones((1, self.num_features)))
            self.beta = value.get("beta", np.zeros((1, self.num_features)))

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
        Get the output shape (batch_size, num_features)  of the layer's data.

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
        assert len(input_data.shape) == 2, "Input data must have shape (batch, num_features)"

        if self.is_training():
            # Compute mean and variance over the input dimension
            mean = np.nanmean(input_data, axis=0)
            var = np.nanvar(input_data, axis=0)

            # Update running statistics with exponential moving average
            self.running_mean += self.momentum * (mean - self.running_mean)
            self.running_var += self.momentum * (var - self.running_var)

            self.mean = mean.reshape((1, self.num_features))
            self.var = var.reshape((1, self.num_features))
        else:
            self.mean = self.running_mean[None, :]
            self.var = self.running_var[None, :]

        # Normalize input data
        x_normalized = (input_data - self.mean) / np.sqrt(self.var + self.eps)

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
        assert len(upstream_gradients.shape) == 2, "Upstream gradients must have shape (batch, num_features)"

        m = self.input.shape[0]

        x_minus_mean = self.input - self.mean
        sqrt_var_eps = np.sqrt(self.var + self.eps)

        dx_normalized = upstream_gradients * self.gamma if self.affine else upstream_gradients

        # Compute gradients of mean and variance
        dvar = np.sum(dx_normalized * x_minus_mean * -0.5 * (self.var + self.eps) ** (-1.5), axis=0, keepdims=True)
        dmean = np.sum(dx_normalized * -1.0 / sqrt_var_eps, axis=0, keepdims=True) + dvar * np.sum(-2.0 * x_minus_mean, axis=0, keepdims=True) / m

        # Compute retrograde (backward) gradients for input
        self.retrograde = dx_normalized / sqrt_var_eps
        self.retrograde += dvar * 2.0 * x_minus_mean / m
        self.retrograde += dmean / m

        # Compute gradients of gamma and beta and update gamma and beta using optimizer
        if self.affine:
            x_normalized = x_minus_mean / sqrt_var_eps
            dgamma = np.sum(upstream_gradients * x_normalized, axis=0, keepdims=True)
            dbeta = np.sum(upstream_gradients, axis=0, keepdims=True)
            self.optimizer.update([self.gamma, self.beta], [dgamma, dbeta])

