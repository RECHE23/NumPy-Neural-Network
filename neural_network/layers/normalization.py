from typing import Optional, Tuple
import numpy as np
from . import Layer


class Normalization(Layer):
    """
    A normalization layer for neural network architectures.

    Parameters:
    -----------
    norm : str, optional
        The normalization method to use. Default is 'minmax'.
    dtype : str, optional
        Data type to which the input data should be converted. Default is 'float32'.
    samples : np.ndarray or None, optional
        The samples to compute normalization parameters. If provided, it precomputes the normalization.
    *args, **kwargs:
        Additional arguments to pass to the base class.

    Attributes:
    -----------
    metric : str
        The normalization metric to use.
    dtype : str
        Data type to which the input data should be converted.
    samples : np.ndarray or None
        The samples used for precomputing normalization parameters.
    norm : float or None
        The normalization parameter.
    input_shape : tuple of int
        The shape of the input to the layer.
    output_shape : tuple of int
        The shape of the output from the layer.

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Compute the normalized output of the normalization layer using the given input data.
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Compute the retrograde gradients for the normalization layer.
    _evaluate_norm(samples: np.ndarray) -> None:
        Compute the normalization parameter based on the provided samples.
    """

    def __init__(self, norm: str = 'minmax', dtype: str = 'float32', samples: Optional[np.ndarray] = None, *args, **kwargs):
        """
        Initialize the Normalization with the given parameters.

        Parameters:
        -----------
        norm : str, optional
            The normalization method to use. Default is 'minmax'.
        dtype : str, optional
            Data type to which the input data should be converted. Default is 'float32'.
        samples : np.ndarray or None, optional
            The samples to compute normalization parameters. If provided, it precomputes the normalization.
        *args, **kwargs:
            Additional arguments to pass to the base class.
        """
        self.metric: str = norm
        self.dtype: str = dtype
        self.samples: Optional[np.ndarray] = samples
        self.norm: Optional[float] = None
        if samples is not None:
            self._evaluate_norm(samples)
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        norm = f"norm={self.norm}, " if self.norm else ""
        return f"{self.__class__.__name__}({norm}dtype={self.dtype})"

    @property
    def parameters_count(self) -> int:
        return 1

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape (batch_size, ...) of the data.
        """
        return self.input.shape

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Compute the normalized output of the normalization layer using the given input data.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data for the normalization layer.
        """
        self.output = input_data.astype(self.dtype)
        if not self.norm:
            self._evaluate_norm(self.output)
        self.output /= self.norm

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: Optional[np.ndarray] = None) -> None:
        """
        Compute the retrograde gradients for the normalization layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Upstream gradients coming from the subsequent layer.
        y_true : np.ndarray
            The true labels used for calculating the retrograde gradient.
        """
        self.retrograde = upstream_gradients * self.norm

    def _evaluate_norm(self, samples: np.ndarray) -> None:
        """
        Compute the normalization parameter based on the provided samples.

        Parameters:
        -----------
        samples : np.ndarray
            The samples to compute normalization parameters.
        """
        samples = samples.astype(self.dtype)
        if self.metric == 'minmax':
            self.norm = np.max(samples) - np.min(samples)
        else:
            self.norm = np.linalg.norm(samples, ord=self.metric)
