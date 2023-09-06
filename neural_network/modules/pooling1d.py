from typing import Tuple, Optional, Dict, Any
import numpy as np
from . import Module


class Pool1d(Module):
    """
    Abstract base class for 1D pooling layers in neural network architectures.
    Provides common methods and properties for pooling layers.

    Parameters
    ----------
    kernel_size : int, (default=2)
        The size of the pooling window in the width dimension.
    stride : int, (default=1)
        The stride for the pooling operation in the width dimension.

    Attributes
    ----------
    kernel_size : int
        The size of the pooling window.
    stride : int
        The stride for the pooling operation.
    input_shape : tuple of int
        The shape of the input to the layer.
    output_shape : tuple of int
        The shape of the output from the layer.
    input_length : int
        The length of the input sequence.
    output_length : int
        The length of the output sequence.

    Methods
    -------
    __init__(self, kernel_size=2, stride=1, *args, **kwargs)
        Initialize the 1D pooling layer.
    __repr__(self)
        Return a string representation of the pooling layer.
    _forward_propagation(self, input_data)
        Perform forward propagation through the pooling layer.
    _backward_propagation(self, upstream_gradients, y_true)
        Perform backward propagation through the pooling layer.
    _get_windows(self, input_data)
        Get the windows for the pooling operation.
    """

    def __init__(self, kernel_size: int = 2, stride: int = 1, *args, **kwargs):
        """
        Initialize the 1D pooling layer.

        Parameters
        ----------
        kernel_size : int, (default=2)
            The size of the pooling window in the width dimension.
        stride : int, (default=1)
            The stride for the pooling operation in the width dimension.
        *args, **kwargs:
            Additional arguments to pass to the base class.
        """
        super().__init__(*args, **kwargs)

        self.state = {
            "kernel_size": kernel_size,
            "stride": stride
        }

    def __repr__(self) -> str:
        """
        Return a string representation of the pooling layer.
        """
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride})"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the current state of the pooling layer.
        """
        return self.__class__.__name__, {
            "kernel_size": self.kernel_size,
            "stride": self.stride
        }

    @state.setter
    def state(self, value) -> None:
        """
        Set the state of the pooling layer.
        """
        self.kernel_size = value["kernel_size"]
        self.stride = value["stride"]

    @property
    def parameters_count(self) -> int:
        """
        Get the total number of parameters in the layer.
        """
        return 0

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape (batch_size, out_channels, output_length) of the layer's data.

        Returns:
        --------
        Tuple[int, ...]
            Output shape of the layer's data.
        """
        return self.input.shape[0], self.input.shape[1], self.output_length

    @property
    def input_length(self) -> int:
        """
        Get the length of the input sequence.
        """
        return self.input.shape[2]

    @property
    def output_length(self) -> int:
        """
        Calculate and get the length of the output sequence after pooling.
        """
        return (self.input_length - self.kernel_size) // self.stride + 1

    @property
    def pool_windows(self):
        assert self.input is not None, "Forward was not called before calling pool_windows!"
        return self._get_windows(self.input)

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Abstract method to implement the actual forward propagation.

        Parameters
        ----------
        input_data : array-like, shape (n_samples, ...)
            The input data to propagate through the layer.
        """
        raise NotImplementedError

    def _backward_propagation(self, upstream_gradients: Optional[np.ndarray], y_true: Optional[np.ndarray] = None) -> None:
        """
        Abstract method to implement the actual backward propagation.

        Parameters
        ----------
        upstream_gradients : array-like, shape (n_samples, ...)
            Gradients received from the subsequent layer during backward propagation.
        y_true : array-like, shape (n_samples, ...)
            The true target values corresponding to the input data.
        """
        raise NotImplementedError

    def _get_windows(self, input_data: np.ndarray) -> np.ndarray:
        """
        Get the windows for the pooling operation.

        Parameters
        ----------
        input_data : np.ndarray
            The input data to the layer.

        Returns
        -------
        np.ndarray
            Pooling windows.
        """
        # Calculate the parameters for creating the windows
        pool_windows_shape = (*self.output_shape, self.kernel_size)
        strides = (
            input_data.strides[0], input_data.strides[1],
            self.stride * input_data.strides[2],
            input_data.strides[2]
        )

        # Create a view into the input data to get the pooling windows
        return np.lib.stride_tricks.as_strided(input_data, shape=pool_windows_shape, strides=strides)


class MaxPool1d(Pool1d):
    """
    Max pooling layer for 1D data in neural network architectures.

    Methods
    -------
    _forward_propagation(self, input_data)
        Perform forward propagation through the max pooling layer.
    _backward_propagation(self, upstream_gradients, y_true)
        Perform backward propagation through the max pooling layer.
    """

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform forward propagation using max pooling.

        Parameters
        ----------
        input_data : np.ndarray
            The input data for the max pooling layer.
        """
        pool_windows = self.pool_windows

        self.output = np.nanmax(pool_windows, axis=3)

        self.pool_window_indices = np.argmax(pool_windows, axis=3)

    def _backward_propagation(self, upstream_gradients: Optional[np.ndarray], y_true: Optional[np.ndarray] = None) -> None:
        """
        Perform backward propagation for max pooling layer.

        Parameters
        ----------
        upstream_gradients : np.ndarray
            Gradients received from the subsequent layer during backward propagation.
        y_true : np.ndarray
            The true target values corresponding to the input data.
        """
        assert hasattr(self, 'pool_window_indices'), "No pool indices found. Make sure forward propagation was performed."

        self.retrograde = np.zeros_like(self.input)

        pool_windows = self._get_windows(self.retrograde)

        for (b, c, w), i, grad in zip(np.ndindex(*self.output_shape), np.nditer(self.pool_window_indices), np.nditer(upstream_gradients)):
            pool_windows[b, c, w, i] += grad


class AvgPool1d(Pool1d):
    """
    Average pooling layer for 1D data in neural network architectures.

    Methods
    -------
    _forward_propagation(self, input_data)
        Perform forward propagation through the average pooling layer.
    _backward_propagation(self, upstream_gradients, y_true)
        Perform backward propagation through the average pooling layer.
    """

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform forward propagation using average pooling.

        Parameters
        ----------
        input_data : np.ndarray
            The input data for the average pooling layer.
        """
        self.output = np.nanmean(self.pool_windows, axis=3)

    def _backward_propagation(self, upstream_gradients: Optional[np.ndarray], y_true: Optional[np.ndarray] = None) -> None:
        """
        Perform backward propagation for average pooling layer.

        Parameters
        ----------
        upstream_gradients : np.ndarray
            Gradients received from the subsequent layer during backward propagation.
        y_true : np.ndarray
            The true target values corresponding to the input data.
        """
        self.retrograde = np.zeros_like(self.input)

        pool_windows = self._get_windows(self.retrograde)

        norm = upstream_gradients / self.kernel_size

        for pos, n in zip(np.ndindex(*self.output_shape), np.nditer(norm)):
            pool_windows[pos] += n
