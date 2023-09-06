from typing import Tuple, Optional, Union, Dict, Any
from abc import abstractmethod
import numpy as np
from . import Module
from neural_network.functions import pair


class Pool2d(Module):
    """
    Abstract base class for 2D pooling layers in neural network architectures.
    Provides common methods and properties for pooling layers.

    Parameters
    ----------
    kernel_size : tuple of int, (default=(2, 2))
        The size of the pooling window in the (height, width) dimensions.
    stride : tuple of int, (default=(1, 1))
        The stride for the pooling operation in the (height, width) dimensions.

    Attributes
    ----------
    kernel_size : tuple of int
        The size of the pooling window.
    stride : tuple of int
        The stride for the pooling operation.
    input_shape : tuple of int
        The shape of the input to the layer.
    output_shape : tuple of int
        The shape of the output from the layer.
    input_dimensions : tuple of int
        The spatial dimensions of the input (height, width).
    output_dimensions : tuple of int
        The spatial dimensions of the output (height, width).

    Methods
    -------
    __init__(self, kernel_size=(2, 2), stride=(1, 1), *args, **kwargs)
        Initialize the 2D pooling layer.
    __repr__(self)
        Return a string representation of the pooling layer.
    _forward_propagation(self, input_data)
        Perform forward propagation through the pooling layer.
    _backward_propagation(self, upstream_gradients, y_true)
        Perform backward propagation through the pooling layer.
    _get_windows(self, input_data)
        Get the windows for the pooling operation.
    """

    def __init__(self, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], *args, **kwargs):
        """
        Initialize the 2D pooling layer.

        Parameters
        ----------
        kernel_size : tuple of int, (default=(2, 2))
            The size of the pooling window in the (height, width) dimensions.
        stride : tuple of int, (default=(1, 1))
            The stride for the pooling operation in the (height, width) dimensions.
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
    def state(self, value: Dict[str, Any]) -> None:
        """
        Set the state of the pooling layer.
        """
        self.kernel_size = pair(value["kernel_size"])
        self.stride = pair(value["stride"])

    @property
    def parameters_count(self) -> int:
        """
        Get the total number of parameters in the module.
        """
        return 0

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape (batch_size, out_channels, output_height, output_width)  of the layer's data.

        Returns:
        --------
        Tuple[int, ...]
            Output shape of the layer's data.
        """
        return self.input.shape[0], self.input.shape[1], self.output_dimensions[0], self.output_dimensions[1]

    @property
    def input_dimensions(self) -> Tuple[int, int]:
        """
        Get the input dimensions (height, width) of the data.
        """
        return self.input.shape[2], self.input.shape[3]

    @property
    def output_dimensions(self) -> Tuple[int, int]:
        """
        Calculate and get the output dimensions (height, width) after pooling.
        """
        output_height = (self.input_dimensions[0] - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (self.input_dimensions[1] - self.kernel_size[1]) // self.stride[1] + 1
        return output_height, output_width

    @property
    def pool_windows(self):
        assert self.input is not None, "Forward was not called before calling pool_windows!"
        return self._get_windows(self.input)

    @abstractmethod
    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Abstract method to implement the actual forward propagation.

        Parameters
        ----------
        input_data : array-like, shape (n_samples, ...)
            The input data to propagate through the layer.
        """
        raise NotImplementedError

    @abstractmethod
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
        pool_windows_shape = (*self.output_shape, *self.kernel_size)
        strides = (
                      input_data.strides[0], input_data.strides[1],
                      self.stride[0] * input_data.strides[2],
                      self.stride[1] * input_data.strides[3],
                      input_data.strides[2], input_data.strides[3]
                  )

        # Create a view into the input data to get the pooling windows
        return np.lib.stride_tricks.as_strided(input_data, shape=pool_windows_shape, strides=strides)


class MaxPool2d(Pool2d):
    """
    Max pooling layer for 2D data in neural network architectures.

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

        self.output = np.nanmax(pool_windows, axis=(4, 5))

        if self.stride == self.kernel_size:
            # If there is no overlapping indices in the windows, this is way faster:
            max_values = self.output.repeat(self.stride[0], axis=2).repeat(self.stride[1], axis=3)
            input_window = input_data[:, :, :self.output_dimensions[0] * self.stride[0], :self.output_dimensions[1] * self.stride[1]]
            # Create a mask indicating the positions of maximum values in the pooling windows
            self.pool_window_indices = np.equal(input_window, max_values).astype(np.int8)
        else:
            # If there is overlapping indices in the windows, this provides an accurate result:
            self.pool_window_indices = np.argmax(pool_windows.reshape((*pool_windows.shape[:-2], -1)), axis=4)
            self.pool_window_indices = np.unravel_index(self.pool_window_indices.ravel(), shape=self.kernel_size)

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

        if self.stride == self.kernel_size:
            # If there is no overlapping indices in the windows, this is way faster:
            gradients = upstream_gradients.repeat(self.stride[0], axis=2).repeat(self.stride[1], axis=3)
            gradients = np.multiply(gradients, self.pool_window_indices)
            self.retrograde[:, :, :gradients.shape[2], :gradients.shape[3]] = gradients
        else:
            # If there is overlapping indices in the windows, this provides an accurate result:
            pool_windows = self._get_windows(self.retrograde)

            for (b, c, h, w), i, j, grad in zip(np.ndindex(*self.output_shape), *self.pool_window_indices, np.nditer(upstream_gradients)):
                pool_windows[b, c, h, w, i, j] += grad


class AvgPool2d(Pool2d):
    """
    Average pooling layer for 2D data in neural network architectures.

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
        self.output = np.nanmean(self.pool_windows, axis=(4, 5))

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

        norm = upstream_gradients / np.prod(self.kernel_size)

        if self.stride == self.kernel_size:
            # If there is no overlapping indices in the windows, this is way faster:
            pool_windows += norm[:, :, :, :, None, None]
        else:
            # If there is overlapping indices in the windows, this provides an accurate result:
            for pos in np.ndindex(*self.output_shape):
                pool_windows[pos] += norm[pos]
