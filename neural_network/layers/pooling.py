from typing import Tuple, Optional
from abc import abstractmethod
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from . import Layer
from neural_network.functions import pair


class Pooling2DLayer(Layer):
    """
    Abstract base class for 2D pooling layers in neural network architectures.
    Provides common methods and properties for pooling layers.

    Attributes:
    -----------
    pool_size : tuple of int
        Size of the pooling window.
    stride : tuple of int
        Stride (vertical stride, horizontal stride).

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Abstract method to perform forward propagation for pooling layers.
    _backward_propagation(upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> None:
        Abstract method to perform backward propagation for pooling layers.
    """

    def __init__(self, pool_size: Tuple[int, int], stride: Tuple[int, int], *args, **kwargs):
        """
        Initialize a Pooling2DLayer.

        Parameters:
        -----------
        pool_size : tuple of int
            Size of the pooling window.
        stride : tuple of int
            Stride (vertical stride, horizontal stride).
        *args, **kwargs:
            Additional arguments to pass to the base class.
        """
        super().__init__(*args, **kwargs)
        self.pool_size: Tuple[int, int] = pair(pool_size)
        self.stride: Tuple[int, int] = pair(stride)

    @property
    def input_shape(self) -> Tuple[int, int]:
        """
        Get the input shape (height, width) of the data.
        """
        return self.input.shape[2], self.input.shape[3]

    @property
    def output_shape(self) -> Tuple[int, int]:
        """
        Calculate and get the output shape (height, width) after pooling.
        """
        output_height = (self.input_shape[0] - self.pool_size[0]) // self.stride[0] + 1
        output_width = (self.input_shape[1] - self.pool_size[1]) // self.stride[1] + 1
        return output_height, output_width

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


class MaxPooling2DLayer(Pooling2DLayer):
    """
    Max pooling layer for 2D data in neural network architectures.

    Parameters:
    -----------
    pool_size : tuple of int
        Size of the pooling window.
    stride : tuple of int
        Stride (vertical stride, horizontal stride).
    *args, **kwargs:
        Additional arguments to pass to the base class.
    """

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform forward propagation using max pooling.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data for the max pooling layer.
        """
        # Retrieve input dimensions and output dimensions
        batch_size, input_channels, input_height, input_width = input_data.shape
        output_height, output_width = self.output_shape

        # Initialize output and max indices arrays
        self.output = np.zeros((batch_size, input_channels, output_height, output_width))
        self.max_indices = np.zeros_like(self.output, dtype=np.int32)

        # Iterate over each pooling region and calculate max values and indices
        for i in range(output_height):
            for j in range(output_width):
                start_i, start_j = i * self.stride[0], j * self.stride[1]
                end_i, end_j = start_i + self.pool_size[0], start_j + self.pool_size[1]

                pool_slice = input_data[:, :, start_i:end_i, start_j:end_j]
                pool_windows = sliding_window_view(pool_slice, (1, 1) + self.pool_size)

                pool_flat = pool_windows.reshape(batch_size, input_channels, -1)
                max_indices_local = np.argmax(pool_flat, axis=2)

                self.max_indices[:, :, i, j] = max_indices_local
                self.output[:, :, i, j] = np.max(pool_flat, axis=2)

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Perform backward propagation for max pooling layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Gradients received from the subsequent layer during backward propagation.
        y_true : np.ndarray
            The true target values corresponding to the input data.
        """
        batch_size, input_channels, input_height, input_width = self.input.shape
        output_height, output_width = self.output_shape

        self.retrograde = np.zeros_like(self.input)

        for i in range(output_height):
            for j in range(output_width):
                start_i, start_j = i * self.stride[0], j * self.stride[1]
                end_i, end_j = start_i + self.pool_size[0], start_j + self.pool_size[1]
                max_indices = np.argmax(self.input[:, :, start_i:end_i, start_j:end_j].reshape(batch_size, input_channels, -1), axis=2)
                max_indices_i, max_indices_j = np.unravel_index(max_indices, self.pool_size)

                for b in range(batch_size):
                    for c in range(input_channels):
                        self.retrograde[b, c, start_i:end_i, start_j:end_j][max_indices_i[b, c], max_indices_j[b, c]] = upstream_gradients[b, c, i, j]


class AveragePooling2DLayer(Pooling2DLayer):
    """
    Average pooling layer for 2D data in neural network architectures.

    Parameters:
    -----------
    pool_size : tuple of int
        Size of the pooling window.
    stride : tuple of int
        Stride (vertical stride, horizontal stride).
    *args, **kwargs:
        Additional arguments to pass to the base class.
    """

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform forward propagation using average pooling.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data for the average pooling layer.
        """
        batch_size, input_channels, input_height, input_width = input_data.shape
        output_height, output_width = self.output_shape

        self.output = np.zeros((batch_size, input_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                start_i, start_j = i * self.stride[0], j * self.stride[1]
                end_i, end_j = start_i + self.pool_size[0], start_j + self.pool_size[1]
                self.output[:, :, i, j] = np.mean(input_data[:, :, start_i:end_i, start_j:end_j], axis=(2, 3))

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Perform backward propagation for average pooling layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Gradients received from the subsequent layer during backward propagation.
        y_true : np.ndarray
            The true target values corresponding to the input data.
        """
        output_height, output_width = self.output_shape

        self.retrograde = np.zeros_like(self.input)

        for i in range(output_height):
            for j in range(output_width):
                start_i, start_j = i * self.stride[0], j * self.stride[1]
                end_i, end_j = start_i + self.pool_size[0], start_j + self.pool_size[1]
                self.retrograde[:, :, start_i:end_i, start_j:end_j] = upstream_gradients[:, :, i, j][:, :, None, None] / (self.pool_size[0] * self.pool_size[1])
