from typing import Tuple, Optional
from abc import abstractmethod
import numpy as np
from . import Layer
from neural_network.functions import pair


class Pooling2DLayer(Layer):
    """
    Abstract base class for 2D pooling layers in neural network architectures.
    Provides common methods and properties for pooling layers.

    Parameters
    ----------
    pool_size : tuple of int, (default=(2, 2))
        The size of the pooling window in the (height, width) dimensions.
    stride : tuple of int, (default=(1, 1))
        The stride for the pooling operation in the (height, width) dimensions.

    Attributes
    ----------
    pool_size : tuple of int
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
    __init__(self, pool_size=(2, 2), stride=(1, 1), *args, **kwargs)
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

    def __init__(self, pool_size: Tuple[int, int], stride: Tuple[int, int], *args, **kwargs):
        """
        Initialize the 2D pooling layer.

        Parameters
        ----------
        pool_size : tuple of int, (default=(2, 2))
            The size of the pooling window in the (height, width) dimensions.
        stride : tuple of int, (default=(1, 1))
            The stride for the pooling operation in the (height, width) dimensions.
        *args, **kwargs:
            Additional arguments to pass to the base class.
        """
        super().__init__(*args, **kwargs)
        self.pool_size: Tuple[int, int] = pair(pool_size)
        self.stride: Tuple[int, int] = pair(stride)

    def __repr__(self) -> str:
        """
        Return a string representation of the pooling layer.
        """
        return f"{self.__class__.__name__}(pool_size={self.pool_size}, stride={self.stride})"

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape (batch_size, output_channels, output_height, output_width) of the data.
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
        output_height = (self.input_dimensions[0] - self.pool_size[0]) // self.stride[0] + 1
        output_width = (self.input_dimensions[1] - self.pool_size[1]) // self.stride[1] + 1
        return output_height, output_width

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
    def _backward_propagation(self, upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> None:
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
        pool_windows_shape = (*self.output_shape, *self.pool_size)
        strides = (
                      input_data.strides[0], input_data.strides[1],
                      self.stride[0] * input_data.strides[2],
                      self.stride[1] * input_data.strides[3],
                      input_data.strides[2], input_data.strides[3]
                  )

        # Create a view into the input data to get the pooling windows
        return np.lib.stride_tricks.as_strided(input_data, shape=pool_windows_shape, strides=strides)


class MaxPooling2DLayer(Pooling2DLayer):
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
        # Create a view into the input data to get the pooling windows
        pool_windows = self._get_windows(input_data)

        # Evaluate the max pooling
        self.output = np.nanmax(pool_windows, axis=(4, 5))

        max_values = self.output.repeat(self.stride[0], axis=2).repeat(self.stride[1], axis=3)
        input_window = input_data[:, :, :self.output_dimensions[0] * self.stride[0], :self.output_dimensions[1] * self.stride[1]]
        # Create a mask indicating the positions of maximum values in the pooling windows
        self.mask = np.equal(input_window, max_values).astype(np.int8)

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Perform backward propagation for max pooling layer.

        Parameters
        ----------
        upstream_gradients : np.ndarray
            Gradients received from the subsequent layer during backward propagation.
        y_true : np.ndarray
            The true target values corresponding to the input data.
        """
        # Initialize retrograde array
        self.retrograde = np.zeros_like(self.input)

        if self.stride == self.pool_size:  # Overlapping indices can cause bugs in memory access.
            # Upsample the gradients and apply the mask
            gradients = upstream_gradients.repeat(self.stride[0], axis=2).repeat(self.stride[1], axis=3)
            gradients = np.multiply(gradients, self.mask)
            self.retrograde[:, :, :gradients.shape[2], :gradients.shape[3]] = gradients
        else:
            # Unpooling with non-overlapping strides
            batch_size, input_channels, output_height, output_width = self.output_shape
            for i in range(output_height):
                for j in range(output_width):
                    start_i, start_j = i * self.stride[0], j * self.stride[1]
                    end_i, end_j = start_i + self.pool_size[0], start_j + self.pool_size[1]

                    # Find the indices of max values in the pooling window
                    pooling_window = self.input[:, :, start_i:end_i, start_j:end_j].reshape(batch_size, input_channels, -1)
                    max_indices_i, max_indices_j = np.unravel_index(np.argmax(pooling_window, axis=2), self.pool_size)

                    # Distribute gradients to the positions of max values in the input window
                    for b in range(batch_size):
                        for c in range(input_channels):
                            self.retrograde[b, c, start_i:end_i, start_j:end_j][max_indices_i[b, c], max_indices_j[b, c]] = upstream_gradients[b, c, i, j]


class AveragePooling2DLayer(Pooling2DLayer):
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
        # Create a view into the input data to get the pooling windows
        pool_windows = self._get_windows(input_data)

        # Evaluate the average pooling
        self.output = np.nanmean(pool_windows, axis=(4, 5))

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Perform backward propagation for average pooling layer.

        Parameters
        ----------
        upstream_gradients : np.ndarray
            Gradients received from the subsequent layer during backward propagation.
        y_true : np.ndarray
            The true target values corresponding to the input data.
        """
        output_height, output_width = self.output_dimensions

        self.retrograde = np.zeros_like(self.input)

        for i in range(output_height):
            for j in range(output_width):
                start_i, start_j = i * self.stride[0], j * self.stride[1]
                end_i, end_j = start_i + self.pool_size[0], start_j + self.pool_size[1]
                self.retrograde[:, :, start_i:end_i, start_j:end_j] = upstream_gradients[:, :, i, j][:, :, None, None] / (self.pool_size[0] * self.pool_size[1])
