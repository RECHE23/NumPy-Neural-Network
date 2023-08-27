from typing import Tuple, Union, List
import numpy as np
from . import Layer


class Reshape(Layer):
    """
    A layer for reshaping the input data to a specified shape.

    Parameters:
    -----------
    output_shape : tuple
        Desired shape of the output data.
    *args, **kwargs:
        Additional arguments to pass to the base class.

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Reshape the input data to the specified output shape.
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Reshape the upstream gradients to match the input shape.

    Attributes:
    -----------
    input_shape : tuple or None
        Shape of the input data.
    output_shape : tuple
        Desired shape of the output data.
    """

    def __init__(self, output_shape: Tuple[int, ...], *args, **kwargs):
        """
        Initialize the Reshape with the desired output shape.

        Parameters:
        -----------
        output_shape : tuple
            Desired shape of the output data.
        *args, **kwargs:
            Additional arguments to pass to the base class.
        """
        self._output_shape: Tuple[int, ...] = output_shape
        super().__init__(*args, **kwargs)

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape (batch_size, output_shape) of the data.
        """
        return self._output_shape

    def __repr__(self) -> str:
        input_shape = f"input_shape={self.input_shape}, " if self.input else ""
        return f"{self.__class__.__name__}({input_shape}output_shape={self.output_shape})"

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Reshape the input data to the specified output shape.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data to be reshaped.
        """
        self.output = np.reshape(input_data, (-1, *self.output_shape))

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Reshape the upstream gradients to match the input shape.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Upstream gradients coming from the subsequent layer.
        y_true : np.ndarray
            The true labels (not used in this layer).
        """
        self.retrograde = np.reshape(upstream_gradients, (-1, *self.input_shape[1:]))


class Flatten(Layer):
    """
    A layer for flattening the input data with specified start and end dimensions.

    Parameters:
    -----------
    start_dim : int, optional
        The dimension to start flattening from. Default is 1.
    end_dim : int, optional
        The dimension to end flattening at. Default is -1.

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Flatten the input data with specified dimensions.
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Reshape the upstream gradients back to the original shape.

    Attributes:
    -----------
    original_shape : Tuple[int, ...] or None
        Shape of the input data before flattening.
    start_dim : int
        The dimension to start flattening from.
    end_dim : int
        The dimension to end flattening at.
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1, *args, **kwargs):
        """
        Initialize the Flatten layer with start and end dimensions.

        Parameters:
        -----------
        start_dim : int, optional
            The dimension to start flattening from. Default is 1.
        end_dim : int, optional
            The dimension to end flattening at. Default is -1.
        *args, **kwargs:
            Additional arguments to pass to the base class.
        """
        self.start_dim = start_dim
        self.end_dim = end_dim
        super().__init__(*args, **kwargs)

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Flatten the input data with specified dimensions.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data to be flattened.
        """
        self.original_shape = input_data.shape
        self.output = np.reshape(input_data, (input_data.shape[0], -1))

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Reshape the upstream gradients back to the original shape.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Upstream gradients coming from the subsequent layer.
        y_true : np.ndarray
            The true labels (not used in this layer).
        """
        self.retrograde = np.reshape(upstream_gradients, self.original_shape)


class Unflatten(Layer):
    """
    A layer for unflattening the input data.

    Parameters:
    -----------
    dim : Union[int, str]
        Dimension along which the tensor was flattened.
    unflattened_size : Union[Tuple[int, ...], List[int]]
        Desired shape of the unflattened dimension.

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Reshape the input data to the specified output shape.
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Reshape the upstream gradients back to the original shape.

    Attributes:
    -----------
    original_shape : Tuple[int, ...] or None
        Shape of the input data before unflattening.
    """

    def __init__(self, dim: Union[int, str], unflattened_size: Union[Tuple[int, ...], List[int]], *args, **kwargs):
        """
        Initialize the Unflatten with the desired output shape.

        Parameters:
        -----------
        dim : Union[int, str]
            Dimension along which the tensor was flattened.
        unflattened_size : Union[Tuple[int, ...], List[int]]
            Desired shape of the unflattened dimension.
        *args, **kwargs:
            Additional arguments to pass to the base class.
        """
        self._dim = dim
        self._unflattened_size = unflattened_size
        super().__init__(*args, **kwargs)

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Reshape the input data to the specified output shape.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data to be reshaped.
        """
        self.original_shape = input_data.shape
        self.output = np.reshape(input_data, (-1, *self._unflattened_size))

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Reshape the upstream gradients back to the original shape.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Upstream gradients coming from the subsequent layer.
        y_true : np.ndarray
            The true labels (not used in this layer).
        """
        self.retrograde = np.reshape(upstream_gradients, self.original_shape)
