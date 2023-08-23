from typing import Union, Tuple, Optional, Iterable
import numpy as np
from . import Layer
from neural_network.functions import correlate2d, convolve2d, parallel_iterator, pair


class Convolutional2DLayer(Layer):
    """
    A 2D convolutional layer for neural network architectures.

    Parameters:
    -----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels (number of kernels).
    kernel_size : int or tuple of int
        Size of the convolutional kernels.
    input_shape : tuple of int
        Input shape (height, width) of the data.
    initialization : str, optional
        Weight initialization method: "xavier" or "he" (default is "xavier").
    *args, **kwargs:
        Additional arguments to pass to the base class.

    Attributes:
    -----------
    kernels_gradients : np.ndarray
        Gradients of the kernels.
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    kernel_shape : tuple of int
        Shape of the convolutional kernels.
    input_shape : tuple of int
        Input shape (height, width) of the data.
    kernels : np.ndarray
        Convolutional kernels.
    output_shape : tuple of int
        Output shape after convolution.
    biases : np.ndarray
        Biases for each output channel.

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Compute the output of the convolutional layer using the given input data.
    _forward_propagation_helper(args: tuple) -> None:
        Helper function for parallel forward propagation computation.
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Compute the retrograde gradients for the convolutional layer.
    _backward_propagation_helper(args: tuple) -> None:
        Helper function for parallel backward propagation computation.
    _initialize_parameters_xavier()
        Initialize layer parameters using Xavier initialization.
    _initialize_parameters_he()
        Initialize layer parameters using He initialization.
    """

    def __init__(self, input_channels: int, output_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 input_shape: Tuple[int, int], initialization: str = "xavier", *args, **kwargs):
        """
        Initialize the Convolutional2DLayer with the given parameters.

        Parameters:
        -----------
        input_channels : int
            Number of input channels.
        output_channels : int
            Number of output channels (number of kernels).
        kernel_size : int or tuple of int
            Size of the convolutional kernels.
        input_shape : tuple of int
            Input shape (height, width) of the data.
        initialization : str, optional
            Weight initialization method: "xavier" or "he" (default is "xavier").
        *args, **kwargs:
            Additional arguments to pass to the base class.
        """
        super().__init__(*args, **kwargs)

        self.kernels_gradients: Optional[np.ndarray] = None
        self.input_channels: int = input_channels
        self.output_channels: int = output_channels
        self.kernel_shape: Tuple[int, int] = pair(kernel_size)
        self.input_shape: Tuple[int, int] = input_shape
        self.output_shape: Tuple[int, int] = (self.input_shape[0] - self.kernel_shape[0] + 1, self.input_shape[1] - self.kernel_shape[1] + 1)

        if initialization == "xavier":
            self._initialize_parameters_xavier()
        elif initialization == "he":
            self._initialize_parameters_he()
        else:
            raise ValueError("Invalid initialization method. Use 'xavier' or 'he'.")

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Compute the output of the convolutional layer using the given input data.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data for the convolutional layer.
        """
        biases_tiled = np.tile(self.biases[:, np.newaxis, np.newaxis], (1, *self.output_shape))
        self.output = np.repeat(np.expand_dims(biases_tiled, axis=0), self.n_samples, axis=0)

        parallel_iterator(self._forward_propagation_helper,
                          range(self.n_samples), range(self.output_channels), range(self.input_channels))

    def _forward_propagation_helper(self, args: Tuple[Iterable, Iterable, Iterable]) -> None:
        """
        Helper function for parallel forward propagation computation.

        Parameters:
        -----------
        args : tuple
            Tuple of arguments: (sample, kernel_layer, input_layer).
        """
        sample, kernel_layer, input_layer = args
        input_data = self.input[sample, input_layer]
        kernel = self.kernels[kernel_layer, input_layer]

        self.output[sample, kernel_layer] += correlate2d(input_data, kernel, "valid")

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Compute the retrograde gradients for the convolutional layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Upstream gradients coming from the subsequent layer.
        y_true : np.ndarray
            The true labels used for calculating the retrograde gradient.
        """
        self.kernels_gradients = np.empty((self.n_samples, *self.kernels.shape))
        self.retrograde = np.zeros((self.n_samples, self.input_channels, *self.input_shape))

        parallel_iterator(self._backward_propagation_helper,
                          range(self.n_samples), range(self.output_channels), range(self.input_channels))

        self.optimizer.update([self.kernels, self.biases],
                              [np.sum(self.kernels_gradients, axis=0), np.sum(upstream_gradients, axis=(0, 2, 3))])

    def _backward_propagation_helper(self, args: Tuple[Iterable, Iterable, Iterable]) -> None:
        """
        Helper function for parallel backward propagation computation.

        Parameters:
        -----------
        args : tuple
            Tuple of arguments: (sample, kernel_layer, input_layer).
        """
        sample, kernel_layer, input_layer = args
        input_data = self.input[sample, input_layer]
        upstream_gradient = self.upstream_gradients[sample, kernel_layer]
        kernel = self.kernels[kernel_layer, input_layer]

        self.kernels_gradients[sample, kernel_layer, input_layer] = correlate2d(input_data, upstream_gradient, "valid")
        self.retrograde[sample, input_layer] += convolve2d(upstream_gradient, kernel, "full")

    def _initialize_parameters_xavier(self) -> None:
        """
        Initialize the convolutional kernels using the Xavier initialization method.
        """
        a = np.sqrt(6 / (np.prod(self.input_shape) + np.prod(self.kernel_shape)))
        self.kernels = np.random.uniform(-a, a, (self.output_channels, self.input_channels, *self.kernel_shape))
        self.biases = np.zeros((self.output_channels,))

    def _initialize_parameters_he(self) -> None:
        """
        Initialize the convolutional kernels using the He initialization method.
        """
        a = np.sqrt(2 / self.input_channels)
        self.kernels = np.random.normal(0, a, (self.output_channels, self.input_channels, *self.kernel_shape))
        self.biases = np.zeros((self.output_channels,))
