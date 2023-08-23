from typing import Union, Tuple, Optional
import numpy as np
from . import Layer
from neural_network.functions import pair


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
    stride : tuple of int
        Stride (vertical stride, horizontal stride).
    padding : tuple of int
        Padding (vertical padding, horizontal padding).
    initialization : str, optional
        Weight initialization method: "xavier" or "he" (default is "xavier").
    *args, **kwargs:
        Additional arguments to pass to the base class.

    Attributes:
    -----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    kernel_size : tuple of int
        Shape of the convolutional kernels.
    stride : tuple of int
        Stride (vertical stride, horizontal stride).
    padding : tuple of int
        Padding (vertical padding, horizontal padding).
    weights : np.ndarray
        Convolutional kernels.
    bias : np.ndarray
        Biases for each output channel.
    initialization : str
        Weight initialization method.
    input_shape : tuple of int
        Input shape (height, width) of the data.
    windows : np.ndarray
        Memorized convolution windows from the forward propagation.
    """

    def __init__(self, input_channels: int, output_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                 initialization: str = "xavier", *args, **kwargs):
        """
        Initialize a Convolutional2DLayer.

        Parameters
        ----------
        input_channels : int
            Number of input channels.
        output_channels : int
            Number of output channels (number of kernels).
        kernel_size : int or tuple of int
            Size of the convolutional kernels.
        stride : tuple of int, optional
            Stride (vertical stride, horizontal stride). Defaults to (1, 1).
        padding : tuple of int, optional
            Padding (vertical padding, horizontal padding). Defaults to (0, 0).
        initialization : str, optional
            Weight initialization method: "xavier" or "he" (default is "xavier").
        *args, **kwargs:
            Additional arguments to pass to the base class.
        """
        super().__init__(*args, **kwargs)

        self.input_channels: int = input_channels
        self.output_channels: int = output_channels
        self.kernel_size: Tuple[int, int] = pair(kernel_size)
        self.stride: Tuple[int, int] = pair(stride)
        self.padding: Tuple[int, int] = pair(padding)
        self.initialization: str = initialization
        self.input_shape: Optional[Tuple[int, int]] = None
        self.windows: Optional[np.ndarray] = None

        if initialization == "xavier":
            self._initialize_parameters_xavier()
        elif initialization == "he":
            self._initialize_parameters_he()
        else:
            raise ValueError("Invalid initialization method. Use 'xavier' or 'he'.")

    def __repr__(self) -> str:
        """
        Return a string representation of the Convolutional2DLayer.
        """
        return (
            f"{self.__class__.__name__}(input_channels={self.input_channels}, output_channels="
            f"{self.output_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding="
            f"{self.padding}, optimizer={self.optimizer}, initialization={self.initialization})"
        )

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Compute the output of the convolutional layer using the given input data.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data for the convolutional layer.
        """
        # Compute output shape:
        self.input_shape = h, w = input_data.shape[2:]
        out_h = (h - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        out_w = (w - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1

        # Generate windows:
        self.windows = self._get_windows(input_data, (out_h, out_w), self.kernel_size,
                                         padding=self.padding,
                                         stride=self.stride)

        # Perform convolution and add bias:
        self.output = np.einsum('bihwkl,oikl->bohw', self.windows, self.weights, optimize='greedy')
        self.output += np.expand_dims(self.bias, axis=(0, 2, 3))

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
        # Compute padding for input and output:
        v_padding = self.kernel_size[0] - 1 if self.padding[0] == 0 else self.padding[0]
        h_padding = self.kernel_size[1] - 1 if self.padding[1] == 0 else self.padding[1]

        # Generate windows for upstream gradients:
        out_windows = self._get_windows(upstream_gradients, self.input_shape, self.kernel_size,
                                        padding=(v_padding, h_padding),
                                        dilation=(self.stride[0] - 1, self.stride[1] - 1))

        # Rotate kernel for convolution:
        rot_kern = np.rot90(self.weights, 2, axes=(2, 3))

        # Compute gradients:
        db = np.sum(upstream_gradients, axis=(0, 2, 3))
        dw = np.einsum('bihwkl,bohw->oikl', self.windows, upstream_gradients, optimize='greedy')
        dx = np.einsum('bohwkl,oikl->bihw', out_windows, rot_kern, optimize='greedy')

        # Update parameters and retrograde gradients:
        self.optimizer.update([self.weights, self.bias], [dw, db])
        self.retrograde = dx

    def _initialize_parameters_xavier(self) -> None:
        """
        Initialize the convolutional kernels using the Xavier initialization method.
        """
        a = np.sqrt(6 / (np.prod((self.output_channels, self.input_channels, *self.kernel_size))))
        self.weights = np.random.uniform(-a, a, (self.output_channels, self.input_channels, *self.kernel_size))
        self.bias = np.zeros((self.output_channels,))

    def _initialize_parameters_he(self) -> None:
        """
        Initialize the convolutional kernels using the He initialization method.
        """
        a = np.sqrt(2 / self.input_channels)
        self.weights = np.random.normal(0, a, (self.output_channels, self.input_channels, *self.kernel_size))
        self.bias = np.zeros((self.output_channels,))

    @staticmethod
    def _get_windows(input_data: np.ndarray, output_size: Tuple[int, int], kernel_size: Tuple[int, int],
                     padding: Tuple[int, int] = (0, 0), stride: Tuple[int, int] = (1, 1),
                     dilation: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        Generate convolution windows for the input data.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data.
        output_size : tuple of int
            Size of the output.
        kernel_size : tuple of int
            Size of the convolutional kernels.
        padding : tuple of int, optional
            Padding added to input data. Defaults to (0, 0).
        stride : tuple of int, optional
            Stride for the convolution operation. Defaults to (1, 1).
        dilation : tuple of int, optional
            Dilation rate for the convolution operation. Defaults to (0, 0).

        Returns
        -------
        numpy.ndarray
            Window views for the convolution operation.
        """
        # Dilate and pad the input if necessary:
        if dilation[0] != 0:
            input_data = np.insert(input_data, range(1, input_data.shape[2]), 0, axis=2)
        if dilation[1] != 0:
            input_data = np.insert(input_data, range(1, input_data.shape[3]), 0, axis=3)
        if padding[0] != 0 or padding[1] != 0:
            input_data = np.pad(input_data, pad_width=((0,), (0,), (padding[0],), (padding[1],)),
                                mode='constant', constant_values=(0.,))

        # Get the strides on the input array:
        batch_str, channel_str, kern_h_str, kern_w_str = input_data.strides

        # Create a view into the array with the given shape and strides:
        return np.lib.stride_tricks.as_strided(
            input_data,
            (*input_data.shape[:2], *output_size, *kernel_size),
            (batch_str, channel_str, stride[0] * kern_h_str, stride[1] * kern_w_str, kern_h_str, kern_w_str)
        )
