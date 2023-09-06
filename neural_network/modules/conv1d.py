import numpy as np
from typing import Tuple, Optional, Dict, Any
from . import Module

try:
    import opt_einsum.contract as einsum
except ImportError:
    from numpy import einsum


class Conv1d(Module):
    """
    A 1D convolutional layer for neural network architectures.

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (number of kernels).
    kernel_size : int
        Size of the convolutional kernels.
    stride : int, optional
        Stride for the convolution operation. Defaults to 1.
    padding : int, optional
        Padding added to input data. Defaults to 0.
    initialization : str, optional
        Weight initialization method: "xavier" or "he" (default is "xavier").
    *args, **kwargs:
        Additional arguments to pass to the base class.

    Attributes:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernels.
    stride : int
        Stride for the convolution operation.
    padding : int
        Padding added to input data.
    initialization : str
        Weight initialization method.
    weight : np.ndarray
        Convolutional kernels.
    bias : np.ndarray
        Biases for each output channel.

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Compute the output of the convolutional layer using the given input data.
    _backward_propagation(upstream_gradients: np.ndarray, y_true: Optional[np.ndarray] = None) -> None:
        Compute the retrograde gradients for the convolutional layer.
    _initialize_parameters(initialization: str) -> None:
        Initialize convolutional kernels using the specified initialization method.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, initialization: str = "xavier", *args, **kwargs):
        """
        Initialize a Conv1d layer.

        Parameters:
        -----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels (number of kernels).
        kernel_size : int
            Size of the convolutional kernels.
        stride : int, optional
            Stride for the convolution operation. Defaults to 1.
        padding : int, optional
            Padding added to input data. Defaults to 0.
        initialization : str, optional
            Weight initialization method: "xavier" or "he" (default is "xavier").
        *args, **kwargs:
            Additional arguments to pass to the base class.
        """
        super().__init__(*args, **kwargs)

        self.in_channels: int
        self.out_channels: int
        self.kernel_size: int
        self.stride: int
        self.padding: int
        self.initialization: str
        self.weight: np.ndarray
        self.bias: np.ndarray

        self.state = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "initialization": initialization
        }

        self._initialize_parameters(self.initialization)

    def __repr__(self) -> str:
        """
        Return a string representation of the Conv1d layer.
        """
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels="
            f"{self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding="
            f"{self.padding}, initialization={self.initialization})"
        )

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the state of the Conv1d layer.

        Returns:
        --------
        Tuple[str, Dict[str, Any]]:
            The layer's class name and a dictionary containing the layer's state.
        """
        return self.__class__.__name__, {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "initialization": self.initialization,
            "weight": self.weight,
            "bias": self.bias
        }

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """
        Set the state of the Conv1d layer.

        Parameters:
        -----------
        value : Tuple[str, Dict[str, Any]]
            The layer's class name and a dictionary containing the layer's state.
        """
        assert value["in_channels"] > 0, "Number of input channels must be greater than 0"
        assert value["out_channels"] > 0, "Number of output channels must be greater than 0"

        self.in_channels = value["in_channels"]
        self.out_channels = value["out_channels"]
        self.kernel_size = value["kernel_size"]
        self.stride = value["stride"]
        self.padding = value["padding"]
        self.initialization = value["initialization"]
        self.weight = value.get("weight", None)
        self.bias = value.get("bias", None)

    @property
    def parameters_count(self) -> int:
        """
        Get the total number of parameters in the module.
        """
        return np.prod(self.weight.shape) + np.prod(self.bias.shape)

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        """
        Get the output shape (batch_size, out_channels, output_length) of the layer's data.

        Returns:
        --------
        Tuple[int, int, int]:
            Output shape of the layer's data.
        """
        return self.input.shape[0], self.out_channels, self.output_length

    @property
    def input_length(self) -> int:
        """
        Get the length of the input data.
        """
        return self.input.shape[2]

    @property
    def output_length(self) -> int:
        """
        Calculate and get the output length after convolution.
        """
        return (self.input_length + 2 * self.padding - self.kernel_size) // self.stride + 1

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Compute the output of the convolutional layer using the given input data.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data for the convolutional layer.
        """
        assert len(input_data.shape) == 3, "Input data must have shape (batch, channels, length)"

        # Generate windows:
        self.windows = self._get_windows(input_data, self.output_length, self.kernel_size, padding=self.padding, stride=self.stride)

        # Perform convolution and add bias:
        self.output = einsum('bilk,oik->bol', self.windows, self.weight, optimize=True)
        self.output += np.expand_dims(self.bias, axis=(0, 2))

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: Optional[np.ndarray] = None) -> None:
        """
        Compute the retrograde gradients for the convolutional layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Upstream gradients coming from the subsequent layer.
        y_true : np.ndarray
            The true labels used for calculating the retrograde gradient.
        """
        assert len(upstream_gradients.shape) == 3, "Upstream gradients must have shape (batch, channels, length)"
        assert upstream_gradients.shape[1] == self.out_channels, "Upstream gradients channels don't match"

        # Compute padding for input and output:
        padding = self.kernel_size - 1 if self.padding == 0 else self.padding

        # Generate windows for upstream gradients:
        out_windows = self._get_windows(upstream_gradients, self.input_length, self.kernel_size, padding=padding, dilation=self.stride - 1)

        # Compute gradients:
        db = np.sum(upstream_gradients, axis=(0, 2))
        dw = einsum('bilk,bol->oik', self.windows, upstream_gradients, optimize=True)
        dx = einsum('bolk,oik->bil', out_windows, np.flip(self.weight, axis=2), optimize=True)

        # Update parameters and retrograde gradients:
        self.optimizer.update([self.weight, self.bias], [dw, db])
        self.retrograde = dx

    def _initialize_parameters(self, initialization: str) -> None:
        """
        Initialize convolutional kernels using the specified initialization method.

        Parameters:
        -----------
        initialization : str
            Initialization method to use ("xavier" or "he").
        """
        if initialization == "xavier":
            a = np.sqrt(6 / (self.out_channels * self.in_channels * self.kernel_size))
            self.weight = np.random.uniform(-a, a, (self.out_channels, self.in_channels, self.kernel_size))
        elif initialization == "he":
            a = np.sqrt(2 / (self.in_channels * self.kernel_size))
            self.weight = np.random.normal(0, a, (self.out_channels, self.in_channels, self.kernel_size))
        else:
            raise ValueError("Invalid initialization method. Use 'xavier' or 'he'.")

        self.bias = np.zeros((self.out_channels,))

    @staticmethod
    def _get_windows(input_data: np.ndarray, output_size: int, kernel_size: int, padding: int = 0, stride: int = 1, dilation: int = 0) -> np.ndarray:
        assert len(input_data.shape) == 3, "Input data must have shape (batch, channels, length)"

        # Dilate and pad the input if necessary:
        if dilation != 0:
            input_data = np.insert(input_data, range(1, input_data.shape[2]), 0, axis=2)
        if padding != 0:
            input_data = np.pad(input_data, pad_width=((0,), (0,), (padding,)), mode='constant', constant_values=0)

        # Get the strides on the input array:
        batch_str, channel_str, kern_l_str = input_data.strides

        # Create a view into the array with the given shape and strides:
        return np.lib.stride_tricks.as_strided(
            input_data,
            (*input_data.shape[:2], output_size, kernel_size),
            (batch_str, channel_str, stride * kern_l_str, kern_l_str)
        )
