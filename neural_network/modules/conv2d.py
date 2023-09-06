import numpy as np
from typing import Union, Tuple, Optional, Dict, Any
from . import Module
from neural_network.functions import pair

try:
    import opt_einsum.contract as einsum
except ImportError:
    from numpy import einsum


class Conv2d(Module):
    """
    A 2D convolutional layer for neural network architectures.

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
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
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : tuple of int
        Size of the convolutional kernels.
    stride : tuple of int
        Stride (vertical stride, horizontal stride).
    padding : tuple of int
        Padding (vertical padding, horizontal padding).
    initialization : str
        Weight initialization method.
    weight : np.ndarray
        Convolutional kernels.
    bias : np.ndarray
        Biases for each output channel.
    windows : np.ndarray
        Memorized convolution windows from the forward propagation.

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Compute the output of the convolutional layer using the given input data.
    _backward_propagation(upstream_gradients: np.ndarray, y_true: Optional[np.ndarray] = None) -> None:
        Compute the retrograde gradients for the convolutional layer.
    _initialize_parameters(initialization: str) -> None:
        Initialize convolutional kernels using the specified initialization method.
    _get_windows(input_data: np.ndarray, output_size: Tuple[int, int], kernel_size: Tuple[int, int],
                 padding: Tuple[int, int] = (0, 0), stride: Tuple[int, int] = (1, 1),
                 dilation: Tuple[int, int] = (0, 0)) -> np.ndarray:
        Generate convolution windows for the input data.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                 initialization: str = "xavier", *args, **kwargs):
        """
        Initialize a Conv2d layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
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

        self.in_channels: int
        self.out_channels: int
        self.kernel_size: Tuple[int, int]
        self.stride: Tuple[int, int]
        self.padding: Tuple[int, int]
        self.initialization: str
        self.weight: np.ndarray
        self.bias: np.ndarray
        self.windows: Optional[np.ndarray] = None

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
        Return a string representation of the Conv2d layer.
        """
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels="
            f"{self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding="
            f"{self.padding}, optimizer={self.optimizer}, initialization={self.initialization})"
        )

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the state of the Conv2d layer.

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
        Set the state of the Conv2d layer.

        Parameters:
        -----------
        value : Tuple[str, Dict[str, Any]]
            The layer's class name and a dictionary containing the layer's state.
        """
        assert value["in_channels"] > 0, "Number of input channels must be greater than 0"
        assert value["out_channels"] > 0, "Number of output channels must be greater than 0"

        self.in_channels = value["in_channels"]
        self.out_channels = value["out_channels"]
        self.kernel_size = pair(value["kernel_size"])
        self.stride = pair(value["stride"])
        self.padding = pair(value["padding"])
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
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape (batch_size, out_channels, output_height, output_width)  of the layer's data.

        Returns:
        --------
        Tuple[int, ...]
            Output shape of the layer's data.
        """
        return self.input.shape[0], self.out_channels, self.output_dimensions[0], self.output_dimensions[1]

    @property
    def input_dimensions(self) -> Tuple[int, int]:
        """
        Get the input shape (height, width) of the data.
        """
        return self.input.shape[2], self.input.shape[3]

    @property
    def output_dimensions(self) -> Tuple[int, int]:
        """
        Calculate and get the output shape (height, width) after convolution and pooling.
        """
        output_height = (self.input_dimensions[0] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        output_width = (self.input_dimensions[1] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        return output_height, output_width

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Compute the output of the convolutional layer using the given input data.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data for the convolutional layer.
        """
        assert len(input_data.shape) == 4, "Input data must have shape (batch, channels, height, width)"

        # Generate windows:
        self.windows = self._get_windows(input_data, self.output_dimensions, self.kernel_size,
                                         padding=self.padding, stride=self.stride)

        # Perform convolution and add bias:
        self.output = einsum('bihwkl,oikl->bohw', self.windows, self.weight, optimize=True)
        self.output += np.expand_dims(self.bias, axis=(0, 2, 3))

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
        assert len(upstream_gradients.shape) == 4, "Upstream gradients must have shape (batch, channels, height, width)"
        assert upstream_gradients.shape[1] == self.out_channels, "Upstream gradients channels don't match"

        # Compute padding for input and output:
        v_padding = self.kernel_size[0] - 1 if self.padding[0] == 0 else self.padding[0]
        h_padding = self.kernel_size[1] - 1 if self.padding[1] == 0 else self.padding[1]

        # Generate windows for upstream gradients:
        out_windows = self._get_windows(upstream_gradients, self.input_dimensions, self.kernel_size,
                                        padding=(v_padding, h_padding),
                                        dilation=(self.stride[0] - 1, self.stride[1] - 1))

        # Rotate kernel for convolution:
        rot_kern = np.rot90(self.weight, 2, axes=(2, 3))

        # Compute gradients:
        db = np.sum(upstream_gradients, axis=(0, 2, 3))
        dw = einsum('bihwkl,bohw->oikl', self.windows, upstream_gradients, optimize=True)
        dx = einsum('bohwkl,oikl->bihw', out_windows, rot_kern, optimize=True)

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
            a = np.sqrt(6 / (np.prod((self.out_channels, self.in_channels, *self.kernel_size))))
            self.weight = np.random.uniform(-a, a, (self.out_channels, self.in_channels, *self.kernel_size))
        elif initialization == "he":
            a = np.sqrt(2 / self.in_channels)
            self.weight = np.random.normal(0, a, (self.out_channels, self.in_channels, *self.kernel_size))
        else:
            raise ValueError("Invalid initialization method. Use 'xavier' or 'he'.")

        self.bias = np.zeros((self.out_channels,))

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
        assert len(input_data.shape) == 4, "Input data must have shape (batch, channels, height, width)"

        # Dilate and pad the input if necessary:
        if dilation[0] != 0:
            input_data = np.insert(input_data, range(1, input_data.shape[2]), 0, axis=2)
        if dilation[1] != 0:
            input_data = np.insert(input_data, range(1, input_data.shape[3]), 0, axis=3)
        if padding[0] != 0 or padding[1] != 0:
            input_data = np.pad(input_data, pad_width=((0,), (0,), (padding[0],), (padding[1],)),
                                mode='constant', constant_values=0)

        # Get the strides on the input array:
        batch_str, channel_str, kern_h_str, kern_w_str = input_data.strides

        # Create a view into the array with the given shape and strides:
        return np.lib.stride_tricks.as_strided(
            input_data,
            (*input_data.shape[:2], *output_size, *kernel_size),
            (batch_str, channel_str, stride[0] * kern_h_str, stride[1] * kern_w_str, kern_h_str, kern_w_str)
        )
