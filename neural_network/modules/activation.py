from typing import Callable, Tuple, Dict, Any
from functools import partial
import numpy as np
from . import Layer
from neural_network.functions.activation import activation_functions


class ActivationLayer(Layer):
    """
    A layer implementing activation functions for neural network architectures.

    Parameters:
    -----------
    activation_function : str
        The activation function to be applied during forward and backward propagation.

    Attributes:
    -----------
    activation_function_name: str
        The name of the activation function. Default is "relu".
    activation_function : callable
        The activation function.
    output : np.ndarray
        The output of the activation layer after forward propagation.
    retrograde : np.ndarray
        The gradient used for backward propagation.
    input_shape : tuple of int
        The shape of the input to the layer.
    output_shape : tuple of int
        The shape of the output from the layer.

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Compute the output of the activation layer using the given input data.
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Compute the retrograde gradient for the activation layer.
    """

    def __init__(self, activation_function: str = "relu", *args, **kwargs):
        """
        Initialize the ActivationLayer with the given activation function.

        Parameters:
        -----------
        activation_function : callable
            The activation function to be applied during forward and backward propagation. Default is "relu".
        """
        super().__init__(*args, **kwargs)

        self.activation_function_name: str
        self.activation_function: Callable

        self.state = {
            "activation_function_name": activation_function
        }

    def __repr__(self) -> str:
        """
        Return a string representation of the activation layer.
        """
        activation_function = f"activation_function={self.activation_function_name}" if self.__class__.__name__ == "ActivationLayer" else ""
        return f"{self.__class__.__name__}({activation_function})"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the current state of the activation layer.
        """
        return self.__class__.__name__, {
            "activation_function_name": self.activation_function_name
        }

    @state.setter
    def state(self, value) -> None:
        """
        Set the state of the activation layer.
        """
        activation_function_name = value["activation_function_name"].lower().strip()
        assert activation_function_name in activation_functions, f"Invalid activation function. Choose from {list(activation_functions.keys())}"

        self.activation_function_name: str = activation_function_name
        self.activation_function: Callable = activation_functions[activation_function_name]

    @property
    def parameters_count(self) -> int:
        """
        Get the total number of parameters in the layer.
        """
        return 0

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape (batch_size, ...) of the layer's data.

        Returns:
        --------
        Tuple[int, ...]
            Output shape of the layer's data.
        """
        return self.input.shape

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Compute the output of the activation layer using the given input data.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data for the activation layer.
        """
        self.output = self.activation_function(input_data)

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Compute the retrograde gradient for the activation layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            The gradient coming from the subsequent layer.
        y_true : np.ndarray
            The true labels used for calculating the retrograde gradient.
        """
        self.retrograde = self.activation_function(self.input, prime=True) * upstream_gradients


class ReLU(ActivationLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(activation_function="relu", *args, **kwargs)


class Tanh(ActivationLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(activation_function="tanh", *args, **kwargs)


class Sigmoid(ActivationLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(activation_function="sigmoid", *args, **kwargs)


class LeakyReLU(ActivationLayer):
    def __init__(self, alpha=0.01, *args, **kwargs):
        super().__init__(activation_function="leaky_relu", *args, alpha=alpha, **kwargs)


class ELU(ActivationLayer):
    def __init__(self, alpha=1.0, *args, **kwargs):
        super().__init__(activation_function="elu", *args, alpha=alpha, **kwargs)


class Swish(ActivationLayer):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(activation_function="swish", *args, beta=beta, **kwargs)


class ArcTan(ActivationLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(activation_function="arctan", *args, **kwargs)


class Gaussian(ActivationLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(activation_function="gaussian", *args, **kwargs)


class SiLU(ActivationLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(activation_function="silu", *args, **kwargs)


class BentIdentity(ActivationLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(activation_function="bent_identity", *args, **kwargs)


class SELU(ActivationLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(activation_function="selu", *args, **kwargs)


class CELU(ActivationLayer):
    def __init__(self, alpha=1.0, *args, **kwargs):
        super().__init__(activation_function="celu", *args, alpha=alpha, **kwargs)


class GELU(ActivationLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(activation_function="gelu", *args, **kwargs)


class Softplus(ActivationLayer):
    def __init__(self, beta=1, threshold=20, *args, **kwargs):
        super().__init__(activation_function="softplus", *args, beta=beta, threshold=threshold, **kwargs)


class Mish(ActivationLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(activation_function="mish", *args, **kwargs)
