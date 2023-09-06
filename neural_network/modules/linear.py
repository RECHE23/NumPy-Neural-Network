import numpy as np
from typing import Tuple, Optional, Any, Dict
from . import Module

try:
    import opt_einsum.contract as einsum
except ImportError:
    from numpy import einsum


class Linear(Module):
    """
    Fully Connected (Dense) Layer for a neural network.

    A fully connected layer connects every neuron from the previous layer to every neuron in this layer.
    The layer applies weights and biases to the input data during forward propagation.

    Parameters:
    -----------
    in_features : int
        Number of input features or neurons from the previous layer.
    out_features : int
        Number of neurons in this layer.
    initialization : str, optional
        Initialization method to use ("xavier" or "he"). Default is "xavier".
    *args, **kwargs
        Additional arguments passed to the base class Module.

    Attributes:
    -----------
    in_features : int
        Number of input features or neurons from the previous layer.
    out_features : int
        Number of neurons in this layer.
    initialization : str
        Initialization method used for weight initialization.
    weight : np.ndarray, shape (out_features, in_features)
        Learnable weights for the connections between input and output neurons.
    bias : np.ndarray, shape (out_features,)
        Learnable biases added to each neuron's weighted sum during forward propagation.
    input_shape : tuple of int
        The shape of the input to the layer.
    output_shape : tuple of int
        The shape of the output from the layer.

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Perform forward propagation through the fully connected layer.
    _backward_propagation(upstream_gradients: np.ndarray, y_true: Optional[np.ndarray] = None) -> None:
        Perform backward propagation through the fully connected layer.
    _initialize_parameters(initialization: str) -> None:
        Initialize layer parameters using the specified initialization method.
    """

    def __init__(self, in_features: int, out_features: int, initialization: str = "xavier", *args, **kwargs):
        """
        Initialize the Linear layer with chosen initialization.

        Parameters:
        -----------
        in_features : int
            Number of input features or neurons from the previous layer.
        out_features : int
            Number of neurons in this layer.
        initialization : str, optional
            Initialization method to use ("xavier" or "he"). Default is "xavier".
        *args, **kwargs
            Additional arguments passed to the base class Module.
        """
        super().__init__(*args, **kwargs)

        self.in_features: int
        self.out_features: int
        self.initialization: str
        self.weight: np.ndarray
        self.bias: np.ndarray

        self.state = {
            "in_features": in_features,
            "out_features": out_features,
            "initialization": initialization,
        }

        self._initialize_parameters(initialization)

    def __repr__(self) -> str:
        """
        Return a string representation of the linear layer.
        """
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, optimizer={self.optimizer}, initialization={self.initialization})"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the state of the Linear layer.

        Returns:
        --------
        Tuple[str, Dict[str, Any]]:
            The layer's class name and a dictionary containing the layer's state.
        """
        return self.__class__.__name__, {
            "optimizer_state": self.optimizer.state,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "initialization": self.initialization,
            "weight": self.weight,
            "bias": self.bias
        }

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """
        Set the state of the Linear layer.

        Parameters:
        -----------
        value : Tuple[str, Dict[str, Any]]
            A tuple containing the layer's class name and a dictionary representing the layer's state.
        """
        self.in_features = value["in_features"]
        self.out_features = value["out_features"]
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
        Get the output shape (batch_size, out_features) of the data.

        Returns:
        --------
        Tuple[int, ...]:
            The shape of the output from the layer.
        """
        return self.input.shape[0], self.out_features

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform forward propagation through the fully connected layer.

        Parameters:
        -----------
        input_data : np.ndarray, shape (n_samples, in_features)
            The input data to propagate through the layer.
        """
        assert input_data.shape[1] == self.in_features, "Input size doesn't match"

        self.output = einsum("ij,kj->ik", input_data, self.weight, optimize=True) + self.bias

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: Optional[np.ndarray] = None) -> None:
        """
        Perform backward propagation through the fully connected layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray, shape (n_samples, out_features)
            Gradients received from the subsequent layer during backward propagation.
        y_true : np.ndarray, shape (n_samples, ...)
            The true target values corresponding to the input data.
        """
        assert upstream_gradients.shape[1] == self.out_features, "Upstream gradients size doesn't match"

        self.retrograde = einsum("ij,jk->ik", upstream_gradients, self.weight, optimize=True)
        weights_error = einsum("ji,jk->ki", self.input, upstream_gradients, optimize=True)

        self.optimizer.update([self.weight, self.bias], [weights_error, np.sum(upstream_gradients, axis=0)])

    def _initialize_parameters(self, initialization: str) -> None:
        """
        Initialize layer parameters using the specified initialization method.

        Parameters:
        -----------
        initialization : str
            Initialization method to use ("xavier" or "he").
        """
        if initialization == "xavier":
            a = np.sqrt(6 / (self.in_features + self.out_features))
            self.weight = np.random.uniform(-a, a, (self.out_features, self.in_features))
        elif initialization == "he":
            a = np.sqrt(2 / self.in_features)
            self.weight = np.random.normal(0, a, (self.out_features, self.in_features))
        else:
            raise ValueError("Invalid initialization method. Use 'xavier' or 'he'.")
        self.bias = np.zeros((self.out_features,))
