from typing import Tuple, Optional
import numpy as np
from . import Layer


class Linear(Layer):
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
    *args, **kwargs
        Additional arguments passed to the base class Layer.

    Attributes:
    -----------
    in_features : int
            Number of input features or neurons from the previous layer.
    out_features : int
        Number of neurons in this layer.
    weight : array-like, shape (out_features, in_features)
        Learnable weights for the connections between input and output neurons.
    bias : array-like, shape (out_features,)
        Learnable biases added to each neuron's weighted sum during forward propagation.
    input_shape : tuple of int
        The shape of the input to the layer.
    output_shape : tuple of int
        The shape of the output from the layer.

    Methods:
    --------
    _forward_propagation(input_data)
        Perform forward propagation through the fully connected layer.
    _backward_propagation(upstream_gradients, y_true)
        Perform backward propagation through the fully connected layer.
    _initialize_parameters_xavier()
        Initialize layer parameters using Xavier initialization.
    _initialize_parameters_he()
        Initialize layer parameters using He initialization.

    """
    def __init__(self, in_features: int, out_features: int, initialization: str = "xavier", *args, **kwargs):
        """
        Initialize the Linear with chosen initialization.

        Parameters:
        -----------
        in_features : int
            Number of input features or neurons from the previous layer.
        out_features : int
            Number of neurons in this layer.
        initialization : str, optional
            Initialization method to use ("xavier" or "he"). Default is "xavier".
        *args, **kwargs
            Additional arguments passed to the base class Layer.

        """
        super().__init__(*args, **kwargs)

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.initialization: str = initialization

        self._initialize_parameters(initialization)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, optimizer={self.optimizer}, initialization={self.initialization})"

    @property
    def parameters_count(self) -> int:
        return np.prod(self.weight.shape) + np.prod(self.bias.shape)

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape (batch_size, out_features) of the data.
        """
        return self.input.shape[0], self.out_features

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform forward propagation through the fully connected layer.

        Parameters:
        -----------
        input_data : array-like, shape (n_samples, in_features)
            The input data to propagate through the layer.
        """
        assert input_data.shape[1] == self.in_features, "Input size doesn't match"

        self.output = np.einsum("ij,kj->ik", self.input, self.weight, optimize='greedy') + self.bias

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: Optional[np.ndarray] = None) -> None:
        """
        Perform backward propagation through the fully connected layer.

        Parameters:
        -----------
        upstream_gradients : array-like, shape (n_samples, out_features)
            Gradients received from the subsequent layer during backward propagation.
        y_true : array-like, shape (n_samples, ...)
            The true target values corresponding to the input data.
        """
        assert upstream_gradients.shape[1] == self.out_features, "Upstream gradients size doesn't match"

        self.retrograde = np.einsum("ij,jk->ik", upstream_gradients, self.weight, optimize='greedy')
        weights_error = np.einsum("ji,jk->ki", self.input, upstream_gradients, optimize='greedy')

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
