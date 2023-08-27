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

        self.input_size: int = in_features
        self.output_size: int = out_features
        self.initialization: str = initialization

        if initialization == "xavier":
            self._initialize_parameters_xavier()
        elif initialization == "he":
            self._initialize_parameters_he()
        else:
            raise ValueError("Invalid initialization method. Use 'xavier' or 'he'.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_features={self.input_size}, out_features={self.output_size}, optimizer={self.optimizer}, initialization={self.initialization})"

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape (batch_size, out_features) of the data.
        """
        return self.input.shape[0], self.output_size

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform forward propagation through the fully connected layer.

        Parameters:
        -----------
        input_data : array-like, shape (n_samples, in_features)
            The input data to propagate through the layer.
        """
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
        self.retrograde = np.einsum("ij,jk->ik", upstream_gradients, self.weight, optimize='greedy')
        weights_error = np.einsum("ji,jk->ki", self.input, upstream_gradients, optimize='greedy')

        self.optimizer.update([self.weight, self.bias], [weights_error, np.sum(upstream_gradients, axis=0)])

    def _initialize_parameters_xavier(self) -> None:
        """
        Initialize layer parameters using Xavier initialization.
        """
        a = np.sqrt(6 / (self.input_size + self.output_size))
        self.weight = np.random.uniform(-a, a, (self.output_size, self.input_size))
        self.bias = np.zeros((self.output_size,))

    def _initialize_parameters_he(self) -> None:
        """
        Initialize layer parameters using He initialization.
        """
        a = np.sqrt(2 / self.input_size)
        self.weight = np.random.normal(0, a, (self.output_size, self.input_size))
        self.bias = np.zeros((self.output_size,))
