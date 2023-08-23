import numpy as np
from . import Layer


class FullyConnectedLayer(Layer):
    """
    Fully Connected (Dense) Layer for a neural network.

    A fully connected layer connects every neuron from the previous layer to every neuron in this layer.
    The layer applies weights and biases to the input data during forward propagation.

    Parameters:
    -----------
    input_size : int
        Number of input features or neurons from the previous layer.
    output_size : int
        Number of neurons in this layer.
    *args, **kwargs
        Additional arguments passed to the base class Layer.

    Attributes:
    -----------
    input_size : int
            Number of input features or neurons from the previous layer.
    output_size : int
        Number of neurons in this layer.
    weights : array-like, shape (input_size, output_size)
        Learnable weights for the connections between input and output neurons.
    bias : array-like, shape (output_size,)
        Learnable biases added to each neuron's weighted sum during forward propagation.

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
    def __init__(self, input_size: int, output_size: int, initialization: str = "xavier", *args, **kwargs):
        """
        Initialize the FullyConnectedLayer with chosen initialization.

        Parameters:
        -----------
        input_size : int
            Number of input features or neurons from the previous layer.
        output_size : int
            Number of neurons in this layer.
        initialization : str, optional
            Initialization method to use ("xavier" or "he"). Default is "xavier".
        *args, **kwargs
            Additional arguments passed to the base class Layer.

        """
        super().__init__(*args, **kwargs)

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.initialization: str = initialization

        if initialization == "xavier":
            self._initialize_parameters_xavier()
        elif initialization == "he":
            self._initialize_parameters_he()
        else:
            raise ValueError("Invalid initialization method. Use 'xavier' or 'he'.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_size={self.input_size}, output_size={self.output_size}, optimizer={self.optimizer}, initialization={self.initialization})"

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform forward propagation through the fully connected layer.

        Parameters:
        -----------
        input_data : array-like, shape (n_samples, input_size)
            The input data to propagate through the layer.
        """
        self.output = np.einsum("ij,jk", self.input, self.weights, optimize='greedy') + self.bias

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Perform backward propagation through the fully connected layer.

        Parameters:
        -----------
        upstream_gradients : array-like, shape (n_samples, output_size)
            Gradients received from the subsequent layer during backward propagation.
        y_true : array-like, shape (n_samples, ...)
            The true target values corresponding to the input data.
        """
        self.retrograde = np.einsum("ij,kj", upstream_gradients, self.weights, optimize='greedy')
        weights_error = np.einsum("ji,jk", self.input, upstream_gradients, optimize='greedy')

        self.optimizer.update([self.weights, self.bias], [weights_error, np.sum(upstream_gradients, axis=0)])

    def _initialize_parameters_xavier(self) -> None:
        """
        Initialize layer parameters using Xavier initialization.
        """
        a = np.sqrt(6 / (self.input_size + self.output_size))
        self.weights = np.random.uniform(-a, a, (self.input_size, self.output_size))
        self.bias = np.zeros((self.output_size,))

    def _initialize_parameters_he(self) -> None:
        """
        Initialize layer parameters using He initialization.
        """
        a = np.sqrt(2 / self.input_size)
        self.weights = np.random.normal(0, a, (self.input_size, self.output_size))
        self.bias = np.zeros((self.output_size,))
