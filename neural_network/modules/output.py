from typing import Union, Callable, Tuple, Dict, Any
import numpy as np
from . import Module
from neural_network.functions.activation import activation_functions
from neural_network.functions.loss import loss_functions
from neural_network.functions.output import output_functions


class OutputLayer(Module):
    """
    An output layer for neural network architectures.

    Parameters:
    -----------
    activation_function_name: str
        The name of the activation function. Default is "relu".
    loss_function_name: str
        The name of the loss function. Default is "categorical_cross_entropy".
    activation_function : callable
        Activation function applied to the layer's output.
    loss_function : callable
        Loss function to calculate the loss between predicted and true values.
    *args, **kwargs:
        Additional arguments to pass to the base class.

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Compute the output of the output layer using the given input data.
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Compute the retrograde gradients for the output layer.
    loss(y_true: np.ndarray, y_pred: np.ndarray, prime: bool = False) -> float:
        Compute the loss between true and predicted values.

    Attributes:
    -----------
    activation_function : callable
        Activation function applied to the layer's output.
    loss_function : callable
        Loss function to calculate the loss between predicted and true values.
    input_shape : tuple of int
        The shape of the input to the layer.
    output_shape : tuple of int
        The shape of the output from the layer.
    """
    activation_function_name: str
    loss_function_name: str
    activation_function: Callable
    loss_function: Callable

    def __init__(self, activation_function: str = "tanh", loss_function: str = "mean_squared_error", *args, **kwargs):
        """
        Initialize the OutputLayer with the given activation and loss functions.

        Parameters:
        -----------
        activation_function : str
            Activation function applied to the layer's output. Default is "relu".
        loss_function : str
            Loss function to calculate the loss between predicted and true values. Default is "categorical_cross_entropy".
        *args, **kwargs:
            Additional arguments to pass to the base class.
        """
        super().__init__(*args, **kwargs)

        self.state = {
            "activation_function_name": activation_function,
            "loss_function_name": loss_function,
        }

    def __repr__(self) -> str:
        """
        Return a string representation of the output layer.
        """
        return f"{self.__class__.__name__}(activation_function={self.activation_function_name}, loss_function={self.loss_function_name})"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Return the state of the output layer.

        Returns:
        ----------
        state : tuple
            A tuple containing the class name and a dictionary of attributes.
        """
        return self.__class__.__name__, {
            "activation_function_name": self.activation_function_name,
            "loss_function_name": self.loss_function_name,
        }

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """
        Set the state of the output layer.

        Parameters:
        -----------
        value : dict
            A dictionary containing the activation and loss function names.

        Raises:
        -------
        AssertionError
            If the activation or loss function name is invalid.
        """
        activation_function_name = value["activation_function_name"].lower().strip()
        loss_function_name = value["loss_function_name"].lower().strip()
        assert activation_function_name in activation_functions, f"Invalid activation function. Choose from {list(activation_functions.keys())}"
        assert loss_function_name in loss_functions, f"Invalid loss function. Choose from {list(loss_functions.keys())}"

        self.activation_function_name: str = activation_function_name
        self.loss_function_name: str = loss_function_name
        self.activation_function: Callable = activation_functions[activation_function_name]
        self.loss_function: Callable = loss_functions[loss_function_name]

    @property
    def parameters_count(self) -> int:
        """
        Get the total number of parameters in the module.
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
        Compute the output of the output layer using the given input data.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data for the output layer.
        """
        self.output = self.activation_function(self.input)

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Compute the retrograde gradients for the output layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Upstream gradients coming from the subsequent layer.
        y_true : np.ndarray
            The true labels used for calculating the retrograde gradient.
        """
        self.retrograde = self.loss_function(y_true, self.output, prime=True)

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray, *args, **kwargs) -> Union[float, np.ndarray]:
        """
        Compute the loss between true and predicted values.

        Parameters:
        -----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted values.
        *args, **kwargs
            Additional arguments passed to the loss function.

        Returns:
        --------
        loss : float
            The computed loss value.
        """
        return self.loss_function(y_true, y_pred, *args, **kwargs)


class SoftmaxBinaryCrossEntropy(OutputLayer):
    """
    An output layer for neural network architectures with softmax activation and binary cross-entropy loss.

    Methods:
    --------
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Compute the retrograde gradients for the output layer.
    """

    def __repr__(self) -> str:
        """
        Return a string representation of the SoftmaxBinaryCrossEntropy layer.
        """
        return f"{self.__class__.__name__}()"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Return the state of the output layer.

        Returns:
        ----------
        state : tuple
            A tuple containing the class name and a dictionary of attributes.
        """
        return super().state

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """
        Set the state of the output layer.

        It bypasses the default state of the base class OutputLayer to avoid an AssertionError since softmax isn't part of activation_functions.

        Parameters:
        -----------
        value : dict
            A dictionary containing the activation and loss function names.
        """
        OutputLayer.state.fset(self, value)
        self.activation_function_name = "softmax"
        self.loss_function_name = "binary_cross_entropy"
        self.activation_function = output_functions[self.activation_function_name]
        self.loss_function = loss_functions[self.loss_function_name]

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Compute the retrograde gradients for the output layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Upstream gradients coming from the subsequent layer.
        y_true : np.ndarray
            The true labels used for calculating the retrograde gradient.
        """
        y_pred_clipped = np.clip(self.output, 1e-10, 1 - 1e-10)
        self.retrograde = y_pred_clipped - y_true


class SoftmaxCategoricalCrossEntropy(OutputLayer):
    """
    An output layer for neural network architectures with softmax activation and categorical cross-entropy loss.

    Methods:
    --------
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Compute the retrograde gradients for the output layer.
    """
    def __repr__(self) -> str:
        """
        Return a string representation of the SoftmaxCategoricalCrossEntropy layer.
        """
        return f"{self.__class__.__name__}()"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Return the state of the output layer.

        Returns:
        ----------
        state : tuple
            A tuple containing the class name and a dictionary of attributes.
        """
        return super().state

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """
        Set the state of the output layer.

        It bypasses the default state of the base class OutputLayer to avoid an AssertionError since softmax isn't part of activation_functions.

        Parameters:
        -----------
        value : dict
            A dictionary containing the activation and loss function names.
        """
        OutputLayer.state.fset(self, value)
        self.activation_function_name = "softmax"
        self.loss_function_name = "categorical_cross_entropy"
        self.activation_function = output_functions[self.activation_function_name]
        self.loss_function = loss_functions[self.loss_function_name]

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Compute the retrograde gradients for the output layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Upstream gradients coming from the subsequent layer.
        y_true : np.ndarray
            The true labels used for calculating the retrograde gradient.
        """
        y_pred_clipped = np.clip(self.output, 1e-10, 1 - 1e-10)
        self.retrograde = y_pred_clipped - y_true


class SoftminBinaryCrossEntropy(OutputLayer):
    """
    An output layer for neural network architectures with softmin activation and binary cross-entropy loss.

    Methods:
    --------
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Compute the retrograde gradients for the output layer.
    """

    def __repr__(self) -> str:
        """
        Return a string representation of the SoftminBinaryCrossEntropy layer.
        """
        return f"{self.__class__.__name__}()"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Return the state of the output layer.

        Returns:
        ----------
        state : tuple
            A tuple containing the class name and a dictionary of attributes.
        """
        return super().state

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """
        Set the state of the output layer.

        It bypasses the default state of the base class OutputLayer to avoid an AssertionError since softmin isn't part of activation_functions.

        Parameters:
        -----------
        value : dict
            A dictionary containing the activation and loss function names.
        """
        OutputLayer.state.fset(self, value)
        self.activation_function_name = "softmin"
        self.loss_function_name = "binary_cross_entropy"
        self.activation_function = output_functions[self.activation_function_name]
        self.loss_function = loss_functions[self.loss_function_name]

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Compute the retrograde gradients for the output layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Upstream gradients coming from the subsequent layer.
        y_true : np.ndarray
            The true labels used for calculating the retrograde gradient.
        """
        y_pred_clipped = np.clip(self.output, 1e-10, 1 - 1e-10)
        self.retrograde = y_pred_clipped - y_true


class SoftminCategoricalCrossEntropy(OutputLayer):
    """
    An output layer for neural network architectures with softmin activation and categorical cross-entropy loss.

    Methods:
    --------
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Compute the retrograde gradients for the output layer.
    """
    def __repr__(self) -> str:
        """
        Return a string representation of the SoftminCategoricalCrossEntropy layer.
        """
        return f"{self.__class__.__name__}()"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Return the state of the output layer.

        Returns:
        ----------
        state : tuple
            A tuple containing the class name and a dictionary of attributes.
        """
        return super().state

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """
        Set the state of the output layer.

        It bypasses the default state of the base class OutputLayer to avoid an AssertionError since softmin isn't part of activation_functions.

        Parameters:
        -----------
        value : dict
            A dictionary containing the activation and loss function names.
        """
        OutputLayer.state.fset(self, value)
        self.activation_function_name = "softmin"
        self.loss_function_name = "categorical_cross_entropy"
        self.activation_function = output_functions[self.activation_function_name]
        self.loss_function = loss_functions[self.loss_function_name]

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Compute the retrograde gradients for the output layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Upstream gradients coming from the subsequent layer.
        y_true : np.ndarray
            The true labels used for calculating the retrograde gradient.
        """
        y_pred_clipped = np.clip(self.output, 1e-10, 1 - 1e-10)
        self.retrograde = y_pred_clipped - y_true