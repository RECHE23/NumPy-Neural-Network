from typing import Any, Dict, Tuple
import numpy as np
from . import Module


class Identity(Module):
    """
    Identity layer that passes input data through without modification.

    This layer is useful when you want to add skip connections or simply pass data from one part of
    the network to another without any transformation.

    Parameters:
    -----------
    *args, **kwargs
        Additional arguments passed to the base class Module.

    Attributes:
    -----------
    input_shape : tuple of int
        The shape of the input to the layer.
    output_shape : tuple of int
        The shape of the output from the layer.

    Methods:
    --------
    _forward_propagation(input_data: np.ndarray) -> None:
        Perform forward propagation through the identity layer.
    _backward_propagation(upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        Perform backward propagation through the identity layer.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Identity layer.

        Parameters:
        -----------
        *args, **kwargs
            Additional arguments passed to the base class Module.
        """
        super().__init__(*args, **kwargs)

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the state of the identity layer.

        Returns:
        --------
        Tuple[str, Dict[str, Any]]:
            The layer's class name and an empty dictionary representing the layer's state.
        """
        return self.__class__.__name__, {}

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """
        Set the state of the identity layer.

        Parameters:
        -----------
        value : Tuple[str, Dict[str, Any]]
            A tuple containing the layer's class name and a dictionary representing the layer's state.
        """
        pass

    @property
    def parameters_count(self) -> int:
        """
        Get the total number of parameters in the module.
        """
        return 0

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape of the data.

        Returns:
        --------
        Tuple[int, ...]:
            The shape of the output from the identity layer, which is the same as the input shape.
        """
        return self.input_shape

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform forward propagation through the identity layer.

        Parameters:
        -----------
        input_data : np.ndarray
            The input data to propagate through the layer.
        """
        self.output = input_data

    def _backward_propagation(self, upstream_gradients: np.ndarray, y_true: np.ndarray) -> None:
        """
        Perform backward propagation through the identity layer.

        Parameters:
        -----------
        upstream_gradients : np.ndarray
            Gradients received from the subsequent layer during backward propagation.
        y_true : np.ndarray
            The true target values corresponding to the input data.
        """
        self.retrograde = upstream_gradients
