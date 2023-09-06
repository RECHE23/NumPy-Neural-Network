import numpy as np
from typing import Any, Dict, Optional, Tuple
from neural_network.modules import Module


class SkipConnection(Module):
    """
    SkipConnection Module

    This module represents a Skip Connection in a neural network. It combines the output of an inner module with its
    input, effectively allowing the network to bypass one or more layers.

    Parameters
    ----------
    inner_module : Module
        The inner module whose output is combined with the input.

    Attributes
    ----------
    inner_module : Module
        The inner module contained within the SkipConnection.
    output_shape : Tuple[int, ...]
        The shape of the output produced by this module.
    parameters_count : int
        The total number of parameters in this module and its inner module.

    Methods
    -------
    forward(input_data: np.ndarray) -> None
        Perform forward propagation through the SkipConnection.
    backward(upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> None
        Perform backward propagation through the SkipConnection.

    Properties
    ----------
    state : Tuple[str, Dict[str, Any]]
        Get or set the state of the SkipConnection, including the inner module's state.
    """

    def __init__(self, inner_module: Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_module = inner_module

    def __repr__(self):
        return f"SkipConnection(inner_module={self.inner_module})"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get or set the state of the SkipConnection, including the inner module's state.

        Returns
        -------
        Tuple[str, Dict[str, Any]]
            A tuple representing the class name and inner module's state.
        """
        return self.__class__.__name__, {
            "inner_module": self.inner_module.state
        }

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """
        Set the state of the SkipConnection, including the inner module's state.

        Parameters
        ----------
        value : Tuple[str, Dict[str, Any]]
            A tuple representing the class name and inner module's state.

        Returns
        -------
        None
        """
        class_name, module_state = value["inner_module"]
        module = globals()[class_name](module_state)
        module.state = module_state
        self.inner_module = module

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the output produced by this module.

        Returns
        -------
        Tuple[int, ...]
            The shape of the output.
        """
        return self.inner_module.input_shape

    @property
    def parameters_count(self) -> int:
        """
        Get the total number of parameters in the module.
        """
        return self.inner_module.parameters_count

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform forward propagation through the SkipConnection.

        Parameters
        ----------
        input_data : np.ndarray
            The input data for the forward pass.

        Returns
        -------
        None
        """
        inner_output = self.inner_module.forward(input_data)

        assert self.inner_module.input_shape == self.inner_module.output_shape

        self.output = inner_output + input_data

    def _backward_propagation(self, upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> None:
        """
        Perform backward propagation through the SkipConnection.

        Parameters
        ----------
        upstream_gradients : Optional[np.ndarray]
            The gradients propagated backward from the next layer.
            This can be None if this is the output layer.
        y_true : np.ndarray
            The ground truth values.

        Returns
        -------
        None
        """
        self.retrograde = self.inner_module.backward(upstream_gradients, y_true) + upstream_gradients
