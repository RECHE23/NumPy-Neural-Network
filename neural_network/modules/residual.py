import numpy as np
from typing import Any, Dict, Optional, Tuple
from neural_network.modules import Module, Sequential


class ResidualBlock(Module):
    """
    ResidualBlock Module

    This module represents a block with a skip connection in a neural network.
    It combines the output of an inner module with its input, effectively allowing the network to bypass one or more layers.

    Parameters
    ----------
    inner_modules : Module
        The inner modules whose output is combined with the input.

    Attributes
    ----------
    inner_modules : Module
        The inner module contained within the ResidualBlock.
    output_shape : Tuple[int, ...]
        The shape of the output produced by this module.
    parameters_count : int
        The total number of parameters in this module and its inner module.

    Methods
    -------
    forward(input_data: np.ndarray) -> None
        Perform forward propagation through the ResidualBlock.
    backward(upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> None
        Perform backward propagation through the ResidualBlock.

    Properties
    ----------
    state : Tuple[str, Dict[str, Any]]
        Get or set the state of the ResidualBlock, including the inner module's state.
    """

    def __init__(self, *inner_modules: Module, **kwargs):
        super().__init__(**kwargs)
        self.inner_modules = Sequential(*inner_modules)

    def __repr__(self):
        return f"ResidualBlock(inner_modules={self.inner_modules})"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get or set the state of the ResidualBlock, including the inner module's state.

        Returns
        -------
        Tuple[str, Dict[str, Any]]
            A tuple representing the class name and inner module's state.
        """
        return self.__class__.__name__, {
            "inner_modules": self.inner_modules.state
        }

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """
        Set the state of the ResidualBlock, including the inner module's state.

        Parameters
        ----------
        value : Tuple[str, Dict[str, Any]]
            A tuple representing the class name and inner module's state.

        Returns
        -------
        None
        """
        class_name, module_state = value["inner_modules"]
        module = globals()[class_name](module_state)
        module.state = module_state
        self.inner_modules = module

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the output produced by this module.

        Returns
        -------
        Tuple[int, ...]
            The shape of the output.
        """
        return self.inner_modules.input_shape

    @property
    def parameters_count(self) -> int:
        """
        Get the total number of parameters in the module.
        """
        return self.inner_modules.parameters_count

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform forward propagation through the ResidualBlock.

        Parameters
        ----------
        input_data : np.ndarray
            The input data for the forward pass.

        Returns
        -------
        None
        """
        inner_output = self.inner_modules.forward(input_data)

        assert self.inner_modules.input_shape == self.inner_modules.output_shape, f"Input shape: {self.inner_modules.input_shape} ≠ Output shape: {self.inner_modules.output_shape}"

        self.output = inner_output + input_data

    def _backward_propagation(self, upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> None:
        """
        Perform backward propagation through the ResidualBlock.

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
        self.retrograde = self.inner_modules.backward(upstream_gradients, y_true) + upstream_gradients
