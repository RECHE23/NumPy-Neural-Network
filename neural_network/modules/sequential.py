from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from . import Module


class Sequential(Module):
    """
    A sequential neural network layer composed of multiple sub-layers.

    This class allows you to create a sequential neural network layer by adding
    sub-layers in a sequence. The forward and backward propagation methods
    will be executed on each sub-layer in the order they were added.

    Parameters:
    -----------
    layers : list of Module
        The list of sub-layers in the sequential model.

    Methods:
    --------
    add(layer: Module) -> None:
        Add a sub-layer to the sequential model.

    _forward_propagation(input_data: np.ndarray) -> None:
        Perform forward propagation through all sub-layers.

    _backward_propagation(upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> None:
        Perform backward propagation through all sub-layers.
    """

    def __init__(self, *layers: Module):
        """
        Initialize a sequential neural network layer.

        Parameters:
        -----------
        layers : list of Module, optional
            The list of sub-layers in the sequential model.
        """
        super().__init__()

        self.sub_layers: List[Module] = list(layers) if layers is not None else []

    def __str__(self):
        layer_str = "\n".join([f"\t({i}): {layer}" for i, layer in enumerate(self.sub_layers)])
        return f"Sequential(\n{layer_str}\n)"

    def __repr__(self):
        """
        Return a string representation of the sequential module.
        """
        layer_str = ", ".join([f"{i}: {repr(layer)}" for i, layer in enumerate(self.sub_layers)])
        return f"Sequential({layer_str})"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        return self.__class__.__name__, {
            "sub_layers_state": [layer.state for layer in self.sub_layers]
        }

    @state.setter
    def state(self, value) -> None:
        for class_name, layer_state in value["sub_layers_state"]:
            layer = globals()[class_name](layer_state)
            layer.state = layer_state
            self.sub_layers.append(layer)

    def add(self, layer: Module) -> None:
        """
        Add a sub-layer to the sequential model.

        Parameters:
        -----------
        layer : Module
            The sub-layer to be added to the sequential model.
        """
        self.sub_layers.append(layer)

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape (batch_size, ...) of the sequential model.

        Returns:
        --------
        Tuple[int, ...]
            Output shape of the sequential model's data.
        """
        if self.sub_layers:
            return self.sub_layers[-1].output_shape
        else:
            return ()

    @property
    def parameters_count(self) -> int:
        """
        Get the total number of parameters in the sequential model.
        """
        return sum(layer.parameters_count for layer in self.sub_layers)

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform forward propagation through all sub-layers.

        Parameters:
        -----------
        input_data : np.ndarray, shape (n_samples, ...)
            The input data to propagate through the sequential model.
        """
        for layer in self.sub_layers:
            input_data = layer.forward(input_data)
        self.output = input_data

    def _backward_propagation(self, upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> None:
        """
        Perform backward propagation through all sub-layers.

        Parameters:
        -----------
        upstream_gradients : np.ndarray, shape (n_samples, ...)
            Gradients received from the subsequent layer during backward propagation.
        y_true : np.ndarray, shape (n_samples, ...)
            The true target values corresponding to the input data.
        """
        retrograde = upstream_gradients
        for layer in reversed(self.sub_layers):
            retrograde = layer.backward(retrograde, y_true)
        self.retrograde = retrograde

    def is_training(self, value: Optional[bool] = None) -> bool:
        """
        Get or set the training status of the sequential layer.

        Parameters:
        -----------
        value : bool, optional
            If provided, set the training status to this value. If None, return the current training status.

        Returns:
        --------
        training_status : bool
            Current training status of the neural network.
        """
        if value is not None:
            self._is_training = value
            for layer in self.sub_layers:
                layer.is_training(value)
        return self._is_training
