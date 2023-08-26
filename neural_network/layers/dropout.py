from typing import Tuple, Optional
import numpy as np
from . import Layer


class Dropout(Layer):
    def __init__(self, p: float = 0.5, *args, **kwargs):
        """
        Create a dropout layer.

        Parameters:
        ----------
        p : float
            The probability of dropping out a neuron's output. Must be in the range [0, 1).

        *args, **kwargs : Additional arguments
            Additional arguments to be passed to the parent class constructor.
        """
        assert 0 <= p < 1, "Dropout rate must be in the range [0, 1)"
        super().__init__(*args, **kwargs)
        self.dropout_rate = p
        self.scaling = 1.0 / (1.0 - p)
        self.dropout_mask = None

    def __repr__(self) -> str:
        """
        Return a string representation of the dropout layer.
        """
        return f"{self.__class__.__name__}(p={self.dropout_rate})"

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape of the layer.

        Returns:
        ----------
        output_shape : tuple
            The shape of the layer's output data.
        """
        return self.input.shape

    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Perform the forward propagation step.

        Parameters:
        ----------
        input_data : np.ndarray
            The input data.
        """
        if self.is_training():
            self.dropout_mask = np.random.rand(*input_data.shape) > self.dropout_rate
            self.output = self.dropout_mask * input_data * self.scaling
        else:
            self.output = input_data

    def _backward_propagation(self, upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> None:
        """
        Perform the backward propagation step.

        Parameters:
        ----------
        upstream_gradients : np.ndarray or None
            Gradients flowing in from the layer above.

        y_true : np.ndarray
            The true labels for the data.
        """
        self.retrograde = self.dropout_mask * upstream_gradients * self.scaling
