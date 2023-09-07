from typing import Optional, Tuple, Dict, Any
from abc import abstractmethod
import numpy as np
from neural_network.tools import trace
from neural_network.optimizers import Optimizer, Adam


class Module:
    """
    Base class for creating neural network modules.

    This class defines the fundamental structure and behavior of a neural network module.
    It provides methods for both forward and backward propagation, allowing for the
    building of complex neural network architectures.

    Attributes:
    -----------
    input : np.ndarray or None
        The input data passed to the module during forward propagation.
    output : np.ndarray or None
        The output data after applying the module's operations.
    retrograde : np.ndarray or None
        The gradients propagated backward through the module during backward propagation.
    upstream_gradients : np.ndarray or None
        The gradients received from the subsequent module during backward propagation.
    n_samples : int or None
        The number of samples in the input data. Used for handling batch operations.
    optimizer : Optimizer
        The optimizer to use for gradient descent.
    _optimizer_instance : Optimizer or None
        The optimizer instance to update the module's parameters during training.
    _is_training : bool
        Flag indicating whether the module is in training mode.

    Methods:
    --------
    forward(input_data: np.ndarray) -> np.ndarray:
        Perform forward propagation through the module.
    backward(upstream_gradients: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        Perform backward propagation through the module.

    """

    def __init__(self, optimizer: Optional[Optimizer] = None, *args, **kwargs):
        """
        Initialize a neural network module.

        Parameters:
        -----------
        optimizer : Optimizer or None, optional
            The optimizer to use for updating the module's parameters during training.
            If not provided, the default optimizer Adam with learning rate 1e-3 and
            decay 1e-4 will be used.
        """
        assert not args and not kwargs, f"Unused arguments are being passed to base class Module:\nargs:{args}\nkwargs:{kwargs}"

        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None
        self.retrograde: Optional[np.ndarray] = None
        self.upstream_gradients: Optional[np.ndarray] = None
        self.n_samples: Optional[int] = None
        self._optimizer_instance: Optional[Optimizer] = optimizer
        self._is_training: bool = False

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        Make the module callable for forward propagation.

        Returns:
        --------
        np.ndarray
            Output of the module after applying its operations.
        """
        return self.forward(*args, **kwargs)

    def __str__(self) -> str:
        """
        Convert the module to a string representation.

        Returns:
        --------
        str
            String representation of the module.
        """
        return repr(self)

    def __repr__(self) -> str:
        """
        Convert the module to a detailed string representation.

        Returns:
        --------
        str
            Detailed string representation of the module.
        """
        return f"{self.__class__.__name__}()"

    @property
    @abstractmethod
    def state(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the current state of the module.

        Returns:
        --------
        Tuple[str, Dict[str, Any]]
            A tuple containing the class name and a dictionary of the module's state.
        """
        raise NotImplementedError

    @state.setter
    @abstractmethod
    def state(self, valu: Dict[str, Any]) -> None:
        """
        Set the state of the module.

        Parameters:
        -----------
        value : Tuple[str, Dict[str, Any]]
            A tuple containing the class name and a dictionary of the module's state.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters_count(self) -> int:
        """
        Get the total number of parameters in the module.
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Get the input shape of the module's data.

        Returns:
        --------
        Tuple[int, ...]
            Input shape of the module's data.
        """
        return self.input.shape if self.input is not None else None

    @property
    @abstractmethod
    def output_shape(self) -> Tuple[int, ...]:
        """
        Get the output shape (batch_size, ...) of the module's data.

        Returns:
        --------
        Tuple[int, ...]
            Output shape of the module's data.
        """
        raise NotImplementedError

    def is_training(self, value: Optional[bool] = None) -> bool:
        """
        Get or set the training mode of the module.

        Parameters:
        -----------
        value : bool, optional
            If provided, set the training mode of the module.
            If not provided, return the current training mode.

        Returns:
        --------
        bool
            The current training mode of the module. If 'value' is provided, the updated training mode.
        """
        if value is not None:
            self._is_training = value
        return self._is_training

    @property
    def optimizer(self) -> Optimizer:
        """
        Get the optimizer instance for updating the module's parameters during training.

        Returns:
        --------
        Optimizer
            The optimizer instance.
        """
        if self._optimizer_instance is None:
            self._optimizer_instance = Adam()
        return self._optimizer_instance

    @optimizer.setter
    def optimizer(self, value: Optimizer) -> None:
        """
        Set the optimizer instance for updating the module's parameters during training.

        Parameters:
        -----------
        value : Optimizer
            The optimizer instance.
        """
        self._optimizer_instance = value

    @trace()
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the module.

        Parameters:
        -----------
        input_data : np.ndarray, shape (n_samples, ...)
            The input data to propagate through the module.

        Returns:
        --------
        np.ndarray, shape (n_samples, ...)
            The output of the module after applying its operations.
        """
        self.input = input_data
        self.n_samples = input_data.shape[0]
        self._forward_propagation(input_data)
        self.n_samples = None
        return self.output

    @abstractmethod
    def _forward_propagation(self, input_data: np.ndarray) -> None:
        """
        Abstract method to implement the actual forward propagation.

        Parameters:
        -----------
        input_data : np.ndarray, shape (n_samples, ...)
            The input data to propagate through the module.
        """
        raise NotImplementedError

    @trace()
    def backward(self, upstream_gradients: Optional[np.ndarray], y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform backward propagation through the module.

        Parameters:
        -----------
        upstream_gradients : np.ndarray, shape (n_samples, ...)
            Gradients received from the subsequent module during backward propagation.
        y_true : np.ndarray, shape (n_samples, ...)
            The true target values corresponding to the input data.

        Returns:
        --------
        np.ndarray, shape (n_samples, ...)
            Gradients propagated backward through the module.
        """
        assert self.input is not None, "You need a forward propagation before a backward propagation..."
        self.upstream_gradients = upstream_gradients
        self.n_samples = None if upstream_gradients is None else upstream_gradients.shape[0]
        self._backward_propagation(upstream_gradients, y_true)
        self.n_samples = None
        return self.retrograde

    @abstractmethod
    def _backward_propagation(self, upstream_gradients: Optional[np.ndarray], y_true: np.ndarray) -> None:
        """
        Abstract method to implement the actual backward propagation.

        Parameters:
        -----------
        upstream_gradients : np.ndarray, shape (n_samples, ...)
            Gradients received from the subsequent module during backward propagation.
        y_true : np.ndarray, shape (n_samples, ...)
            The true target values corresponding to the input data.
        """
        raise NotImplementedError
