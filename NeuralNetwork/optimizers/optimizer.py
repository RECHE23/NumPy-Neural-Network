from abc import abstractmethod
import numpy as np


class Optimizer:
    """
    Base class for defining optimization algorithms.

    This class defines the fundamental structure and behavior of an optimizer.
    Optimizers are used to update the parameters of a neural network during training.

    Parameters:
    -----------
    learning_rate : float, optional
        The learning rate controlling the step size of parameter updates. Default is 1e-3.
    decay : float, optional
        The learning rate decay factor applied at the end of each epoch. Default is 0.
    lr_min : float, optional
        The minimum allowed learning rate after decay. Default is 0.
    lr_max : float, optional
        The maximum allowed learning rate after decay. Default is np.inf.
    *args, **kwargs
        Additional arguments passed to the optimizer.

    Methods:
    --------
    update(parameters, gradients)
        Update the parameters based on the gradients and learning rate.
    next_epoch()
        Update the learning rate for the next epoch based on decay and clipping.

    """
    def __init__(self, learning_rate: float = 1e-3, decay: float = 0, lr_min: float = 0, lr_max: float = np.inf,
                 *args, **kwargs):
        """
        Initialize the Optimizer with hyperparameters.

        Parameters:
        -----------
        learning_rate : float, optional
            The learning rate controlling the step size of parameter updates. Default is 1e-3.
        decay : float, optional
            The learning rate decay factor applied at the end of each epoch. Default is 0.
        lr_min : float, optional
            The minimum allowed learning rate after decay. Default is 0.
        lr_max : float, optional
            The maximum allowed learning rate after decay. Default is np.inf.
        *args, **kwargs
            Additional arguments passed to the optimizer.

        """
        self.learning_rate: float = learning_rate
        self.decay: float = decay
        self.lr_min: float = lr_min
        self.lr_max: float = lr_max

    def __str__(self) -> str:
        """
        Return a string representation of the optimizer's class name.
        """
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """
        Return a string representation of the optimizer with its hyperparameters.
        """
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate}, decay={self.decay})"

    @abstractmethod
    def update(self, parameters: list, gradients: list) -> None:
        """
        Update the parameters based on the gradients and learning rate.

        Parameters:
        -----------
        parameters : list
            List of parameter arrays.
        gradients : list
            List of gradient arrays corresponding to the parameters.

        """
        raise NotImplementedError

    def next_epoch(self) -> None:
        """
        Update the learning rate for the next epoch based on decay and clipping.
        """
        self.learning_rate *= 1 - self.decay
        self.learning_rate = np.clip(self.learning_rate, self.lr_min, self.lr_max)
