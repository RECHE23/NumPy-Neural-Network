from abc import abstractmethod
from typing import Tuple, Dict, Any

import numpy as np


class Optimizer:
    """
    Base class for defining optimization algorithms.

    This class defines the fundamental structure and behavior of an optimizer.
    Optimizers are used to update the parameters of a neural network during training.

    Parameters:
    -----------
    lr : float, optional
        The learning rate controlling the step size of parameter updates. Default is 1e-3.
    lr_decay : float, optional
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
    lr: float
    lr_decay: float
    lr_min: float
    lr_max: float

    def __init__(self, lr: float = 1e-3, lr_decay: float = 0, lr_min: float = 0, lr_max: float = np.inf,
                 *args, **kwargs):
        """
        Initialize the Optimizer with hyperparameters.

        Parameters:
        -----------
        lr : float, optional
            The learning rate controlling the step size of parameter updates. Default is 1e-3.
        lr_decay : float, optional
            The learning rate decay factor applied at the end of each epoch. Default is 0.
        lr_min : float, optional
            The minimum allowed learning rate after decay. Default is 0.
        lr_max : float, optional
            The maximum allowed learning rate after decay. Default is np.inf.
        *args, **kwargs
            Additional arguments passed to the optimizer.

        """

        state = {
            "lr": lr,
            "lr_decay": lr_decay,
            "lr_min": lr_min,
            "lr_max": lr_max
        }
        Optimizer.state.fset(self, state)

    def __str__(self) -> str:
        """
        Return a string representation of the optimizer's class name.
        """
        return repr(self)

    def __repr__(self) -> str:
        """
        Return a string representation of the optimizer with its hyperparameters.
        """
        return f"{self.__class__.__name__}(lr={self.lr}, lr_decay={self.lr_decay})"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        return self.__class__.__name__, {
            "lr": float(self.lr),
            "lr_decay": float(self.lr_decay),
            "lr_min": float(self.lr_min),
            "lr_max": float(self.lr_max)
        }

    @state.setter
    def state(self, value) -> None:
        assert value["lr_min"] >= 0, "Learning rate should be positive."
        assert value["lr_decay"] >= 0, "Decay should be positive."
        assert value["lr_min"] < value["lr"] < value["lr_max"], f"Learning rate should be in the range ({value['lr_min']}, {value['lr_max']})."

        self.lr = value["lr"]
        self.lr_decay = value["lr_decay"]
        self.lr_min = value["lr_min"]
        self.lr_max = value["lr_max"]

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
        self.lr *= 1 - self.lr_decay
        self.lr = np.clip(self.lr, self.lr_min, self.lr_max)
