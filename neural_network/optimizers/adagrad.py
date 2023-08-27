from typing import List, Optional
import numpy as np
from .optimizer import Optimizer


class Adagrad(Optimizer):
    """
    Adagrad optimizer for adaptive gradient updates.

    This optimizer adapts the learning rates individually for each parameter.

    Parameters:
    -----------
    eps : float, optional
        A small constant added to prevent division by zero. Default is 1e-7.
    lr : float, optional
        The learning rate controlling the step size of parameter updates. Default is 1e-3.
    lr_decay : float, optional
        The learning rate decay factor applied at the end of each epoch. Default is 0.
    lr_min : float, optional
        The minimum allowed learning rate after decay. Default is 0.
    lr_max : float, optional
        The maximum allowed learning rate after decay. Default is np.inf.
    *args, **kwargs
        Additional arguments passed to the base class Optimizer.

    Attributes:
    -----------
    eps : float
        A small constant added to prevent division by zero.
    sum_sq_gradients : list of arrays or None
        The cumulative sum of squared gradients, initialized to None.

    Methods:
    --------
    update(parameters, gradients)
        Update the parameters using the Adagrad algorithm.

    """
    def __init__(self, epsilon: float = 1e-7, *args, **kwargs):
        """
        Initialize the Adagrad optimizer with hyperparameters.

        Parameters:
        -----------
        eps : float, optional
            A small constant added to prevent division by zero. Default is 1e-7.
        *args, **kwargs
            Additional arguments passed to the base class Optimizer.

        """
        self.epsilon: float = epsilon
        self.sum_sq_gradients: Optional[List[np.ndarray]] = None
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        """
        Return a string representation of the optimizer with its hyperparameters.
        """
        return super().__repr__()[:-1] + f", eps={self.epsilon})"

    def update(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> None:
        """
        Update the parameters using the Adagrad algorithm.

        Parameters:
        -----------
        parameters : list of arrays
            List of parameter arrays to be updated.
        gradients : list of arrays
            List of gradient arrays corresponding to the parameters.

        """
        if self.sum_sq_gradients is None:
            self.sum_sq_gradients = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        for i, (sum_sq_gradient, parameter, gradient) in enumerate(zip(self.sum_sq_gradients, parameters, gradients)):
            # Update sum_sq_gradient gradient: sum_sq_gradient += gradient^2
            sum_sq_gradient += gradient * gradient

            # Update parameter: parameter -= lr * gradient / (sqrt(sum_sq_gradient) + eps)
            parameter -= self.lr * gradient / (np.sqrt(sum_sq_gradient) + self.epsilon)

            # Update attributes
            self.sum_sq_gradients[i] = sum_sq_gradient
