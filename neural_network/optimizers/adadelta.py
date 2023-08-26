from typing import List, Optional
import numpy as np
from .optimizer import Optimizer


class Adadelta(Optimizer):
    """
    Adadelta optimizer for adaptive gradient updates.

    This optimizer adapts the learning rates individually for each parameter.

    Parameters:
    -----------
    rho : float, optional
        The decay rate for the moving average of squared gradients. Default is 0.9.
    epsilon : float, optional
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
    rho : float
        The decay rate for the moving average of squared gradients.
    epsilon : float
        A small constant added to prevent division by zero.
    avg_sq_gradients : list of arrays or None
        The moving average of squared gradients, initialized to None.
    delta : list of arrays or None
        The moving average of squared parameter updates, initialized to None.

    Methods:
    --------
    update(parameters, gradients)
        Update the parameters using the Adadelta algorithm.

    """
    def __init__(self, rho: float = 0.9, epsilon: float = 1e-7, *args, **kwargs):
        """
        Initialize the Adadelta optimizer with hyperparameters.

        Parameters:
        -----------
        rho : float, optional
            The decay rate for the moving average of squared gradients. Default is 0.9.
        epsilon : float, optional
            A small constant added to prevent division by zero. Default is 1e-7.
        *args, **kwargs
            Additional arguments passed to the base class Optimizer.

        """
        self.rho: float = rho
        self.epsilon: float = epsilon
        self.avg_sq_gradients: Optional[List[np.ndarray]] = None
        self.delta: Optional[List[np.ndarray]] = None
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        """
        Return a string representation of the optimizer with its hyperparameters.
        """
        return super().__repr__()[:-1] + f", rho={self.rho}, epsilon={self.epsilon})"

    def update(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> None:
        """
        Update the parameters using the Adadelta algorithm.

        Parameters:
        -----------
        parameters : list of arrays
            List of parameter arrays to be updated.
        gradients : list of arrays
            List of gradient arrays corresponding to the parameters.

        """
        if self.avg_sq_gradients is None:
            self.avg_sq_gradients = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        if self.delta is None:
            self.delta = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        for i, (avg_sq_gradient, delta, parameter, gradient) in enumerate(zip(self.avg_sq_gradients, self.delta, parameters, gradients)):
            # Update avg_sq_gradient gradient: E[g^2] = rho * E[g^2] + (1 - rho) * gradient^2
            avg_sq_gradient = self.rho * avg_sq_gradient + (1 - self.rho) * gradient * gradient

            # Calculate update: update = gradient * sqrt(delta + epsilon) / sqrt(avg_sq_gradient + epsilon)
            update = gradient * np.sqrt(delta + self.epsilon) / np.sqrt(avg_sq_gradient + self.epsilon)

            # Update parameter: parameter -= lr * update
            parameter -= self.lr * update

            # Update delta: delta = rho * delta + (1 - rho) * update^2
            delta = self.rho * delta + (1 - self.rho) * update * update

            # Update attributes
            self.avg_sq_gradients[i] = avg_sq_gradient
            self.delta[i] = delta
