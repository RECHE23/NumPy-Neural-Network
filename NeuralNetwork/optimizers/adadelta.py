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
    learning_rate : float, optional
        The learning rate controlling the step size of parameter updates. Default is 1e-3.
    decay : float, optional
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
    cache : list of arrays or None
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
        self.cache: Optional[List[np.ndarray]] = None
        self.delta: Optional[List[np.ndarray]] = None
        super().__init__(*args, **kwargs)

    def update(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Update the parameters using the Adadelta algorithm.

        Parameters:
        -----------
        parameters : list of arrays
            List of parameter arrays to be updated.
        gradients : list of arrays
            List of gradient arrays corresponding to the parameters.

        Returns:
        --------
        updated_parameters : list of arrays
            List of updated parameter arrays.

        """
        if self.cache is None:
            self.cache = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        if self.delta is None:
            self.delta = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        updated_parameters = []
        for i, (cached, delta, parameter, gradient) in enumerate(zip(self.cache, self.delta, parameters, gradients)):
            # Update cached gradient: E[g^2] = rho * E[g^2] + (1 - rho) * gradient^2
            cached = self.rho * cached + (1 - self.rho) * gradient * gradient

            # Calculate update: update = gradient * sqrt(delta + epsilon) / sqrt(cached + epsilon)
            update = gradient * np.sqrt(delta + self.epsilon) / np.sqrt(cached + self.epsilon)

            # Update parameter: parameter -= learning_rate * update
            parameter -= self.learning_rate * update

            # Update delta: delta = rho * delta + (1 - rho) * update^2
            delta = self.rho * delta + (1 - self.rho) * update * update

            # Update attributes
            self.cache[i] = cached
            self.delta[i] = delta
            updated_parameters.append(parameter)

        return updated_parameters
