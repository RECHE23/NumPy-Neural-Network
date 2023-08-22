from typing import List, Optional
import numpy as np
from .optimizer import Optimizer


class Adamax(Optimizer):
    """
    Adamax optimizer for adaptive gradient updates.

    This optimizer adapts the learning rates individually for each parameter.

    Parameters:
    -----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates. Default is 0.9.
    beta2 : float, optional
        Exponential decay rate for the infinite norm estimates. Default is 0.999.
    epsilon : float, optional
        A small constant added to prevent division by zero. Default is 1e-8.
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
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the infinite norm estimates.
    epsilon : float
        A small constant added to prevent division by zero.
    ms : list of arrays or None
        First moment estimates, initialized to None.
    vs : list of arrays or None
        Second moment estimates, initialized to None.
    t : int
        Time step.

    Methods:
    --------
    update(parameters, gradients)
        Update the parameters using the Adamax algorithm.

    """
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, *args, **kwargs):
        """
        Initialize the Adamax optimizer with hyperparameters.

        Parameters:
        -----------
        beta1 : float, optional
            Exponential decay rate for the first moment estimates. Default is 0.9.
        beta2 : float, optional
            Exponential decay rate for the infinite norm estimates. Default is 0.999.
        epsilon : float, optional
            A small constant added to prevent division by zero. Default is 1e-8.
        *args, **kwargs
            Additional arguments passed to the base class Optimizer.

        """
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        self.ms: Optional[List[np.ndarray]] = None  # First moment estimates
        self.vs: Optional[List[np.ndarray]] = None  # Second moment estimates
        self.t: int = 0                             # Time step
        super().__init__(*args, **kwargs)

    def update(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Update the parameters using the Adamax algorithm.

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
        self.t += 1
        a_t = self.learning_rate / (1 - self.beta1 ** self.t)

        if self.ms is None:
            self.ms = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        if self.vs is None:
            self.vs = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        updated_parameters = []
        for i, (m, v, parameter, gradient) in enumerate(zip(self.ms, self.vs, parameters, gradients)):
            # Update first moment estimate: m = beta1 * m + (1 - beta1) * gradient
            m = self.beta1 * m + (1 - self.beta1) * gradient

            # Update second moment estimate: v = max(beta2 * v, abs(gradient))
            v = np.maximum(self.beta2 * v, np.abs(gradient))

            # Update parameter: parameter -= a_t * m / (v + epsilon)
            parameter -= a_t * m / (v + self.epsilon)

            # Update attributes
            self.ms[i] = m
            self.vs[i] = v
            updated_parameters.append(parameter)

        return updated_parameters
