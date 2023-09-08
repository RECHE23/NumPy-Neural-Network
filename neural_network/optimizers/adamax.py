from typing import List, Optional, Tuple, Dict, Any
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
        Exponential decay rate for the infinite metric estimates. Default is 0.999.
    eps : float, optional
        A small constant added to prevent division by zero. Default is 1e-8.
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
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the infinite metric estimates.
    eps : float
        A small constant added to prevent division by zero.
    first_moments : list of arrays or None
        First moment estimates, initialized to None.
    second_moments : list of arrays or None
        Second moment estimates, initialized to None.
    time_step : int
        Time step.

    Methods:
    --------
    update(parameters, gradients)
        Update the parameters using the Adamax algorithm.

    """
    beta1: float
    beta2: float
    epsilon: float
    first_moments: Optional[List[np.ndarray]]
    second_moments: Optional[List[np.ndarray]]
    time_step: int

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, *args, **kwargs):
        """
        Initialize the Adamax optimizer with hyperparameters.

        Parameters:
        -----------
        beta1 : float, optional
            Exponential decay rate for the first moment estimates. Default is 0.9.
        beta2 : float, optional
            Exponential decay rate for the infinite metric estimates. Default is 0.999.
        eps : float, optional
            A small constant added to prevent division by zero. Default is 1e-8.
        *args, **kwargs
            Additional arguments passed to the base class Optimizer.

        """
        super().__init__(*args, **kwargs)

        self.first_moments = None
        self.second_moments = None
        self.time_step = 0

        state = Optimizer.state.fget(self)[1]
        state.update({
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon
        })
        Adamax.state.fset(self, state)

    def __repr__(self) -> str:
        """
        Return a string representation of the optimizer with its hyperparameters.
        """
        return super().__repr__()[:-1] + f", beta1={self.beta1}, beta2={self.beta2}, eps={self.epsilon})"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        state = {
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon
        }

        state.update(Optimizer.state.fget(self)[1])

        return self.__class__.__name__, state

    @state.setter
    def state(self, value) -> None:
        Optimizer.state.fset(self, value)
        self.beta1 = value["beta1"]
        self.beta2 = value["beta2"]
        self.epsilon = value["epsilon"]

    def update(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> None:
        """
        Update the parameters using the Adamax algorithm.

        Parameters:
        -----------
        parameters : list of arrays
            List of parameter arrays to be updated.
        gradients : list of arrays
            List of gradient arrays corresponding to the parameters.

        """
        self.time_step += 1

        # Compute the bias-corrected learning rate
        adjusted_learning_rate = self.lr / (1 - self.beta1 ** self.time_step)

        if self.first_moments is None:
            self.first_moments = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        if self.second_moments is None:
            self.second_moments = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        for i, (first_moment, second_moment, parameter, gradient) in enumerate(zip(self.first_moments, self.second_moments, parameters, gradients)):
            # Update first moment estimate: first_moment = beta1 * first_moment + (1 - beta1) * gradient
            first_moment = self.beta1 * first_moment + (1 - self.beta1) * gradient

            # Update second moment estimate: second_moment = max(beta2 * second_moment, abs(gradient))
            second_moment = np.maximum(self.beta2 * second_moment, np.abs(gradient))

            # Update parameter: parameter -= adjusted_learning_rate * first_moment / (second_moment + eps)
            parameter -= adjusted_learning_rate * first_moment / (second_moment + self.epsilon)

            # Update attributes
            self.first_moments[i] = first_moment
            self.second_moments[i] = second_moment
