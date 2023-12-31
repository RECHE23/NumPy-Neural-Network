from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from .optimizer import Optimizer


class NesterovMomentum(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG) optimizer.

    This optimizer incorporates momentum and adjusts the gradient computation using
    the previous velocity for faster convergence.

    Parameters:
    -----------
    momentum : float, optional
        The momentum coefficient. Default is 0.9.
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
    momentum : float
        The momentum coefficient.
    velocity : list of arrays or None
        The velocity of parameter updates, initialized to None.

    Methods:
    --------
    update(parameters, gradients)
        Update the parameters using Nesterov Accelerated Gradient (NAG).

    """
    momentum: float
    velocity: Optional[List[np.ndarray]]

    def __init__(self, momentum: float = 0.9, *args, **kwargs):
        """
        Initialize the NesterovMomentum optimizer with hyperparameters.

        Parameters:
        -----------
        momentum : float, optional
            The momentum coefficient. Default is 0.9.
        *args, **kwargs
            Additional arguments passed to the base class Optimizer.

        """
        super().__init__(*args, **kwargs)

        self.velocity = None

        state = Optimizer.state.fget(self)[1]
        state.update({
            "momentum": momentum
        })
        NesterovMomentum.state.fset(self, state)

    def __repr__(self) -> str:
        """
        Return a string representation of the optimizer with its hyperparameters.
        """
        return super().__repr__()[:-1] + f", momentum={self.momentum})"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        state = {
            "momentum": self.momentum
        }

        state.update(Optimizer.state.fget(self)[1])

        return self.__class__.__name__, state

    @state.setter
    def state(self, value) -> None:
        Optimizer.state.fset(self, value)
        self.momentum = value["momentum"]

    def update(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> None:
        """
        Update the parameters using Nesterov Accelerated Gradient (NAG).

        Parameters:
        -----------
        parameters : list of arrays
            List of parameter arrays to be updated.
        gradients : list of arrays
            List of gradient arrays corresponding to the parameters.

        """
        if self.velocity is None:
            self.velocity = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        for i, (velocity, parameter, gradient) in enumerate(zip(self.velocity, parameters, gradients)):
            # Update velocity: velocity = momentum * velocity - lr * gradient
            velocity = self.momentum * velocity - self.lr * gradient

            # Update parameter using Nesterov update: parameter += velocity
            parameter += velocity

            # Update attributes
            self.velocity[i] = velocity
