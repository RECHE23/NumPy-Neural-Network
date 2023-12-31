from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from .optimizer import Optimizer


class RMSprop(Optimizer):
    """
    RMSprop optimizer.

    This optimizer adapts the learning rate for each parameter based on the moving average
    of squared gradients. It aims to mitigate the vanishing and exploding gradient problems.

    Parameters:
    -----------
    rho : float, optional
        The decay factor for the moving average. Default is 0.9.
    eps : float, optional
        A small value added to the denominator for numerical stability. Default is 1e-7.
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
        The decay factor for the moving average.
    eps : float
        A small value added to the denominator for numerical stability.
    squared_gradient_accumulations : list of arrays or None
        The moving average of squared gradients, initialized to None.

    Methods:
    --------
    update(parameters, gradients)
        Update the parameters using RMSprop optimization.

    """
    rho: float
    epsilon: float
    squared_gradient_accumulations: Optional[List[np.ndarray]]

    def __init__(self, rho: float = 0.9, epsilon: float = 1e-7, *args, **kwargs):
        """
        Initialize the RMSprop optimizer with hyperparameters.

        Parameters:
        -----------
        rho : float, optional
            The decay factor for the moving average. Default is 0.9.
        epsilon : float, optional
            A small value added to the denominator for numerical stability. Default is 1e-7.
        *args, **kwargs
            Additional arguments passed to the base class Optimizer.

        """
        super().__init__(*args, **kwargs)

        self.rho: float
        self.epsilon: float
        self.squared_gradient_accumulations: Optional[List[np.ndarray]] = None

        state = Optimizer.state.fget(self)[1]
        state.update({
            "rho": rho,
            "epsilon": epsilon
        })
        RMSprop.state.fset(self, state)

    def __repr__(self) -> str:
        """
        Return a string representation of the optimizer with its hyperparameters.
        """
        return super().__repr__()[:-1] + f", rho={self.rho}, eps={self.epsilon})"

    @property
    def state(self) -> Tuple[str, Dict[str, Any]]:
        state = {
            "rho": self.rho,
            "epsilon": self.epsilon
        }

        state.update(Optimizer.state.fget(self)[1])

        return self.__class__.__name__, state

    @state.setter
    def state(self, value) -> None:
        Optimizer.state.fset(self, value)
        self.rho = value["rho"]
        self.epsilon = value["epsilon"]

    def update(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> None:
        """
        Update the parameters using RMSprop optimization.

        Parameters:
        -----------
        parameters : list of arrays
            List of parameter arrays to be updated.
        gradients : list of arrays
            List of gradient arrays corresponding to the parameters.

        """
        if self.squared_gradient_accumulations is None:
            self.squared_gradient_accumulations = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        for i, (sq_grad_accum, parameter, gradient) in enumerate(zip(self.squared_gradient_accumulations, parameters, gradients)):
            # Update sq_grad_accum: sq_grad_accum = rho * sq_grad_accum + (1 - rho) * gradient * gradient
            sq_grad_accum = self.rho * sq_grad_accum + (1 - self.rho) * gradient * gradient

            # Update parameter using RMSprop update: parameter -= lr * gradient / (sqrt(sq_grad_accum) + eps)
            parameter -= self.lr * gradient / (np.sqrt(sq_grad_accum) + self.epsilon)

            # Update attribute
            self.squared_gradient_accumulations[i] = sq_grad_accum
