from typing import List
import numpy as np
from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    This optimizer updates parameters using the stochastic gradient descent algorithm.

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
        Additional arguments passed to the base class Optimizer.

    Methods:
    --------
    update(parameters, gradients)
        Update the parameters using the stochastic gradient descent algorithm.

    """
    def update(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> None:
        """
        Update the parameters using the stochastic gradient descent algorithm.

        Parameters:
        -----------
        parameters : list of arrays
            List of parameter arrays to be updated.
        gradients : list of arrays
            List of gradient arrays corresponding to the parameters.

        """

        for parameter, gradient in zip(parameters, gradients):
            parameter -= self.learning_rate * gradient
