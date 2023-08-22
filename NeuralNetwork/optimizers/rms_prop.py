import numpy as np
from .optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(self, rho=0.9, epsilon=1e-7, *args, **kwargs):
        self.rho = rho
        self.epsilon = epsilon
        self.cache = None
        super().__init__(*args, **kwargs)

    def update(self, parameters, gradients):
        if self.cache is None:
            self.cache = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        for i, (cached, parameter, gradient) in enumerate(zip(self.cache, parameters, gradients)):
            cached = self.rho * cached + (1 - self.rho) * gradient * gradient
            parameter -= self.learning_rate * gradient / (np.sqrt(cached) + self.epsilon)
            self.cache[i] = cached

        return parameters
