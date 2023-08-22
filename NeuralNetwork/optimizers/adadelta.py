import numpy as np
from .optimizer import Optimizer


class Adadelta(Optimizer):
    def __init__(self, rho=0.9, epsilon=1e-7, *args, **kwargs):
        self.rho = rho
        self.epsilon = epsilon
        self.cache = None
        self.delta = None
        super().__init__(*args, **kwargs)

    def update(self, parameters, gradients):
        if self.cache is None:
            self.cache = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        if self.delta is None:
            self.delta = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        for i, (cached, delta, parameter, gradient) in enumerate(zip(self.cache, self.delta, parameters, gradients)):
            cached = self.rho * cached + (1 - self.rho) * gradient * gradient
            update = gradient * np.sqrt(delta + self.epsilon) / np.sqrt(cached + self.epsilon)
            parameter -= self.learning_rate * update
            delta = self.rho * delta + (1 - self.rho) * update * update
            self.cache[i] = cached
            self.delta[i] = delta

        return parameters
