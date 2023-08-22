import numpy as np
from .optimizer import Optimizer


class Adamax(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, *args, **kwargs):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.ms = None  # First moment estimates
        self.vs = None  # Second moment estimates
        self.t = 0      # Time step
        super().__init__(*args, **kwargs)

    def update(self, parameters, gradients):
        self.t += 1

        a_t = self.learning_rate / (1 - self.beta1**self.t)

        if self.ms is None:
            self.ms = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        if self.vs is None:
            self.vs = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        for i, (m, v, parameter, gradient) in enumerate(zip(self.ms, self.vs, parameters, gradients)):
            m = self.beta1 * m + (1 - self.beta1) * gradient
            v = np.maximum(self.beta2 * v, np.abs(gradient))
            parameter -= a_t * m / (v + self.epsilon)
            self.ms[i] = m
            self.vs[i] = v

        return parameters
