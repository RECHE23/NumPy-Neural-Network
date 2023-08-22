import numpy as np
from .optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, momentum=.9, *args, **kwargs):
        self.momentum = momentum
        self.velocity = None
        super().__init__(*args, **kwargs)

    def update(self, parameters, gradients):
        if self.velocity is None:
            self.velocity = [np.zeros(shape=parameter.shape, dtype=float) for parameter in parameters]

        for i, (velocity, parameter, gradient) in enumerate(zip(self.velocity, parameters, gradients)):
            velocity = self.momentum * velocity - self.learning_rate * gradient
            parameter += velocity
            self.velocity[i] = velocity

        return parameters
