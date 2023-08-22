from .optimizer import Optimizer


class SGD(Optimizer):
    def update(self, parameters, gradients):
        for parameter, gradient in zip(parameters, gradients):
            parameter -= self.learning_rate * gradient
        return parameters
