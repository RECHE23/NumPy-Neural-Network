from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, learning_rate=1e-3, decay=0, lr_min=0, lr_max=np.inf, *args, **kwargs):
        self.learning_rate = learning_rate
        self.decay = decay
        self.lr_min = lr_min
        self.lr_max = lr_max

    @abstractmethod
    def update(self, parameters, gradients):
        raise NotImplementedError

    def next_epoch(self):
        self.learning_rate *= 1 - self.decay
        self.learning_rate = np.clip(self.learning_rate, self.lr_min, self.lr_max)
