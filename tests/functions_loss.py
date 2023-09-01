import numpy as np
import unittest
from functools import partial
from .utils import grad_check_loss, make_random_targets, make_random_predictions
from neural_network.functions.loss import *


class TestGradients(unittest.TestCase):
    def setUp(self):
        self.N = 100
        self.K = 10

    def test_binary_cross_entropy_grad(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=True)
        ana_grad = binary_cross_entropy(y_true, y_pred, prime=True)
        num_grad = grad_check_loss(partial(binary_cross_entropy, multioutput='raw_values'), y_true, y_pred)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_categorical_cross_entropy_grad(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=True)
        ana_grad = categorical_cross_entropy(y_true, y_pred, prime=True)
        num_grad = grad_check_loss(partial(categorical_cross_entropy, multioutput='raw_values'), y_true, y_pred)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_mean_absolute_error_grad(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K)
        ana_grad = mean_absolute_error(y_true, y_pred, prime=True)
        num_grad = grad_check_loss(partial(mean_absolute_error, multioutput='raw_values'), y_true, y_pred)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_mean_squared_error_grad(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K)
        ana_grad = mean_squared_error(y_true, y_pred, prime=True)
        num_grad = grad_check_loss(partial(mean_squared_error, multioutput='raw_values'), y_true, y_pred)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
