import numpy as np
import unittest
from neural_network.functions.utils import convert_targets
from neural_network.functions.loss import *
from neural_network.functions.output import softmax


def grad_check(f, y_true, y_pred, eps=1e-7):
    grad = np.zeros_like(y_true)
    for i in range(grad.shape[1]):
        y_pred_p = y_pred.copy()
        y_pred_p[:, i] += eps
        y_pred_m = y_pred.copy()
        y_pred_m[:, i] -= eps
        grad[:, i] = (f(y_true, y_pred_p) - f(y_true, y_pred_m)) / (2 * eps)
    return grad


def random_targets(N, K):
    array = np.append(np.arange(0, K), np.random.randint(0, K, size=(N - K)))
    np.random.shuffle(array)
    return array


class TestGradients(unittest.TestCase):

    def test_categorical_cross_entropy_grad(self):
        y_true = convert_targets(random_targets(5, 3))
        y_pred = softmax(np.random.randn(5, 3))
        ana_grad = categorical_cross_entropy(y_true, y_pred, prime=True)
        num_grad = grad_check(categorical_cross_entropy, y_true, y_pred)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_mean_squared_error_grad(self):
        y_true = np.random.randint(low=0, high=2, size=(5, 3)).astype(np.float64)
        y_pred = np.random.randn(5, 3)
        ana_grad = mean_squared_error(y_true, y_pred, prime=True)
        num_grad = grad_check(mean_squared_error, y_true, y_pred)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
