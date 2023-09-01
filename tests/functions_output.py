import numpy as np
import unittest
from neural_network.functions.output import *


def jacobian_check(f, x, eps=1e-7):
    x_flat = x.flatten()
    grad_matrix = np.zeros((x.size, x_flat.size))

    for i in range(x_flat.size):
        x_eps_plus = x_flat.copy()
        x_eps_plus[i] += eps
        f_plus = f(x_eps_plus)

        x_eps_minus = x_flat.copy()
        x_eps_minus[i] -= eps
        f_minus = f(x_eps_minus)

        grad_matrix[:, i] = (f_plus - f_minus) / (2 * eps)

    return grad_matrix.reshape(x.shape + (x.size,))


class TestJacobian(unittest.TestCase):

    def test_softmin_jacobian(self):
        x = np.random.randn(5, 3)
        ana_jacobian = jacobian_check(softmin, x)
        num_jacobian = jacobian_check(softmin, x)
        np.testing.assert_allclose(ana_jacobian, num_jacobian, rtol=1e-7, atol=1e-7)

    def test_softmax_jacobian(self):
        x = np.random.randn(5, 3)
        ana_jacobian = jacobian_check(softmax, x)
        num_jacobian = jacobian_check(softmax, x)
        np.testing.assert_allclose(ana_jacobian, num_jacobian, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
