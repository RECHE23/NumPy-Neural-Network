import numpy as np
import unittest
from .utils import jacobian_check, make_random_predictions
from neural_network.functions.output import *


class TestJacobian(unittest.TestCase):
    def setUp(self):
        self.N = 100
        self.K = 10

    def test_softmin_jacobian(self):
        x = make_random_predictions(self.N, self.K)
        ana_jacobian = jacobian_check(softmin, x)
        num_jacobian = jacobian_check(softmin, x)
        np.testing.assert_allclose(ana_jacobian, num_jacobian, rtol=1e-7, atol=1e-7)

    def test_softmax_jacobian(self):
        x = make_random_predictions(self.N, self.K)
        ana_jacobian = jacobian_check(softmax, x)
        num_jacobian = jacobian_check(softmax, x)
        np.testing.assert_allclose(ana_jacobian, num_jacobian, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
