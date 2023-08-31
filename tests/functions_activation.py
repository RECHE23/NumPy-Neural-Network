import numpy as np
import unittest
from neural_network.functions.activation import *


def grad_check(f, x, eps=1e-7):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


class TestGradients(unittest.TestCase):

    def test_relu_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = relu(x, prime=True)
        num_grad = grad_check(relu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_tanh_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = tanh(x, prime=True)
        num_grad = grad_check(tanh, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_sigmoid_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = sigmoid(x, prime=True)
        num_grad = grad_check(sigmoid, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_leaky_relu_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = leaky_relu(x, prime=True)
        num_grad = grad_check(leaky_relu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_elu_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = elu(x, prime=True)
        num_grad = grad_check(elu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_swish_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = swish(x, prime=True)
        num_grad = grad_check(swish, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_arctan_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = arctan(x, prime=True)
        num_grad = grad_check(arctan, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_gaussian_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = gaussian(x, prime=True)
        num_grad = grad_check(gaussian, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_silu_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = silu(x, prime=True)
        num_grad = grad_check(silu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_bent_identity_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = bent_identity(x, prime=True)
        num_grad = grad_check(bent_identity, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_selu_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = selu(x, prime=True)
        num_grad = grad_check(selu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_celu_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = celu(x, prime=True)
        num_grad = grad_check(celu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_erf_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = erf(x, prime=True)
        num_grad = grad_check(erf, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_gelu_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = gelu(x, prime=True)
        num_grad = grad_check(gelu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_softplus_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = softplus(x, prime=True)
        num_grad = grad_check(softplus, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_mish_grad(self):
        x = np.random.randn(5, 3)
        ana_grad = mish(x, prime=True)
        num_grad = grad_check(mish, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
