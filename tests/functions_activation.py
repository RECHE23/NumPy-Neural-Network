from .utils import *
from neural_network.functions.activation import *


class TestGradients(unittest.TestCase):
    def setUp(self):
        self.N = 100
        self.K = 10

    def test_relu_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = relu(x, prime=True)
        num_grad = grad_check_activation(relu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_tanh_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = tanh(x, prime=True)
        num_grad = grad_check_activation(tanh, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_sigmoid_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = sigmoid(x, prime=True)
        num_grad = grad_check_activation(sigmoid, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_leaky_relu_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = leaky_relu(x, prime=True)
        num_grad = grad_check_activation(leaky_relu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_elu_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = elu(x, prime=True)
        num_grad = grad_check_activation(elu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_swish_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = swish(x, prime=True)
        num_grad = grad_check_activation(swish, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_arctan_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = arctan(x, prime=True)
        num_grad = grad_check_activation(arctan, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_gaussian_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = gaussian(x, prime=True)
        num_grad = grad_check_activation(gaussian, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_silu_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = silu(x, prime=True)
        num_grad = grad_check_activation(silu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_bent_identity_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = bent_identity(x, prime=True)
        num_grad = grad_check_activation(bent_identity, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_selu_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = selu(x, prime=True)
        num_grad = grad_check_activation(selu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_celu_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = celu(x, prime=True)
        num_grad = grad_check_activation(celu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_erf_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = erf(x, prime=True)
        num_grad = grad_check_activation(erf, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_gelu_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = gelu(x, prime=True)
        num_grad = grad_check_activation(gelu, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_softplus_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = softplus(x, prime=True)
        num_grad = grad_check_activation(softplus, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_mish_grad(self):
        x = make_random_predictions(self.N, self.K)
        ana_grad = mish(x, prime=True)
        num_grad = grad_check_activation(mish, x)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
