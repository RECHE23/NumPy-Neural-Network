from .utils import *
from neural_network.modules.activation import *


class TestActivationFunctions(unittest.TestCase):

    def setUp(self):
        self.size = (32, 64, 28, 28)

        # Generate random input data
        self.input_data = np.random.randn(*self.size).astype(np.float32)

        # Generate random upstream gradients
        self.upstream_gradients = np.random.randn(*self.size)

        # Create a PyTorch tensor with gradient tracking enabled
        self.torch_input = torch.tensor(self.input_data, dtype=torch.float32, requires_grad=True)

    def test_elu_forward(self):
        # Compute the forward pass outputs:
        torch_output_ = torch_output(torch.nn.ELU(), self.input_data)
        tensorflow_output_ = tensorflow_output(tensorflow.keras.layers.ELU(), self.input_data)
        custom_output_ = custom_output(ELU(), self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-6)

    def test_elu_backward(self):
        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(torch.nn.ELU(), self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(tensorflow.keras.layers.ELU(), self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(ELU(), self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-7, atol=1e-6)

    def test_leakyrelu_forward(self):
        alpha = 0.01

        # Compute the forward pass outputs:
        torch_output_ = torch_output(torch.nn.LeakyReLU(negative_slope=alpha), self.input_data)
        tensorflow_output_ = tensorflow_output(tensorflow.keras.layers.LeakyReLU(alpha=alpha), self.input_data)
        custom_output_ = custom_output(LeakyReLU(alpha=alpha), self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-6)

    def test_leakyrelu_backward(self):
        alpha = 0.01

        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(torch.nn.LeakyReLU(negative_slope=alpha), self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(tensorflow.keras.layers.LeakyReLU(alpha=alpha), self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(LeakyReLU(alpha=alpha), self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-7, atol=1e-6)

    def test_relu_forward(self):
        # Compute the forward pass outputs:
        torch_output_ = torch_output(torch.nn.ReLU(), self.input_data)
        tensorflow_output_ = tensorflow_output(tensorflow.keras.layers.ReLU(), self.input_data)
        custom_output_ = custom_output(ReLU(), self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-6)

    def test_relu_backward(self):
        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(torch.nn.ReLU(), self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(tensorflow.keras.layers.ReLU(), self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(ReLU(), self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-7, atol=1e-6)

    def test_selu_forward(self):
        # Compute the forward pass outputs:
        torch_output_ = torch_output(torch.nn.SELU(), self.input_data)
        tensorflow_output_ = tensorflow_output(tensorflow.keras.activations.selu, self.input_data)
        custom_output_ = custom_output(SELU(), self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-6)

    def test_selu_backward(self):
        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(torch.nn.SELU(), self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(tensorflow.keras.activations.selu, self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(SELU(), self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-7, atol=1e-6)

    def test_celu_forward(self):
        # Compute the forward pass outputs:
        torch_output_ = torch_output(torch.nn.CELU(), self.input_data)
        custom_output_ = custom_output(CELU(), self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6)

    def test_celu_backward(self):
        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(torch.nn.CELU(), self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(CELU(), self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-6)

    def test_gelu_forward(self):
        # Compute the forward pass outputs:
        torch_output_ = torch_output(torch.nn.GELU(), self.input_data)
        tensorflow_output_ = tensorflow_output(tensorflow.keras.activations.gelu, self.input_data)
        custom_output_ = custom_output(GELU(), self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-6)

    def test_gelu_backward(self):
        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(torch.nn.GELU(), self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(tensorflow.keras.activations.gelu, self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(GELU(), self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-7, atol=1e-6)

    def test_softplus_forward(self):
        # Compute the forward pass outputs:
        torch_output_ = torch_output(torch.nn.Softplus(), self.input_data)
        tensorflow_output_ = tensorflow_output(tensorflow.keras.activations.softplus, self.input_data)
        custom_output_ = custom_output(Softplus(), self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-6)

    def test_softplus_backward(self):
        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(torch.nn.Softplus(), self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(tensorflow.keras.activations.softplus, self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(Softplus(), self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-7, atol=1e-6)

    def test_mish_forward(self):
        # Compute the forward pass outputs:
        torch_output_ = torch_output(torch.nn.Mish(), self.input_data)
        tensorflow_output_ = tensorflow_output(tensorflow.keras.activations.mish, self.input_data)
        custom_output_ = custom_output(Mish(), self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-6)

    def test_mish_backward(self):
        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(torch.nn.Mish(), self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(tensorflow.keras.activations.mish, self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(Mish(), self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=5e-7, atol=5e-6)
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=5e-7, atol=5e-6)
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=5e-7, atol=5e-6)

    def test_tanh_forward(self):
        # Compute the forward pass outputs:
        torch_output_ = torch_output(torch.nn.Tanh(), self.input_data)
        tensorflow_output_ = tensorflow_output(tensorflow.keras.activations.tanh, self.input_data)
        custom_output_ = custom_output(Tanh(), self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-6)

    def test_tanh_backward(self):
        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(torch.nn.Tanh(), self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(tensorflow.keras.activations.tanh, self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(Tanh(), self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-7, atol=5e-6)
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-7, atol=5e-6)

    def test_sigmoid_forward(self):
        # Compute the forward pass outputs:
        torch_output_ = torch_output(torch.nn.Sigmoid(), self.input_data)
        tensorflow_output_ = tensorflow_output(tensorflow.keras.activations.sigmoid, self.input_data)
        custom_output_ = custom_output(Sigmoid(), self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-6)

    def test_sigmoid_backward(self):
        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(torch.nn.Sigmoid(), self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(tensorflow.keras.activations.sigmoid, self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(Sigmoid(), self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-7, atol=1e-6)

    def test_silu_forward(self):
        # Compute the forward pass outputs:
        torch_output_ = torch_output(torch.nn.SiLU(), self.input_data)
        tensorflow_output_ = tensorflow_output(tensorflow.nn.silu, self.input_data)
        custom_output_ = custom_output(SiLU(), self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-6)

    def test_silu_backward(self):
        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(torch.nn.SiLU(), self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(tensorflow.nn.silu, self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(SiLU(), self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-7, atol=1e-6)

    def test_swish_forward(self):
        # Compute the forward pass outputs:
        tensorflow_output_ = tensorflow_output(tensorflow.keras.activations.swish, self.input_data)
        custom_output_ = custom_output(Swish(), self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-6)

    def test_swish_backward(self):
        # Compute the retrograde gradients:
        tensorflow_grad_ = tensorflow_grad(tensorflow.keras.activations.swish, self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(Swish(), self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-7, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
