from .utils import *
from neural_network.modules.linear import Linear


class TestLinearLayer(unittest.TestCase):

    def setUp(self):
        # Set up common parameters and data for the tests:
        self.input_size = 10
        self.output_size = 5
        self.batch_size = 8

        # Generate random input data:
        self.input_data = np.random.randn(self.batch_size, self.input_size)

        # Generate random upstream gradients:
        self.upstream_gradients = np.random.randn(self.batch_size, self.output_size)

        # Create instances of PyTorch Linear layer, Tensorflow Dense layer and custom Linear layer:
        self.torch_layer = torch.nn.Linear(self.input_size, self.output_size)
        self.tensorflow_layer = tensorflow.keras.layers.Dense(units=self.output_size, input_shape=(self.input_size, ), use_bias=True, activation=None)
        self.tensorflow_layer(to_tensorflow(self.input_data))  # Tensorflow layer needs a forward pass to initialize...
        self.custom_layer = Linear(self.input_size, self.output_size)

        # Initialize custom layer's weight and bias from torch_layer's parameters:
        self.custom_layer.weight = to_numpy(self.torch_layer.weight)
        self.custom_layer.bias = to_numpy(self.torch_layer.bias)

        # Initialize Tensorflow layer's weight and bias from torch_layer's parameters:
        self.tensorflow_layer.kernel = self.custom_layer.weight.T
        self.tensorflow_layer.bias = self.custom_layer.bias

    def test_forward(self):
        # Compute the forward pass outputs:
        torch_output_ = torch_output(self.torch_layer, self.input_data)
        tensorflow_output_ = tensorflow_output(self.tensorflow_layer, self.input_data)
        custom_output_ = custom_output(self.custom_layer, self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-7)

    def test_backward(self):
        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(self.torch_layer, self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(self.tensorflow_layer, self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(self.custom_layer, self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
