from .utils import *
from neural_network.modules.pooling2d import GlobalMaxPool2d


class TestMaxPool2dLayer(unittest.TestCase):

    def setUp(self):
        # Set up common parameters and data for the tests
        self.input_channels = 3
        self.batch_size = 4
        self.input_height = 32
        self.input_width = 32
        self.input_data = np.random.randn(self.batch_size, self.input_channels, self.input_height, self.input_width)

    def test_forward_backward(self):
        # Set up the layers with specific kernel_size and stride:
        self.torch_layer = lambda x: torch.nn.AdaptiveMaxPool2d((1, 1))(x).squeeze()
        self.tensorflow_layer = tensorflow.keras.layers.GlobalMaxPooling2D()
        self.tensorflow_layer(to_tensorflow(self.input_data))  # Tensorflow layer needs a forward pass to initialize...
        self.custom_layer = GlobalMaxPool2d()
        self.custom_layer.is_training(True)

        # Compute the forward pass outputs:
        torch_output_ = torch_output(self.torch_layer, self.input_data)
        tensorflow_output_ = tensorflow_output(self.tensorflow_layer, self.input_data)
        custom_output_ = custom_output(self.custom_layer, self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-6)

        self.upstream_gradients = np.random.randn(*custom_output_.shape)

        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(self.torch_layer, self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(self.tensorflow_layer, self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(self.custom_layer, self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-7, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
