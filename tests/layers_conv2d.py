from .utils import *
from neural_network.modules.conv2d import Conv2d


class TestConv2dLayer(unittest.TestCase):

    def setUp(self):
        # Set up common parameters and data for the tests
        self.input_channels = 3
        self.output_channels = 8
        self.batch_size = 4
        self.input_height = 32
        self.input_width = 32
        self.input_data = np.random.randn(self.batch_size, self.input_channels, self.input_height, self.input_width)

    def _test_forward_backward(self, kernel_size, stride):
        # Set up the layers with specific kernel_size, stride, and padding:
        self.torch_layer = torch.nn.Conv2d(self.input_channels, self.output_channels, kernel_size=kernel_size, stride=stride)
        input_shape = (self.batch_size, self.input_channels, self.input_height, self.input_width)
        self.tensorflow_layer = tensorflow.keras.layers.Conv2D(self.output_channels, input_shape=input_shape, use_bias=True, activation=None,
                                                               kernel_size=kernel_size, strides=stride)
        self.tensorflow_layer(to_tensorflow(self.input_data))  # Tensorflow layer needs a forward pass to initialize...
        self.custom_layer = Conv2d(self.input_channels, self.output_channels, kernel_size=kernel_size, stride=stride)

        # Initialize custom layer's weight and bias from torch_layer's parameters:
        self.custom_layer.weight = to_numpy(self.torch_layer.weight)
        self.custom_layer.bias = to_numpy(self.torch_layer.bias)

        # Initialize Tensorflow layer's weight and bias from torch_layer's parameters:
        self.tensorflow_layer.kernel = np.moveaxis(self.custom_layer.weight, (0, 1, 2, 3), (3, 2, 0, 1))
        self.tensorflow_layer.bias = self.custom_layer.bias

        # Compute the forward pass outputs:
        torch_output_ = torch_output(self.torch_layer, self.input_data)
        tensorflow_output_ = tensorflow_output(self.tensorflow_layer, self.input_data)
        custom_output_ = custom_output(self.custom_layer, self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-7, atol=1e-6,
                                   err_msg=f"Failed for kernel_size={kernel_size}, stride={stride}")
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6,
                                   err_msg=f"Failed for kernel_size={kernel_size}, stride={stride}")
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-7, atol=1e-6,
                                   err_msg=f"Failed for kernel_size={kernel_size}, stride={stride}")

        self.upstream_gradients = np.random.randn(*custom_output_.shape)

        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(self.torch_layer, self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(self.tensorflow_layer, self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(self.custom_layer, self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-6, atol=1e-5,
                                   err_msg=f"Failed for kernel_size={kernel_size}, stride={stride}")
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-6, atol=1e-5,
                                   err_msg=f"Failed for kernel_size={kernel_size}, stride={stride}")
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-6, atol=1e-5,
                                   err_msg=f"Failed for kernel_size={kernel_size}, stride={stride}")

    def test_different_kernel_stride_padding(self):
        # Test different combinations of kernel_size, stride, and padding
        kernel_stride_combinations = [
            ((3, 3), (1, 1)),
            ((2, 2), (2, 2)),
            ((4, 5), (1, 1)),
            ((4, 5), (2, 1)),
            ((4, 4), (2, 2))
        ]

        for kernel_size, stride in kernel_stride_combinations:
            with self.subTest(kernel_size=kernel_size, stride=stride):
                self._test_forward_backward(kernel_size, stride)


if __name__ == '__main__':
    unittest.main()

# TODO: Investigate why the tests fail for these values with PyTorch:
#       - kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)
#       - kernel_size=(5, 5), stride=(1, 2), padding=(2, 1)
#       - kernel_size=(5, 5), stride=(1, 2), padding=(0, 0)
#       - kernel_size=(3, 4), stride=(2, 3), padding=(0, 0)
