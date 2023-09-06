from .utils import *
from neural_network.modules.pooling2d import MaxPool2d


class TestMaxPool2dLayer(unittest.TestCase):

    def setUp(self):
        # Set up common parameters and data for the tests
        self.input_channels = 3
        self.output_channels = 8
        self.batch_size = 4
        self.input_height = 32
        self.input_width = 32
        self.input_data = np.random.randn(self.batch_size, self.input_channels, self.input_height, self.input_width)

    def _test_forward_backward(self, kernel_size, stride):
        # Set up the layers with specific kernel_size and stride:
        self.torch_layer = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0, dilation=1)
        self.tensorflow_layer = tensorflow.keras.layers.MaxPooling2D(pool_size=kernel_size, strides=stride, padding='valid')
        self.tensorflow_layer(to_tensorflow(self.input_data))  # Tensorflow layer needs a forward pass to initialize...
        self.custom_layer = MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.custom_layer.is_training(True)

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
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-7, atol=1e-6,
                                   err_msg=f"Failed for kernel_size={kernel_size}, stride={stride}")
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-6,
                                   err_msg=f"Failed for kernel_size={kernel_size}, stride={stride}")
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-7, atol=1e-6,
                                   err_msg=f"Failed for kernel_size={kernel_size}, stride={stride}")

    def test_different_kernel_stride(self):
        # Test different combinations of kernel_size and stride
        kernel_stride_combinations = [
            ((1, 1), (1, 1)),
            ((2, 2), (2, 2)),
            ((3, 3), (3, 3)),
            ((3, 3), (1, 1)),
            ((4, 5), (1, 1)),
            ((4, 5), (2, 1)),
            ((4, 4), (2, 2)),
            ((4, 4), (2, 3)),
            ((2, 4), (2, 2))
        ]

        for kernel_size, stride in kernel_stride_combinations:
            with self.subTest(kernel_size=kernel_size, stride=stride):
                self._test_forward_backward(kernel_size, stride)


if __name__ == '__main__':
    unittest.main()