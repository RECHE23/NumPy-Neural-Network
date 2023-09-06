from .utils import *
from neural_network.modules.conv1d import Conv1d


class TestConv1dLayer(unittest.TestCase):

    def setUp(self):
        # Set up common parameters and data for the tests
        self.input_channels = 3
        self.output_channels = 8
        self.batch_size = 10
        self.input_length = 64
        self.input_data = np.random.randn(self.batch_size, self.input_channels, self.input_length)

    def _test_forward_backward(self, kernel_size, stride):
        # Set up the layers with specific kernel_size and stride:
        self.torch_layer = torch.nn.Conv1d(self.input_channels, self.output_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.custom_layer = Conv1d(self.input_channels, self.output_channels, kernel_size=kernel_size, stride=stride, padding=0)

        # Initialize custom layer's weight and bias from torch_layer's parameters:
        self.custom_layer.weight = to_numpy(self.torch_layer.weight)
        self.custom_layer.bias = to_numpy(self.torch_layer.bias)

        # Compute the forward pass outputs:
        torch_output_ = torch_output(self.torch_layer, self.input_data)
        custom_output_ = custom_output(self.custom_layer, self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-7, atol=1e-6,
                                   err_msg=f"Failed for kernel_size={kernel_size}, stride={stride}")

        self.upstream_gradients = np.random.randn(*custom_output_.shape)

        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(self.torch_layer, self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(self.custom_layer, self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-7, atol=1e-6,
                                   err_msg=f"Failed for kernel_size={kernel_size}, stride={stride}")

    def test_different_kernel_stride(self):
        # Test different combinations of kernel_size and stride
        kernel_stride_combinations = [
            (1, 1),
            (2, 1),
            (3, 1),
            (2, 2),
            (4, 1),
            (4, 2)
        ]

        for kernel_size, stride in kernel_stride_combinations:
            with self.subTest(kernel_size=kernel_size, stride=stride):
                self._test_forward_backward(kernel_size, stride)


if __name__ == '__main__':
    unittest.main()
