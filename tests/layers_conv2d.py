import unittest
import numpy as np
import torch
import torch.nn as nn
from neural_network.layers.conv2d import Conv2d


class TestConv2dLayer(unittest.TestCase):

    def setUp(self):
        # Set up common parameters and data for the tests
        self.input_channels = 3
        self.output_channels = 8
        self.batch_size = 4
        self.input_height = 32
        self.input_width = 32
        self.input_data = np.random.randn(self.batch_size, self.input_channels, self.input_height, self.input_width)

    def _test_forward_backward(self, kernel_size, stride, padding):
        # Set up the layers with specific kernel_size, stride, and padding

        self.torch_layer = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        self.custom_layer = Conv2d(self.input_channels, self.output_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding)

        # Initialize custom layer's weight and bias from torch_layer's parameters
        self.custom_layer.weight = self.torch_layer.weight.cpu().detach().numpy()
        self.custom_layer.bias = self.torch_layer.bias.cpu().detach().numpy()

        # Forward pass through both layers
        torch_input = torch.tensor(self.input_data, dtype=torch.float32, requires_grad=True)
        torch_output = self.torch_layer(torch_input)
        custom_output = self.custom_layer(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output.cpu().detach().numpy(), custom_output, rtol=1e-6, atol=1e-6,
                                   err_msg=f"Failed for kernel_size={kernel_size}, stride={stride}, padding={padding}")

        # Compute gradients using backward for PyTorch layer
        torch_output.backward(torch_output.data)

        # Compute gradients using backward for custom layer
        custom_retrograde = self.custom_layer.backward(torch_output.data.cpu().detach().numpy(), None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = torch_input.grad.cpu().detach().numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-6, atol=1e-6,
                                   err_msg=f"Backward Failed for kernel_size={kernel_size}, stride={stride}, padding={padding}")

    def test_different_kernel_stride_padding(self):
        # Test different combinations of kernel_size, stride, and padding
        kernel_stride_padding_combinations = [
            ((3, 3), (1, 1), (0, 0)),
            ((2, 2), (2, 2), (1, 1)),
            ((5, 5), (1, 2), (2, 1))
        ]

        for kernel_size, stride, padding in kernel_stride_padding_combinations:
            with self.subTest(kernel_size=kernel_size, stride=stride, padding=padding):
                self._test_forward_backward(kernel_size, stride, padding)


if __name__ == '__main__':
    unittest.main()

# TODO: Investigate why the tests fail for these values:
#       - kernel_size=(2, 2), stride=(2, 2), padding=(1, 1)
#       - kernel_size=(5, 5), stride=(1, 2), padding=(2, 1)
