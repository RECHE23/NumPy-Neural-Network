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
        self.kernel_size = (3, 3)
        self.stride = (2, 2)
        self.padding = (1, 1)
        self.batch_size = 4
        self.input_height = 32
        self.input_width = 32
        self.input_data = np.random.randn(self.batch_size, self.input_channels, self.input_height, self.input_width)
        self.upstream_gradients = np.random.randn(self.batch_size, self.output_channels,
                                                  (self.input_height - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1,
                                                  (self.input_width - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1)
        self.torch_input = torch.tensor(self.input_data, dtype=torch.float32, requires_grad=True)
        self.torch_layer = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=self.kernel_size,
                                     stride=self.stride, padding=self.padding)
        self.custom_layer = Conv2d(self.input_channels, self.output_channels, kernel_size=self.kernel_size,
                                   stride=self.stride, padding=self.padding)

        # Initialize custom layer's weight and bias from torch_layer's parameters
        self.custom_layer.weight = self.torch_layer.weight.cpu().detach().numpy()
        self.custom_layer.bias = self.torch_layer.bias.cpu().detach().numpy()

    def test_forward(self):
        # Forward pass through both layers
        torch_output = self.torch_layer(self.torch_input).detach().numpy()
        custom_output = self.custom_layer(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=1e-7, atol=1e-6)

    def test_backward(self):
        # Compute gradients using backward for PyTorch layer
        torch_output = self.torch_layer(self.torch_input)
        torch_output.backward(torch.tensor(self.upstream_gradients, dtype=torch.float32))

        # Compute gradients using backward for custom layer
        custom_output = self.custom_layer(self.input_data)
        custom_retrograde = self.custom_layer.backward(self.upstream_gradients, None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = self.torch_input.grad.numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-7, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
