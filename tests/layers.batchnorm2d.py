import unittest
import numpy as np
import torch
import torch.nn as nn
from neural_network.layers.batchnorm2d import BatchNorm2d


class TestBatchNorm2dLayer(unittest.TestCase):
    def setUp(self):
        # Set up common parameters and data for the tests
        self.input_channels = 3
        self.batch_size = 8
        self.input_height = 32
        self.input_width = 32
        self.eps = 1e-08
        self.momentum = 0.1
        self.affine = True
        self.input_data = np.random.randn(self.batch_size, self.input_channels, self.input_height, self.input_width)
        self.upstream_gradients = np.random.randn(self.batch_size, self.input_channels, self.input_height, self.input_width)
        self.torch_input = torch.tensor(self.input_data, dtype=torch.float32, requires_grad=True)
        self.torch_layer = nn.BatchNorm2d(self.input_channels, eps=self.eps, momentum=self.momentum, affine=self.affine)
        self.custom_layer = BatchNorm2d(self.input_channels, eps=self.eps, momentum=self.momentum, affine=self.affine)
        print(self.torch_layer)
        print(self.custom_layer)

    def test_forward(self):
        # Forward pass through both layers
        torch_output = self.torch_layer(self.torch_input).cpu().detach().numpy()
        custom_output = self.custom_layer(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=0.025, atol=0.025)  # TODO: To improve...

    def test_backward(self):
        # Compute gradients using backward for PyTorch layer
        torch_output = self.torch_layer(self.torch_input)
        torch_output.backward(torch.tensor(self.upstream_gradients, dtype=torch.float32))

        # Compute gradients using backward for custom layer
        custom_output = self.custom_layer(self.input_data)
        custom_retrograde = self.custom_layer.backward(self.upstream_gradients, None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = self.torch_input.grad.cpu().detach().numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=0.025, atol=0.025)  # TODO: To improve...


if __name__ == '__main__':
    unittest.main()

# TODO: Add tests for eps, momentum and affine parameter and improve admissible tolerance...
