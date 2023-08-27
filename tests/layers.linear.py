import unittest
import numpy as np
import torch
import torch.nn as nn
from neural_network.layers.linear import Linear


class TestLinearLayer(unittest.TestCase):
    def setUp(self):
        # Set up common parameters and data for the tests
        self.input_size = 10
        self.output_size = 5
        self.batch_size = 8

        # Generate random input data
        self.input_data = np.random.randn(self.batch_size, self.input_size)

        # Create a PyTorch tensor with gradient tracking enabled
        self.torch_input = torch.tensor(self.input_data, dtype=torch.float32, requires_grad=True)

        # Generate random upstream gradients
        self.upstream_gradients = np.random.randn(self.batch_size, self.output_size)

        # Create instances of PyTorch Linear layer and custom Linear layer
        self.torch_layer = nn.Linear(self.input_size, self.output_size)
        self.custom_layer = Linear(self.input_size, self.output_size)

        # Initialize custom layer's weight and bias from torch_layer's parameters
        self.custom_layer.weight = self.torch_layer.weight.cpu().detach().numpy()
        self.custom_layer.bias = self.torch_layer.bias.cpu().detach().numpy()

    def test_forward(self):
        # Forward pass through both layers
        torch_output = self.torch_layer(self.torch_input).detach().numpy()
        custom_output = self.custom_layer(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=1e-7, atol=1e-7)

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
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
