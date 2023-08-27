import unittest
import torch
from neural_network.layers.activation import *


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
        torch_activation = torch.nn.ELU()
        custom_activation = ELU()

        # Forward pass through both layers
        torch_output = torch_activation(self.torch_input).detach().numpy()
        custom_output = custom_activation(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=1e-7, atol=1e-6)

    def test_elu_backward(self):
        torch_activation = torch.nn.ELU()
        custom_activation = ELU()

        # Compute gradients using backward for PyTorch layer
        torch_output = torch_activation(self.torch_input)
        torch_output.backward(torch.tensor(self.upstream_gradients, dtype=torch.float32))

        # Compute gradients using backward for custom layer
        custom_output = custom_activation(self.input_data)
        custom_retrograde = custom_activation.backward(self.upstream_gradients, None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = self.torch_input.grad.cpu().detach().numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-7, atol=1e-6)

    def test_leakyrelu_forward(self):
        torch_activation = torch.nn.LeakyReLU()
        custom_activation = LeakyReLU()

        # Forward pass through both layers
        torch_output = torch_activation(self.torch_input).detach().numpy()
        custom_output = custom_activation(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=1e-7, atol=1e-7)

    def test_leakyrelu_backward(self):
        torch_activation = torch.nn.LeakyReLU()
        custom_activation = LeakyReLU()

        # Compute gradients using backward for PyTorch layer
        torch_output = torch_activation(self.torch_input)
        torch_output.backward(torch.tensor(self.upstream_gradients, dtype=torch.float32))

        # Compute gradients using backward for custom layer
        custom_output = custom_activation(self.input_data)
        custom_retrograde = custom_activation.backward(self.upstream_gradients, None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = self.torch_input.grad.cpu().detach().numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-7, atol=1e-7)

    def test_relu_forward(self):
        torch_activation = torch.nn.ReLU()
        custom_activation = ReLU()

        # Forward pass through both layers
        torch_output = torch_activation(self.torch_input).detach().numpy()
        custom_output = custom_activation(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=1e-7, atol=1e-7)

    def test_relu_backward(self):
        torch_activation = torch.nn.ReLU()
        custom_activation = ReLU()

        # Compute gradients using backward for PyTorch layer
        torch_output = torch_activation(self.torch_input)
        torch_output.backward(torch.tensor(self.upstream_gradients, dtype=torch.float32))

        # Compute gradients using backward for custom layer
        custom_output = custom_activation(self.input_data)
        custom_retrograde = custom_activation.backward(self.upstream_gradients, None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = self.torch_input.grad.cpu().detach().numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-7, atol=1e-7)

    def test_selu_forward(self):
        torch_activation = torch.nn.SELU()
        custom_activation = SELU()

        # Forward pass through both layers
        torch_output = torch_activation(self.torch_input).detach().numpy()
        custom_output = custom_activation(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=1e-7, atol=1e-6)

    def test_selu_backward(self):
        torch_activation = torch.nn.SELU()
        custom_activation = SELU()

        # Compute gradients using backward for PyTorch layer
        torch_output = torch_activation(self.torch_input)
        torch_output.backward(torch.tensor(self.upstream_gradients, dtype=torch.float32))

        # Compute gradients using backward for custom layer
        custom_output = custom_activation(self.input_data)
        custom_retrograde = custom_activation.backward(self.upstream_gradients, None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = self.torch_input.grad.cpu().detach().numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-7, atol=1e-6)

    def test_celu_forward(self):
        torch_activation = torch.nn.CELU()
        custom_activation = CELU()

        # Forward pass through both layers
        torch_output = torch_activation(self.torch_input).detach().numpy()
        custom_output = custom_activation(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=1e-7, atol=1e-6)

    def test_celu_backward(self):
        torch_activation = torch.nn.CELU()
        custom_activation = CELU()

        # Compute gradients using backward for PyTorch layer
        torch_output = torch_activation(self.torch_input)
        torch_output.backward(torch.tensor(self.upstream_gradients, dtype=torch.float32))

        # Compute gradients using backward for custom layer
        custom_output = custom_activation(self.input_data)
        custom_retrograde = custom_activation.backward(self.upstream_gradients, None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = self.torch_input.grad.cpu().detach().numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-7, atol=1e-6)

    def test_gelu_forward(self):
        torch_activation = torch.nn.GELU()
        custom_activation = GELU()

        # Forward pass through both layers
        torch_output = torch_activation(self.torch_input).detach().numpy()
        custom_output = custom_activation(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=1e-7, atol=1e-6)

    def test_gelu_backward(self):
        torch_activation = torch.nn.GELU()
        custom_activation = GELU()

        # Compute gradients using backward for PyTorch layer
        torch_output = torch_activation(self.torch_input)
        torch_output.backward(torch.tensor(self.upstream_gradients, dtype=torch.float32))

        # Compute gradients using backward for custom layer
        custom_output = custom_activation(self.input_data)
        custom_retrograde = custom_activation.backward(self.upstream_gradients, None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = self.torch_input.grad.cpu().detach().numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-7, atol=1e-6)

    def test_softplus_forward(self):
        torch_activation = torch.nn.Softplus()
        custom_activation = Softplus()

        # Forward pass through both layers
        torch_output = torch_activation(self.torch_input).detach().numpy()
        custom_output = custom_activation(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=1e-7, atol=1e-6)

    def test_softplus_backward(self):
        torch_activation = torch.nn.Softplus()
        custom_activation = Softplus()

        # Compute gradients using backward for PyTorch layer
        torch_output = torch_activation(self.torch_input)
        torch_output.backward(torch.tensor(self.upstream_gradients, dtype=torch.float32))

        # Compute gradients using backward for custom layer
        custom_output = custom_activation(self.input_data)
        custom_retrograde = custom_activation.backward(self.upstream_gradients, None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = self.torch_input.grad.cpu().detach().numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-7, atol=1e-6)

    def test_mish_forward(self):
        torch_activation = torch.nn.Mish()
        custom_activation = Mish()

        # Forward pass through both layers
        torch_output = torch_activation(self.torch_input).detach().numpy()
        custom_output = custom_activation(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=1e-7, atol=1e-6)

    def test_mish_backward(self):
        torch_activation = torch.nn.Mish()
        custom_activation = Mish()

        # Compute gradients using backward for PyTorch layer
        torch_output = torch_activation(self.torch_input)
        torch_output.backward(torch.tensor(self.upstream_gradients, dtype=torch.float32))

        # Compute gradients using backward for custom layer
        custom_output = custom_activation(self.input_data)
        custom_retrograde = custom_activation.backward(self.upstream_gradients, None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = self.torch_input.grad.cpu().detach().numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-7, atol=1e-5)

    def test_tanh_forward(self):
        torch_activation = torch.nn.Tanh()
        custom_activation = Tanh()

        # Forward pass through both layers
        torch_output = torch_activation(self.torch_input).detach().numpy()
        custom_output = custom_activation(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=1e-7, atol=1e-7)

    def test_tanh_backward(self):
        torch_activation = torch.nn.Tanh()
        custom_activation = Tanh()

        # Compute gradients using backward for PyTorch layer
        torch_output = torch_activation(self.torch_input)
        torch_output.backward(torch.tensor(self.upstream_gradients, dtype=torch.float32))

        # Compute gradients using backward for custom layer
        custom_output = custom_activation(self.input_data)
        custom_retrograde = custom_activation.backward(self.upstream_gradients, None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = self.torch_input.grad.cpu().detach().numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-7, atol=1e-6)

    def test_sigmoid_forward(self):
        torch_activation = torch.nn.Sigmoid()
        custom_activation = Sigmoid()

        # Forward pass through both layers
        torch_output = torch_activation(self.torch_input).detach().numpy()
        custom_output = custom_activation(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=1e-7, atol=1e-7)

    def test_sigmoid_backward(self):
        torch_activation = torch.nn.Sigmoid()
        custom_activation = Sigmoid()

        # Compute gradients using backward for PyTorch layer
        torch_output = torch_activation(self.torch_input)
        torch_output.backward(torch.tensor(self.upstream_gradients, dtype=torch.float32))

        # Compute gradients using backward for custom layer
        custom_output = custom_activation(self.input_data)
        custom_retrograde = custom_activation.backward(self.upstream_gradients, None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = self.torch_input.grad.cpu().detach().numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-7, atol=1e-6)

    def test_silu_forward(self):
        torch_activation = torch.nn.SiLU()
        custom_activation = SiLU()

        # Forward pass through both layers
        torch_output = torch_activation(self.torch_input).detach().numpy()
        custom_output = custom_activation(self.input_data)

        # Compare the forward pass outputs
        np.testing.assert_allclose(torch_output, custom_output, rtol=1e-7, atol=1e-6)

    def test_silu_backward(self):
        torch_activation = torch.nn.SiLU()
        custom_activation = SiLU()

        # Compute gradients using backward for PyTorch layer
        torch_output = torch_activation(self.torch_input)
        torch_output.backward(torch.tensor(self.upstream_gradients, dtype=torch.float32))

        # Compute gradients using backward for custom layer
        custom_output = custom_activation(self.input_data)
        custom_retrograde = custom_activation.backward(self.upstream_gradients, None)

        # Retrieve gradients from the layer's input (retrograde) for both implementations
        torch_retrograde = self.torch_input.grad.cpu().detach().numpy()

        # Compare the retrograde gradients
        np.testing.assert_allclose(torch_retrograde, custom_retrograde, rtol=1e-7, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
