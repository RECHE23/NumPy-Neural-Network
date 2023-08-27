import unittest
import numpy as np
from neural_network.layers.pooling2d import MaxPool2d, AvgPool2d
import torch
import torch.nn.functional as F


class TestPoolingLayers(unittest.TestCase):

    def test_max_pool_forward(self):
        np.random.seed(42)
        torch.manual_seed(42)
        input_data = np.random.rand(2, 3, 4, 4).astype(np.float32)
        torch_input = torch.from_numpy(input_data)
        pytorch_maxpool = F.max_pool2d(torch_input, kernel_size=2, stride=2)

        maxpool_layer = MaxPool2d(kernel_size=2, stride=2)
        maxpool_layer.is_training(False)
        output = maxpool_layer(input_data)

        self.assertTrue(np.allclose(pytorch_maxpool.numpy(), output, atol=1e-6))

    def test_max_pool_backward(self):
        np.random.seed(42)
        torch.manual_seed(42)
        input_data = np.random.rand(2, 3, 4, 4).astype(np.float32)
        torch_input = torch.from_numpy(input_data)

        # Create random gradients for backward pass
        upstream_gradients = np.random.rand(2, 3, 2, 2).astype(np.float32)
        torch_gradients = torch.from_numpy(upstream_gradients)

        # Perform forward and backward pass using PyTorch
        torch_input.requires_grad = True
        torch_output = F.max_pool2d(torch_input, kernel_size=2, stride=2)
        torch_output.backward(torch_gradients)
        torch_backward_gradients = torch_input.grad.numpy()

        maxpool_layer = MaxPool2d(kernel_size=2, stride=2)
        maxpool_layer.is_training(True)
        output = maxpool_layer(input_data)
        retrograde_gradients = maxpool_layer.backward(upstream_gradients)

        self.assertTrue(np.allclose(torch_backward_gradients, retrograde_gradients, atol=1e-6))

    def test_avg_pool_forward(self):
        np.random.seed(42)
        torch.manual_seed(42)
        input_data = np.random.rand(2, 3, 4, 4).astype(np.float32)
        torch_input = torch.from_numpy(input_data)
        pytorch_avgpool = F.avg_pool2d(torch_input, kernel_size=2, stride=2)

        avgpool_layer = AvgPool2d(kernel_size=2, stride=2)
        avgpool_layer.is_training(False)
        output = avgpool_layer(input_data)

        self.assertTrue(np.allclose(pytorch_avgpool.numpy(), output, atol=1e-6))

    def test_avg_pool_backward(self):
        np.random.seed(42)
        torch.manual_seed(42)
        input_data = np.random.rand(2, 3, 4, 4).astype(np.float32)
        torch_input = torch.from_numpy(input_data)

        # Create random gradients for backward pass
        upstream_gradients = np.random.rand(2, 3, 2, 2).astype(np.float32)
        torch_gradients = torch.from_numpy(upstream_gradients)

        # Perform forward and backward pass using PyTorch
        torch_input.requires_grad = True
        torch_output = F.avg_pool2d(torch_input, kernel_size=2, stride=2)
        torch_output.backward(torch_gradients)
        torch_backward_gradients = torch_input.grad.numpy()

        avgpool_layer = AvgPool2d(kernel_size=2, stride=2)
        avgpool_layer.is_training(True)
        output = avgpool_layer(input_data)
        retrograde_gradients = avgpool_layer.backward(upstream_gradients)

        self.assertTrue(np.allclose(torch_backward_gradients, retrograde_gradients, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
