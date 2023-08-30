import unittest
import numpy as np
from neural_network.modules.shape_manipulation import Reshape, Flatten, Unflatten


class TestShapeManipulationLayers(unittest.TestCase):

    def test_reshape_forward_backward(self):
        input_data = np.random.rand(2, 3, 4, 4)
        output_shape = (1, 12, 4)  # Batch size remains the same, other dimensions are reshaped

        reshape_layer = Reshape(output_shape=output_shape)
        output = reshape_layer(input_data)
        self.assertEqual(output.shape, (2, *output_shape))

        upstream_gradients = np.random.rand(2, *output_shape)
        retrograde_gradients = reshape_layer.backward(upstream_gradients)
        self.assertEqual(retrograde_gradients.shape, input_data.shape)

    def test_flatten_forward_backward(self):
        input_data = np.random.rand(2, 3, 4, 4)
        output_shape = (2, 48)  # Batch size remains the same, all other dimensions are flattened

        flatten_layer = Flatten()
        output = flatten_layer(input_data)
        self.assertEqual(output.shape, output_shape)

        upstream_gradients = np.random.rand(*output_shape)
        retrograde_gradients = flatten_layer.backward(upstream_gradients)
        self.assertEqual(retrograde_gradients.shape, input_data.shape)

    def test_unflatten_forward_backward(self):
        input_data = np.random.rand(2, 48)  # Batch size remains the same, all other dimensions are flattened
        dim = 1
        unflattened_size = (3, 4, 4)  # Unflatten along dim 1 into (3, 4, 4)

        unflatten_layer = Unflatten(dim=dim, unflattened_size=unflattened_size)
        output = unflatten_layer(input_data)
        self.assertEqual(output.shape, (2, *unflattened_size))

        upstream_gradients = np.random.rand(2, *unflattened_size)
        retrograde_gradients = unflatten_layer.backward(upstream_gradients)
        self.assertEqual(retrograde_gradients.shape, input_data.shape)


if __name__ == '__main__':
    unittest.main()
