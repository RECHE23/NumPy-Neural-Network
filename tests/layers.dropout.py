import unittest
import numpy as np
from neural_network.layers.dropout import Dropout


class TestDropout(unittest.TestCase):

    def test_forward_propagation_during_training(self):
        """
        Test that the forward propagation during training introduces dropout.
        """
        dropout_layer = Dropout(p=0.5)
        input_data = np.ones((10, 10))  # Example input data
        dropout_layer.is_training(True)
        output = dropout_layer(input_data)
        # Verify that the output contains only 0 and 2, assuming scaling factor is 2 due to p=0.5
        self.assertTrue(set(np.unique(output)) == {0., 2.})

    def test_forward_propagation_during_inference(self):
        """
        Test that the forward propagation during inference does not introduce dropout.
        """
        dropout_layer = Dropout(p=0.5)
        input_data = np.ones((10, 10))  # Example input data
        dropout_layer.is_training(False)
        output = dropout_layer(input_data)
        # Verify that the output remains unchanged during inference
        self.assertTrue(np.all(output == input_data))

    def test_backward_propagation_during_training(self):
        """
        Test that the backward propagation during training respects dropout.
        """
        dropout_layer = Dropout(p=0.5)
        input_data = np.ones((10, 10))  # Example input data
        dropout_layer.is_training(True)
        dropout_layer(input_data)  # Forward pass
        upstream_gradients = np.ones_like(input_data)
        retrograde_gradients = dropout_layer.backward(upstream_gradients)
        # Verify that retrograde gradients contain only 0 and 2, assuming scaling factor is 2 due to p=0.5
        self.assertTrue(set(np.unique(retrograde_gradients)) == {0., 2.})

    def test_backward_propagation_during_inference(self):
        """
        Test that the backward propagation during inference does not modify gradients.
        """
        dropout_layer = Dropout(p=0.5)
        input_data = np.ones((10, 10))  # Example input data
        dropout_layer.is_training(True)
        dropout_layer(input_data)  # Forward pass in training
        dropout_layer.is_training(False)
        dropout_layer(input_data)  # Forward pass in inference
        upstream_gradients = np.ones_like(input_data)
        retrograde_gradients = dropout_layer.backward(upstream_gradients)
        # Verify that retrograde gradients remain unchanged during inference
        self.assertTrue(np.all(retrograde_gradients == upstream_gradients))


if __name__ == '__main__':
    unittest.main()
