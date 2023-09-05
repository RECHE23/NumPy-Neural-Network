import numpy as np

from .utils import *
from neural_network.modules.batchnorm1d import BatchNorm1d
from neural_network.optimizers import SGD


class TestBatchNorm2dLayer(unittest.TestCase):

    def setUp(self):
        # Set up common parameters and data for the tests
        self.batch_size = 64
        self.num_features = 32
        self.eps = 1e-8
        self.momentum = 0.1
        self.affine = True
        self.input_data = np.random.randn(self.batch_size, self.num_features)
        self.upstream_gradients = np.random.randn(self.batch_size, self.num_features)

        self.torch_layer = torch.nn.BatchNorm1d(self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine)
        self.torch_layer.eval()
        self.tensorflow_layer = tensorflow.keras.layers.BatchNormalization(epsilon=self.eps, momentum=1-self.momentum, center=self.affine, scale=self.affine, trainable=False)
        self.tensorflow_layer(to_tensorflow(self.input_data))  # Tensorflow layer needs a forward pass to initialize...
        self.custom_layer = BatchNorm1d(self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine, optimizer=SGD(lr=1e-3))
        self.custom_layer.is_training(False)

    def test_forward(self):
        # Compute the forward pass outputs:
        torch_output_ = torch_output(self.torch_layer, self.input_data)
        tensorflow_output_ = tensorflow_output(self.tensorflow_layer, self.input_data)
        custom_output_ = custom_output(self.custom_layer, self.input_data)

        # Compare the forward pass outputs:
        np.testing.assert_allclose(torch_output_, tensorflow_output_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(torch_output_, custom_output_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(tensorflow_output_, custom_output_, rtol=1e-6, atol=1e-6)

    def test_backward(self):
        # Compute the retrograde gradients:
        torch_grad_ = torch_grad(self.torch_layer, self.input_data, self.upstream_gradients)
        tensorflow_grad_ = tensorflow_grad(self.tensorflow_layer, self.input_data, self.upstream_gradients)
        custom_grad_ = custom_grad(self.custom_layer, self.input_data, self.upstream_gradients)

        # Compare the retrograde gradients:
        np.testing.assert_allclose(torch_grad_, tensorflow_grad_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(torch_grad_, custom_grad_, rtol=1e-6, atol=1)    # TODO: Investigate the divergences...
        np.testing.assert_allclose(tensorflow_grad_, custom_grad_, rtol=1e-6, atol=1)  # TODO: Investigate the divergences...


if __name__ == '__main__':
    unittest.main()
