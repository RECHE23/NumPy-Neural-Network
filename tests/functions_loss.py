import unittest
from functools import partial

import tensorflow.python.keras.losses
import torch.nn

from .utils import *
from neural_network.functions.loss import *


class TestAgainstFrameworks(unittest.TestCase):

    def setUp(self):
        self.N = 157
        self.K = 13

    def test_binary_cross_entropy_raw(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=True)
        custom_loss_ = binary_cross_entropy(y_true, y_pred, multioutput='raw_values')
        torch_loss_ = np.sum(torch_loss(torch.nn.BCELoss(reduction='none'), y_true, y_pred), axis=-1) / self.K
        tensorflow_loss_ = tensorflow_loss(tensorflow.keras.losses.BinaryCrossentropy(reduction=tensorflow.keras.losses.Reduction.NONE), y_true, y_pred)
        np.testing.assert_allclose(torch_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(custom_loss_, torch_loss_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(custom_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)

    def test_binary_cross_entropy_sum(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=True)
        custom_loss_ = binary_cross_entropy(y_true, y_pred, multioutput='sum')
        torch_loss_ = torch_loss(torch.nn.BCELoss(reduction='sum'), y_true, y_pred) / self.K
        tensorflow_loss_ = tensorflow_loss(tensorflow.keras.losses.BinaryCrossentropy(reduction=tensorflow.keras.losses.Reduction.SUM), y_true, y_pred)
        np.testing.assert_allclose(torch_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(custom_loss_, torch_loss_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(custom_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)

    def test_categorical_cross_entropy_raw(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=False)
        custom_loss_ = categorical_cross_entropy(y_true, softmax(y_pred), multioutput='raw_values')
        torch_loss_ = torch_loss(torch.nn.CrossEntropyLoss(reduction='none'), y_true, y_pred)
        tensorflow_loss_ = tensorflow_loss(tensorflow.keras.losses.CategoricalCrossentropy(reduction=tensorflow.keras.losses.Reduction.NONE, from_logits=True), y_true, y_pred)
        np.testing.assert_allclose(torch_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(custom_loss_, torch_loss_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(custom_loss_, tensorflow_loss_, rtol=1e-6, atol=1e-6)

    def test_categorical_cross_entropy_sum(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=False)
        custom_loss_ = categorical_cross_entropy(y_true, softmax(y_pred), multioutput='sum')
        torch_loss_ = torch_loss(torch.nn.CrossEntropyLoss(reduction='sum'), y_true, y_pred)
        tensorflow_loss_ = tensorflow_loss(tensorflow.keras.losses.CategoricalCrossentropy(reduction=tensorflow.keras.losses.Reduction.SUM, from_logits=True), y_true, y_pred)
        np.testing.assert_allclose(torch_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(custom_loss_, torch_loss_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(custom_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)

    def test_mean_absolute_error_raw(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=False)
        custom_loss_ = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
        torch_loss_ = np.sum(torch_loss(torch.nn.L1Loss(reduction='none'), y_true, y_pred), axis=-1) / self.K
        tensorflow_loss_ = tensorflow_loss(tensorflow.keras.losses.MeanAbsoluteError(reduction=tensorflow.keras.losses.Reduction.NONE), y_true, y_pred)
        np.testing.assert_allclose(torch_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(custom_loss_, torch_loss_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(custom_loss_, tensorflow_loss_, rtol=1e-6, atol=1e-6)

    def test_mean_absolute_error_sum(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=False)
        custom_loss_ = mean_absolute_error(y_true, y_pred, multioutput='sum')
        torch_loss_ = torch_loss(torch.nn.L1Loss(reduction='sum'), y_true, y_pred) / self.K
        tensorflow_loss_ = tensorflow_loss(tensorflow.keras.losses.MeanAbsoluteError(reduction=tensorflow.keras.losses.Reduction.SUM), y_true, y_pred)
        np.testing.assert_allclose(torch_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(custom_loss_, torch_loss_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(custom_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)

    def test_mean_squared_error_raw(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=False)
        custom_loss_ = mean_squared_error(y_true, y_pred, multioutput='raw_values')
        torch_loss_ = np.sum(torch_loss(torch.nn.MSELoss(reduction='none'), y_true, y_pred), axis=-1) / self.K
        tensorflow_loss_ = tensorflow_loss(tensorflow.keras.losses.MeanSquaredError(reduction=tensorflow.keras.losses.Reduction.NONE), y_true, y_pred)
        np.testing.assert_allclose(torch_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(custom_loss_, torch_loss_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(custom_loss_, tensorflow_loss_, rtol=1e-6, atol=1e-6)

    def test_mean_squared_error_sum(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=False)
        custom_loss_ = mean_squared_error(y_true, y_pred, multioutput='sum')
        torch_loss_ = torch_loss(torch.nn.MSELoss(reduction='sum'), y_true, y_pred) / self.K
        tensorflow_loss_ = tensorflow_loss(tensorflow.keras.losses.MeanSquaredError(reduction=tensorflow.keras.losses.Reduction.SUM), y_true, y_pred)
        np.testing.assert_allclose(torch_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(custom_loss_, torch_loss_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(custom_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)

    def test_huber_loss_raw(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=False)
        custom_loss_ = huber_loss(y_true, y_pred, multioutput='raw_values')
        torch_loss_ = np.sum(torch_loss(torch.nn.HuberLoss(reduction='none'), y_true, y_pred), axis=-1) / self.K
        tensorflow_loss_ = tensorflow_loss(tensorflow.keras.losses.Huber(reduction=tensorflow.keras.losses.Reduction.NONE), y_true, y_pred)
        np.testing.assert_allclose(torch_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(custom_loss_, torch_loss_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(custom_loss_, tensorflow_loss_, rtol=1e-6, atol=1e-6)

    def test_huber_loss_sum(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=False)
        custom_loss_ = huber_loss(y_true, y_pred, multioutput='sum')
        torch_loss_ = torch_loss(torch.nn.HuberLoss(reduction='sum'), y_true, y_pred) / self.K
        tensorflow_loss_ = tensorflow_loss(tensorflow.keras.losses.Huber(reduction=tensorflow.keras.losses.Reduction.SUM), y_true, y_pred)
        np.testing.assert_allclose(torch_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(custom_loss_, torch_loss_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(custom_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)

    def test_hinge_loss_raw(self):
        y_true = make_random_targets(self.N, self.K)
        y_true[y_true == 0] = -1
        y_pred = make_random_predictions(self.N, self.K, probability=False)
        custom_loss_ = hinge_loss(y_true, y_pred, multioutput='raw_values')
        #torch_loss_ = np.sum(torch_loss(torch.nn.HingeEmbeddingLoss(reduction='none'), y_true, y_pred), axis=-1) / self.K
        tensorflow_loss_ = tensorflow_loss(tensorflow.keras.losses.Hinge(reduction=tensorflow.keras.losses.Reduction.NONE), y_true, y_pred)
        #np.testing.assert_allclose(torch_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)
        #np.testing.assert_allclose(custom_loss_, torch_loss_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(custom_loss_, tensorflow_loss_, rtol=1e-6, atol=1e-6)

    def test_hinge_loss_sum(self):
        y_true = make_random_targets(self.N, self.K)
        y_true[y_true == 0] = -1
        y_pred = make_random_predictions(self.N, self.K, probability=False)
        custom_loss_ = hinge_loss(y_true, y_pred, multioutput='sum')
        #torch_loss_ = torch_loss(torch.nn.HingeEmbeddingLoss(reduction='sum'), y_true, y_pred) / self.K
        tensorflow_loss_ = tensorflow_loss(tensorflow.keras.losses.Hinge(reduction=tensorflow.keras.losses.Reduction.SUM), y_true, y_pred)
        #np.testing.assert_allclose(torch_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)
        #np.testing.assert_allclose(custom_loss_, torch_loss_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(custom_loss_, tensorflow_loss_, rtol=1e-5, atol=1e-5)


class TestMultiOutput(unittest.TestCase):

    def setUp(self):
        self.N = 157
        self.K = 13

    def test_binary_cross_entropy(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=True)
        raw_values_avg = np.average(binary_cross_entropy(y_true, y_pred, multioutput='raw_values'))
        uniform_average = binary_cross_entropy(y_true, y_pred, multioutput='uniform_average')
        np.testing.assert_allclose(raw_values_avg, uniform_average, rtol=1e-7, atol=1e-7)

    def test_categorical_cross_entropy(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=True)
        raw_values_avg = np.average(categorical_cross_entropy(y_true, y_pred, multioutput='raw_values'))
        uniform_average = categorical_cross_entropy(y_true, y_pred, multioutput='uniform_average')
        np.testing.assert_allclose(raw_values_avg, uniform_average, rtol=1e-7, atol=1e-7)

    def test_mean_absolute_error(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=True)
        raw_values_avg = np.average(mean_absolute_error(y_true, y_pred, multioutput='raw_values'))
        uniform_average = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')
        np.testing.assert_allclose(raw_values_avg, uniform_average, rtol=1e-7, atol=1e-7)

    def test_mean_squared_error(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=True)
        raw_values_avg = np.average(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
        uniform_average = mean_squared_error(y_true, y_pred, multioutput='uniform_average')
        np.testing.assert_allclose(raw_values_avg, uniform_average, rtol=1e-7, atol=1e-7)

    def test_huber_loss(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=True)
        raw_values_avg = np.average(huber_loss(y_true, y_pred, multioutput='raw_values'))
        uniform_average = huber_loss(y_true, y_pred, multioutput='uniform_average')
        np.testing.assert_allclose(raw_values_avg, uniform_average, rtol=1e-7, atol=1e-7)

    def test_hinge_loss(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=True)
        raw_values_avg = np.average(hinge_loss(y_true, y_pred, multioutput='raw_values'))
        uniform_average = hinge_loss(y_true, y_pred, multioutput='uniform_average')
        np.testing.assert_allclose(raw_values_avg, uniform_average, rtol=1e-7, atol=1e-7)


class TestGradients(unittest.TestCase):

    def setUp(self):
        self.N = 157
        self.K = 13

    def test_binary_cross_entropy_grad(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=True)
        ana_grad = binary_cross_entropy(y_true, y_pred, prime=True)
        num_grad = grad_check_loss(partial(binary_cross_entropy, multioutput='raw_values'), y_true, y_pred)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_categorical_cross_entropy_grad(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K, probability=True)
        ana_grad = categorical_cross_entropy(y_true, y_pred, prime=True)
        num_grad = grad_check_loss(partial(categorical_cross_entropy, multioutput='raw_values'), y_true, y_pred)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_mean_absolute_error_grad(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K)
        ana_grad = mean_absolute_error(y_true, y_pred, prime=True)
        num_grad = grad_check_loss(partial(mean_absolute_error, multioutput='raw_values'), y_true, y_pred)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_mean_squared_error_grad(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K)
        ana_grad = mean_squared_error(y_true, y_pred, prime=True)
        num_grad = grad_check_loss(partial(mean_squared_error, multioutput='raw_values'), y_true, y_pred)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_huber_loss_grad(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K)
        ana_grad = huber_loss(y_true, y_pred, prime=True)
        num_grad = grad_check_loss(partial(huber_loss, multioutput='raw_values'), y_true, y_pred)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)

    def test_hinge_loss_grad(self):
        y_true = make_random_targets(self.N, self.K)
        y_pred = make_random_predictions(self.N, self.K)
        ana_grad = hinge_loss(y_true, y_pred, prime=True)
        num_grad = grad_check_loss(partial(hinge_loss, multioutput='raw_values'), y_true, y_pred)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
