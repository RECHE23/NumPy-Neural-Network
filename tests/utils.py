import unittest
from functools import partial
import numpy as np
import torch
import tensorflow
from neural_network.functions.output import softmax


def grad_check_activation(f, x, eps=1e-7):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def grad_check_loss(f, y_true, y_pred, eps=1e-7):
    grad = np.zeros_like(y_true)
    for i in range(grad.shape[1]):
        y_pred_p = y_pred.copy()
        y_pred_p[:, i] += eps
        y_pred_m = y_pred.copy()
        y_pred_m[:, i] -= eps
        grad[:, i] = (f(y_true, y_pred_p) - f(y_true, y_pred_m)) / (2 * eps)
    return grad


def jacobian_check(f, x, eps=1e-7):
    x_flat = x.flatten()
    grad_matrix = np.zeros((x.size, x_flat.size))

    for i in range(x_flat.size):
        x_eps_plus = x_flat.copy()
        x_eps_plus[i] += eps
        f_plus = f(x_eps_plus)

        x_eps_minus = x_flat.copy()
        x_eps_minus[i] -= eps
        f_minus = f(x_eps_minus)

        grad_matrix[:, i] = (f_plus - f_minus) / (2 * eps)

    return grad_matrix.reshape(x.shape + (x.size,))


def make_random_targets(N, K):
    array = np.append(np.arange(0, K), np.random.randint(0, K, size=(N - K)))
    np.random.shuffle(array)
    return np.eye(K)[array]


def make_random_predictions(N, K, probability=False):
    if probability:
        return softmax(np.random.randn(N, K))
    else:
        return np.random.randn(N, K)


def to_pytorch(x):
    return torch.tensor(x, dtype=torch.float32, requires_grad=True)


def to_tensorflow(x):
    if len(x.shape) == 4:
        x = np.moveaxis(x, 1, -1)
    return tensorflow.convert_to_tensor(x, dtype=tensorflow.float32)


def to_numpy(x):
    if tensorflow.is_tensor(x):
        if len(x.shape) == 4:
            x = np.moveaxis(x, -1, 1)
        else:
            x = x.numpy()
    elif isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x


def torch_loss(torch_layer, y_true, y_pred) -> np.ndarray:
    return to_numpy(torch_layer(to_pytorch(y_pred), to_pytorch(y_true)))


def tensorflow_loss(tensorflow_layer, y_true, y_pred) -> np.ndarray:
    return to_numpy(tensorflow_layer(to_tensorflow(y_true), to_tensorflow(y_pred)))


def torch_output(torch_layer, x) -> np.ndarray:
    return to_numpy(torch_layer(to_pytorch(x)))


def tensorflow_output(tensorflow_layer, x) -> np.ndarray:
    return to_numpy(tensorflow_layer(to_tensorflow(x)))


def custom_output(custom_layer, x) -> np.ndarray:
    return custom_layer(x)


def torch_grad(torch_layer, x, upstream_gradients) -> np.ndarray:
    x = to_pytorch(x)
    upstream_gradients = to_pytorch(upstream_gradients)

    torch_output = torch_layer(x)
    torch_output.backward(upstream_gradients)

    return to_numpy(x.grad)


def tensorflow_grad(tensorflow_layer, x, upstream_gradients) -> np.ndarray:
    x = to_tensorflow(x)
    upstream_gradients = to_tensorflow(upstream_gradients)

    with tensorflow.GradientTape() as tape:
        tape.watch(x)
        tf_output = tensorflow_layer(x)

    return to_numpy(tape.gradient(tf_output, x, upstream_gradients))


def custom_grad(custom_layer, x, upstream_gradients) -> np.ndarray:
    custom_layer(x)
    return custom_layer.backward(upstream_gradients, None)
