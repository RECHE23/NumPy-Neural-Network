import numpy as np
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
