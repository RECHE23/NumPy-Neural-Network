from typing import Union
import numpy as np
from neural_network.tools import trace, debug_assert


@trace()
def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, prime: bool = False, multioutput: str = 'sum', epsilon: float = 1e-10) -> Union[np.ndarray, float]:
    """
    Compute the binary cross-entropy loss between predicted and true values.

    Parameters:
    -----------
    y_true : np.ndarray, shape (n_samples, n_classes)
        True target values (binary labels: 0 or 1).
    y_pred : np.ndarray, shape (n_samples, n_classes)
        Predicted values after applying sigmoid activation.
    prime : bool, optional
        If True, compute the derivative of the loss with respect to y_pred. Default is False.
    multioutput : str, optional
        Strategy to handle multioutput data. Options: 'raw_values', 'uniform_average', 'sum'. Default is 'sum'.
    epsilon : float, optional
        Prevents numeric instability. Default is 1e-10.

    Returns:
    --------
    loss : np.ndarray or float
        Binary cross-entropy loss or its derivative if prime is True.
    """
    debug_assert((y_pred >= 0).all() and (y_pred <= 1).all(),
                 "Binary cross-entropy loss takes in predicted probabilities.")

    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

    if prime:
        return (- y_true / y_pred_clipped + (1 - y_true) / (1 - y_pred_clipped)) / y_pred.shape[1]

    if multioutput == 'raw_values':
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped), axis=-1)
    elif multioutput in ('sum', 'uniform_average'):
        loss = 0.0
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                loss -= y_true[i, j] * np.log(y_pred_clipped[i, j]) + (1 - y_true[i, j]) * np.log(1 - y_pred_clipped[i, j])
        if multioutput == 'sum':
            return loss / y_pred.shape[1]
        return loss / np.prod(y_pred.shape)
    else:
        raise ValueError("Invalid 'multioutput' option.")


@trace()
def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, prime: bool = False, multioutput: str = 'sum', epsilon: float = 1e-10) -> Union[np.ndarray, float]:
    """
    Compute the categorical cross-entropy loss between predicted and true values.

    Parameters:
    -----------
    y_true : np.ndarray, shape (n_samples, n_classes)
        True target values with one-hot encoding.
    y_pred : np.ndarray, shape (n_samples, n_classes)
        Predicted values after applying softmax activation.
    prime : bool, optional
        If True, compute the derivative of the loss with respect to y_pred. Default is False.
    multioutput : str, optional
        Strategy to handle multioutput data. Options: 'raw_values', 'uniform_average', 'sum'. Default is 'sum'.
    epsilon : float, optional
        Prevents numeric instability. Default is 1e-10.

    Returns:
    --------
    loss : np.ndarray or float
        Categorical cross-entropy loss or its derivative if prime is True.
    """
    debug_assert((y_pred >= 0).all() and np.allclose(np.sum(y_pred, axis=-1), 1),
                 "Categorical cross-entropy loss takes in a probability distribution.")

    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

    if prime:
        return - y_true / y_pred_clipped

    if multioutput == 'raw_values':
        return -np.sum(y_true * np.log(y_pred_clipped), axis=-1)
    elif multioutput == 'uniform_average':
        return -np.average(np.sum(y_true * np.log(y_pred_clipped), axis=-1))
    elif multioutput == 'sum':
        return -np.sum(y_true * np.log(y_pred_clipped))
    else:
        raise ValueError("Invalid 'multioutput' option.")


@trace()
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, prime: bool = False, multioutput: str = 'sum') -> Union[np.ndarray, float]:
    """
    Compute the mean absolute error loss between predicted and true values.

    Parameters:
    -----------
    y_true : np.ndarray, shape (n_samples, n_classes)
        True target values.
    y_pred : np.ndarray, shape (n_samples, n_classes)
        Predicted values.
    prime : bool, optional
        If True, compute the derivative of the loss with respect to y_pred. Default is False.
    multioutput : str, optional
        Strategy to handle multioutput data. Options: 'raw_values', 'uniform_average', 'sum'. Default is 'sum'.

    Returns:
    --------
    loss : np.ndarray or float
        Mean absolute error loss or its derivative if prime is True.
    """
    if prime:
        return np.sign(y_pred - y_true) / y_true.shape[-1]

    if multioutput == 'raw_values':
        return np.mean(np.abs(y_true - y_pred), axis=-1)
    elif multioutput in ('sum', 'uniform_average'):
        loss = 0.0
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                loss += np.abs(y_true[i, j] - y_pred[i, j])
        if multioutput == 'sum':
            return loss / y_pred.shape[1]
        return loss / np.prod(y_pred.shape)
    else:
        raise ValueError("Invalid 'multioutput' option.")


@trace()
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, prime: bool = False, multioutput: str = 'sum') -> Union[np.ndarray, float]:
    """
    Compute the mean squared error loss between predicted and true values.

    Parameters:
    -----------
    y_true : np.ndarray, shape (n_samples, n_classes)
        True target values.
    y_pred : np.ndarray, shape (n_samples, n_classes)
        Predicted values.
    prime : bool, optional
        If True, compute the derivative of the loss with respect to y_pred. Default is False.
    multioutput : str, optional
        Strategy to handle multioutput data. Options: 'raw_values', 'uniform_average', 'sum'. Default is 'sum'.

    Returns:
    --------
    loss : np.ndarray or float
        Mean squared error loss or its derivative if prime is True.
    """
    if prime:
        return 2 * (y_pred - y_true) / y_true.shape[-1]

    if multioutput == 'raw_values':
        return np.mean(np.power(y_true - y_pred, 2), axis=-1)
    elif multioutput in ('sum', 'uniform_average'):
        loss = 0.0
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                loss += (y_true[i, j] - y_pred[i, j])**2
        if multioutput == 'sum':
            return loss / y_pred.shape[1]
        return loss / np.prod(y_pred.shape)
    else:
        raise ValueError("Invalid 'multioutput' option.")


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0, prime: bool = False, multioutput: str = 'sum') -> Union[np.ndarray, float]:
    """
    Compute the Huber loss between predicted and true values.

    Parameters:
    -----------
    y_true : np.ndarray, shape (n_samples, n_classes)
        True target values.
    y_pred : np.ndarray, shape (n_samples, n_classes)
        Predicted values.
    delta : float, optional
        Threshold for switching between L1 (MAE) and L2 (MSE) loss. Default is 1.0.
    prime : bool, optional
        If True, compute the derivative of the loss with respect to y_pred. Default is False.
    multioutput : str, optional
        Strategy to handle multioutput data. Options: 'raw_values', 'uniform_average', 'sum'. Default is 'sum'.

    Returns:
    --------
    loss : np.ndarray or float
        Huber loss or its derivative if prime is True.
    """
    error = y_true - y_pred
    absolute_error = np.abs(error)

    if prime:
        if delta == 0:
            derivative = error
        else:
            derivative = np.where(absolute_error <= delta, error, delta * np.sign(error))
        return - derivative / y_true.shape[-1]

    if multioutput == 'raw_values':
        loss = np.where(absolute_error <= delta, 0.5 * error**2, delta * absolute_error - 0.5 * delta**2)
        return np.mean(loss, axis=-1)
    elif multioutput in ('sum', 'uniform_average'):
        loss = 0.0
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                if absolute_error[i, j] <= delta:
                    loss += 0.5 * (error[i, j]**2)
                else:
                    loss += delta * absolute_error[i, j] - 0.5 * delta**2
        if multioutput == 'sum':
            return loss / y_pred.shape[1]
        return loss / np.prod(y_pred.shape)
    else:
        raise ValueError("Invalid 'multioutput' option.")


def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray, prime: bool = False, multioutput: str = 'sum') -> Union[np.ndarray, float]:
    """
    Compute the Hinge loss between predicted and true values.

    Parameters:
    -----------
    y_true : np.ndarray, shape (n_samples, n_classes)
        True target values (-1 or 1).
    y_pred : np.ndarray, shape (n_samples, n_classes)
        Predicted values.
    prime : bool, optional
        If True, compute the derivative of the loss with respect to y_pred. Default is False.
    multioutput : str, optional
        Strategy to handle multioutput data. Options: 'raw_values', 'uniform_average', 'sum'. Default is 'sum'.

    Returns:
    --------
    loss : np.ndarray or float
        Hinge loss or its derivative if prime is True.
    """
    if prime:
        margin = y_true * y_pred
        derivative = np.where(margin < 1, -y_true, 0)
        return derivative / y_true.shape[-1]

    if multioutput == 'raw_values':
        loss = np.maximum(0, 1 - y_true * y_pred)
        return np.mean(loss, axis=-1)
    elif multioutput in ('sum', 'uniform_average'):
        loss = 0.0
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                loss += max(0, 1 - y_true[i, j] * y_pred[i, j])
        if multioutput == 'sum':
            return loss / y_pred.shape[1]
        return loss / np.prod(y_pred.shape)
    else:
        raise ValueError("Invalid 'multioutput' option.")


loss_functions = {
    "binary_cross_entropy": binary_cross_entropy,
    "categorical_cross_entropy": categorical_cross_entropy,
    "mean_absolute_error": mean_absolute_error,
    "mean_squared_error": mean_squared_error,
    "huber_loss": huber_loss,
    "hinge_loss": hinge_loss
}
