from typing import Union
import numpy as np
from neural_network.tools import trace, debug_assert


@trace()
def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, prime: bool = False, multioutput: str = 'uniform_average', epsilon: float = 1e-10) -> Union[np.ndarray, float]:
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
        Strategy to handle multioutput data. Options: 'raw_values', 'uniform_average'. Default is 'uniform_average'.
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
        return -np.sum(y_true * np.log(y_pred_clipped))
    else:
        raise ValueError("Invalid 'multioutput' option.")


@trace()
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, prime: bool = False, multioutput: str = 'uniform_average') -> Union[np.ndarray, float]:
    """
    Compute the mean absolute error loss between predicted and true values.

    Parameters:
    -----------
    y_true : np.ndarray, shape (n_samples, ...)
        True target values.
    y_pred : np.ndarray, shape (n_samples, ...)
        Predicted values.
    prime : bool, optional
        If True, compute the derivative of the loss with respect to y_pred. Default is False.
    multioutput : str, optional
        Strategy to handle multioutput data. Options: 'raw_values', 'uniform_average'. Default is 'uniform_average'.

    Returns:
    --------
    loss : np.ndarray or float
        Mean absolute error loss or its derivative if prime is True.
    """
    y_pred = y_pred.squeeze()
    if prime:
        return np.sign(y_pred - y_true) / y_true.shape[-1]
    if multioutput == 'raw_values':
        return np.mean(np.abs(y_true - y_pred), axis=-1)
    elif multioutput == 'uniform_average':
        loss = 0.0
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                loss += np.abs(y_true[i, j] - y_pred[i, j])
        return loss / y_pred.shape[1]
    else:
        raise ValueError("Invalid 'multioutput' option.")


@trace()
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, prime: bool = False, multioutput: str = 'uniform_average') -> Union[np.ndarray, float]:
    """
    Compute the mean squared error loss between predicted and true values.

    Parameters:
    -----------
    y_true : np.ndarray, shape (n_samples, ...)
        True target values.
    y_pred : np.ndarray, shape (n_samples, ...)
        Predicted values.
    prime : bool, optional
        If True, compute the derivative of the loss with respect to y_pred. Default is False.
    multioutput : str, optional
        Strategy to handle multioutput data. Options: 'raw_values', 'uniform_average'. Default is 'uniform_average'.

    Returns:
    --------
    loss : np.ndarray or float
        Mean squared error loss or its derivative if prime is True.
    """
    y_pred = y_pred.squeeze()
    if prime:
        return 2 * (y_pred - y_true) / y_true.shape[-1]
    if multioutput == 'raw_values':
        return np.mean(np.power(y_true - y_pred, 2), axis=-1)
    elif multioutput == 'uniform_average':
        loss = 0.0
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                loss += (y_true[i, j] - y_pred[i, j])**2
        return loss / y_pred.shape[1]
    else:
        raise ValueError("Invalid 'multioutput' option.")


loss_functions = {
    "categorical_cross_entropy": categorical_cross_entropy,
    "mean_absolute_error": mean_absolute_error,
    "mean_squared_error": mean_squared_error
}
