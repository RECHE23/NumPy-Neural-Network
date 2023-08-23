import numpy as np
from neural_network.tools import trace


@trace()
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, prime: bool = False) -> np.ndarray:
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

    Returns:
    --------
    loss : np.ndarray, shape (n_samples, ...)
        Mean squared error loss or its derivative if prime is True.
    """
    y_pred = y_pred.squeeze()
    if prime:
        return 2 * (y_pred - y_true) / y_true.size
    return np.mean(np.power(y_true - y_pred, 2), axis=-1)


@trace()
def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, prime: bool = False) -> np.ndarray:
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

    Returns:
    --------
    loss : np.ndarray, shape (n_samples, n_classes)
        Categorical cross-entropy loss or its derivative if prime is True.
    """
    y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
    if prime:
        return y_pred_clipped - y_true
    return -np.sum(y_true * np.log(y_pred_clipped), axis=-1)


loss_functions = {
    "mean_squared_error": mean_squared_error,
    "categorical_cross_entropy": categorical_cross_entropy
}
