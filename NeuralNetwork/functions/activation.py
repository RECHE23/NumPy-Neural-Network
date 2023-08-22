from typing import Union, Callable
import numpy as np
from NeuralNetwork.tools import trace


@trace()
def relu(x: np.ndarray, prime: bool = False) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply ReLU activation to.
    prime : bool, optional
        If True, return the derivative of the ReLU function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying ReLU activation or its derivative.

    """
    if prime:
        return 1. * (x > 0)
    return x * (x > 0)


@trace()
def tanh(x: np.ndarray, prime: bool = False) -> np.ndarray:
    """
    Hyperbolic Tangent (tanh) activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply tanh activation to.
    prime : bool, optional
        If True, return the derivative of the tanh function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying tanh activation or its derivative.

    """
    if prime:
        return 1 - np.tanh(x)**2
    return np.tanh(x)


@trace()
def sigmoid(x: np.ndarray, prime: bool = False) -> np.ndarray:
    """
    Sigmoid activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply sigmoid activation to.
    prime : bool, optional
        If True, return the derivative of the sigmoid function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying sigmoid activation or its derivative.

    """
    s = 1 / (1 + np.exp(-x))
    if prime:
        return s * (1 - s)
    return s


@trace()
def softmax(x: np.ndarray, prime: bool = False) -> np.ndarray:
    """
    Softmax activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply softmax activation to.
    prime : bool, optional
        If True, return the derivative of the softmax function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying softmax activation or its derivative.

    """
    e = np.exp(x - np.max(x))
    s = e / np.sum(e, axis=-1, keepdims=True)
    if prime:
        return np.diagflat(s) - np.dot(s, s.T)
    return s
