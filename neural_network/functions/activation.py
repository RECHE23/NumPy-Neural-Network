import numpy as np
from neural_network.tools import trace


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
    return np.maximum(x, 0, x)


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


@trace()
def leaky_relu(x: np.ndarray, alpha: float = 0.01, prime: bool = False) -> np.ndarray:
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply Leaky ReLU activation to.
    alpha : float, optional
        Slope of the negative part of the activation function. Default is 0.01.
    prime : bool, optional
        If True, return the derivative of the Leaky ReLU function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying Leaky ReLU activation or its derivative.

    """
    if prime:
        return (x > 0) + (alpha * (x <= 0))
    return np.maximum(x, alpha * x)


@trace()
def elu(x: np.ndarray, alpha: float = 1.0, prime: bool = False) -> np.ndarray:
    """
    Exponential Linear Unit (ELU) activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply ELU activation to.
    alpha : float, optional
        Slope of the negative part of the activation function. Default is 1.0.
    prime : bool, optional
        If True, return the derivative of the ELU function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying ELU activation or its derivative.

    """
    if prime:
        return (x > 0) + (alpha * np.exp(x) * (x <= 0))
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


@trace()
def prelu(x: np.ndarray, alpha: float = 0.01, prime: bool = False) -> np.ndarray:
    """
    Parametric Rectified Linear Unit (PReLU) activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply PReLU activation to.
    alpha : float, optional
        Slope of the negative part of the activation function. Default is 0.01.
    prime : bool, optional
        If True, return the derivative of the PReLU function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying PReLU activation or its derivative.

    """
    if prime:
        return (x > 0) + (alpha * (x <= 0))
    return np.maximum(x, alpha * x)


@trace()
def swish(x: np.ndarray, beta: float = 1.0, prime: bool = False) -> np.ndarray:
    """
    Swish activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply Swish activation to.
    beta : float, optional
        Scaling parameter. Default is 1.0.
    prime : bool, optional
        If True, return the derivative of the Swish function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying Swish activation or its derivative.

    """
    if prime:
        sigmoid_x = 1 / (1 + np.exp(-beta * x))
        return sigmoid_x + beta * x * sigmoid_x * (1 - sigmoid_x)
    return x * sigmoid(beta * x)


@trace()
def arctan(x: np.ndarray, prime: bool = False) -> np.ndarray:
    """
    ArcTan (Inverse Tangent) activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply ArcTan activation to.
    prime : bool, optional
        If True, return the derivative of the ArcTan function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying ArcTan activation or its derivative.

    """
    if prime:
        return 1 / (1 + x**2)
    return np.arctan(x)


@trace()
def gaussian(x: np.ndarray, prime: bool = False) -> np.ndarray:
    """
    Gaussian activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply Gaussian activation to.
    prime : bool, optional
        If True, return the derivative of the Gaussian function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying Gaussian activation or its derivative.

    """
    if prime:
        return -2 * x * np.exp(-x**2)
    return np.exp(-x**2)


@trace()
def silu(x: np.ndarray, prime: bool = False) -> np.ndarray:
    """
    SiLU (Sigmoid-weighted Linear Unit) activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply SiLU activation to.
    prime : bool, optional
        If True, return the derivative of the SiLU function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying SiLU activation or its derivative.

    """
    if prime:
        sigmoid_x = 1 / (1 + np.exp(-x))
        return sigmoid_x * (1 + x * (1 - sigmoid_x))
    return x * sigmoid(x)


@trace()
def bent_identity(x: np.ndarray, prime: bool = False) -> np.ndarray:
    """
    Bent Identity activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply Bent Identity activation to.
    prime : bool, optional
        If True, return the derivative of the Bent Identity function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying Bent Identity activation or its derivative.

    """
    if prime:
        return (2 * x + 1) / (2 * np.sqrt(x**2 + x) + 1)
    return (np.sqrt(x**2 + 1) - 1) + x


# Update the dictionary of activation functions
activation_functions = {
    "relu": relu,
    "tanh": tanh,
    "sigmoid": sigmoid,
    "softmax": softmax,
    "leaky_relu": leaky_relu,
    "elu": elu,
    "prelu": prelu,
    "swish": swish,
    "arctan": arctan,
    "gaussian": gaussian,
    "silu": silu,
    "bent_identity": bent_identity
}
