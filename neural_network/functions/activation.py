import math
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
    sigmoid_x = sigmoid(x)
    if prime:
        return sigmoid_x * (1 + x * (1 - sigmoid_x))
    return x * sigmoid_x


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
        return x / (2 * np.sqrt(x**2 + 1)) + 1
    return 0.5 * (np.sqrt(x**2 + 1) - 1) + x


@trace()
def selu(x: np.ndarray, prime: bool = False) -> np.ndarray:
    """
    Scaled Exponential Linear Unit (SELU) activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply SELU activation to.
    prime : bool, optional
        If True, return the derivative of the SELU function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying SELU activation or its derivative.

    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    if prime:
        return scale * np.where(x > 0, 1, alpha * np.exp(x))
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


@trace()
def celu(x: np.ndarray, alpha: float = 1.0, prime: bool = False) -> np.ndarray:
    """
    Continuously Differentiable Exponential Linear Unit (CELU) activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply CELU activation to.
    alpha : float, optional
        Scaling parameter for the negative region. Default is 1.0.
    prime : bool, optional
        If True, return the derivative of the CELU function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying CELU activation or its derivative.

    """
    if prime:
        return np.where(x > 0, 1.0, alpha * np.exp(x / alpha))
    return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))


@trace()
def erf(x: np.ndarray, prime: bool = False) -> np.ndarray:
    """
    Error Function (erf) activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply erf activation to.
    prime : bool, optional
        If True, return the derivative of the erf function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying erf activation or its derivative.

    """
    if prime:
        return 2 / np.sqrt(np.pi) * np.exp(-x ** 2)
    return np.vectorize(math.erf)(x)


@trace()
def gelu(x: np.ndarray, prime: bool = False, approximate: str = 'none') -> np.ndarray:
    """
    Gaussian Error Linear Unit (GELU) activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply GELU activation to.
    prime : bool, optional
        If True, return the derivative of the GELU function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying GELU activation or its derivative.

    """
    if approximate == 'tanh':
        if prime:
            s = x / np.sqrt(2)
            approx = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))
            return 0.5 + 0.5 * approx + ((0.5 * x * erf(s, prime=True)) / np.sqrt(2))
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    elif approximate == 'none':
        if prime:
            s = x / np.sqrt(2)
            return 0.5 + 0.5 * erf(s) + ((0.5 * x * erf(s, prime=True)) / np.sqrt(2))
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))
    else:
        raise NotImplementedError


@trace()
def softplus(x: np.ndarray, beta: float = 1, threshold: float = 20, prime: bool = False) -> np.ndarray:
    """
    Softplus activation function with custom beta and threshold.

    Parameters:
    -----------
    x : array-like
        Input array to apply softplus activation to.
    beta : float, optional
        The slope of the linear part of the softplus function for positive inputs. Default is 1.
    threshold : float, optional
        The point at which the function transitions from linear to exponential behavior. Default is 20.
    prime : bool, optional
        If True, return the derivative of the softplus function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying softplus activation or its derivative.

    """
    if prime:
        return np.where(x > threshold, 1, 1 / (1 + np.exp(-beta * x)))
    return np.where(x > threshold, x, 1 / beta * np.log1p(np.exp(beta * x)))


@trace()
def mish(x: np.ndarray, prime: bool = False) -> np.ndarray:
    """
    Mish activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply Mish activation to.
    prime : bool, optional
        If True, return the derivative of the Mish function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying Mish activation or its derivative.

    """
    if prime:
        omega = np.exp(3 * x) + 4 * np.exp(2 * x) + (6 + 4 * x) * np.exp(x) + 4 * (1 + x)
        delta = 1 + pow((np.exp(x) + 1), 2)
        return np.exp(x) * omega / pow(delta, 2)
    return x * np.tanh(np.log(1 + np.exp(x)))


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
        return np.einsum('ij,jk->ijk', s, np.eye(s.shape[-1])) - np.einsum('ij,ik->ijk', s, s)
    return s


# Update the dictionary of activation functions
activation_functions = {
    "relu": relu,
    "tanh": tanh,
    "sigmoid": sigmoid,
    "leaky_relu": leaky_relu,
    "elu": elu,
    "swish": swish,
    "arctan": arctan,
    "gaussian": gaussian,
    "silu": silu,
    "bent_identity": bent_identity,
    "selu": selu,
    "celu": celu,
    "gelu": gelu,
    "softplus": softplus,
    "mish": mish,
    "softmax": softmax
}
