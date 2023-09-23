import numpy as np
from neural_network.tools import trace

try:
    import opt_einsum.contract as einsum
except ImportError:
    from numpy import einsum


@trace()
def softmax(x: np.ndarray, axis=-1, prime: bool = False) -> np.ndarray:
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
    e = np.exp(x - np.max(x, axis=axis)[:, np.newaxis])
    s = e / np.sum(e, axis=axis, keepdims=True)

    if prime:
        return einsum('ij,jk->ijk', s, np.eye(s.shape[-1]), optimize=True) - einsum('ij,ik->ijk', s, s, optimize=True)

    return s


@trace()
def softmin(x: np.ndarray, axis=-1, prime: bool = False) -> np.ndarray:
    """
    Softmin activation function.

    Parameters:
    -----------
    x : array-like
        Input array to apply softmin activation to.
    prime : bool, optional
        If True, return the derivative of the softmin function. Default is False.

    Returns:
    --------
    result : array-like
        Result of applying softmin activation or its derivative.

    """
    e = np.exp(-x - np.max(-x, axis=axis)[:, np.newaxis])
    s = e / np.sum(e, axis=axis, keepdims=True)

    if prime:
        return -einsum('ij,jk->ijk', s, np.eye(s.shape[-1]), optimize=True) + einsum('ij,ik->ijk', s, s, optimize=True)

    return s


output_functions = {
    "softmax": softmax,
    "softmin": softmin
}
