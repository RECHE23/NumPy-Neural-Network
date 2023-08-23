import numpy as np
from .utils import apply_padding


def correlate2d(input_array: np.ndarray, kernel: np.ndarray, mode: str = "full", boundary: str = "constant", fillvalue: int = 0) -> np.ndarray:
    """
    Compute 2-dimensional correlation using the provided kernel.

    Equivalent to scipy.signal.correlate2d.

    Parameters:
    -----------
    input_array : array-like
        Input array to be correlated.
    kernel : array-like
        Kernel for the correlation.
    mode : str, optional
        Padding mode. Default is "full".
    boundary : str, optional
        Boundary mode. Default is "constant".
    fillvalue : int, optional
        Fill value for padding. Default is 0.

    Returns:
    --------
    result : array-like
        Correlation result.

    """
    padded_array = apply_padding(input_array, kernel, mode, boundary, fillvalue)
    sliding_view = np.lib.stride_tricks.sliding_window_view(padded_array, kernel.shape)

    return np.einsum('ijkl,kl->ij', sliding_view, kernel)


def convolve2d(input_array: np.ndarray, kernel: np.ndarray, mode: str = "full", boundary: str = "constant", fillvalue: int = 0) -> np.ndarray:
    """
    Compute 2-dimensional convolution using the provided kernel.

    Equivalent to scipy.signal.convolve2d.

    Parameters:
    -----------
    input_array : array-like
        Input array to be convolved.
    kernel : array-like
        Kernel for convolution.
    mode : str, optional
        Padding mode. Default is "full".
    boundary : str, optional
        Boundary mode. Default is "constant".
    fillvalue : int, optional
        Fill value for padding. Default is 0.

    Returns:
    --------
    result : array-like
        Convolution result.

    """
    padded_array = apply_padding(input_array, kernel, mode, boundary, fillvalue, pad_after=True)
    sliding_view = np.lib.stride_tricks.sliding_window_view(padded_array, kernel.shape)

    return np.einsum('ijkl,kl->ij', sliding_view, np.flip(kernel))

