from itertools import product
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import Callable, Tuple


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


def apply_padding(array: np.ndarray, kernel: np.ndarray, mode: str, boundary: str = "constant", fillvalue: int = 0, pad_after: bool = False) -> np.ndarray:
    """
    Apply padding to an array based on the kernel shape and padding mode.

    Parameters:
    -----------
    array : array-like
        Input array to be padded.
    kernel : array-like
        Kernel for convolution.
    mode : str
        Padding mode.
    boundary : str, optional
        Boundary mode. Default is "constant".
    fillvalue : int, optional
        Fill value for padding. Default is 0.
    pad_after : bool, optional
        If True, pad after the array. Default is False.

    Returns:
    --------
    result : array-like
        Padded array.

    """
    if mode != "valid":
        if mode == "full":
            padding_sizes = [(s - 1, s - 1) for s in kernel.shape]
        elif mode == "same":
            padding_sizes = [((s - 1) // 2, s - 1 - (s - 1) // 2) for s in kernel.shape]
            if pad_after:
                padding_sizes = [p[::-1] for p in padding_sizes]
        else:
            raise NotImplementedError

        array = np.pad(array, padding_sizes, mode=boundary, constant_values=fillvalue)
    return array


def parallel_iterator(function: Callable, *iterables: Tuple) -> None:
    """
    Apply a given function to each combination of elements from input iterables in parallel.

    Parameters:
    -----------
    function : callable
        The function to be applied.
    *iterables : tuple of iterable
        The input iterables.

    Returns:
    --------
    None

    """
    with ThreadPoolExecutor() as executor:
        executor.map(function, product(*iterables))
