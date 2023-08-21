from itertools import product
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def correlate2d(input_array, kernel, mode="full"):
    """
    Equivalent to scipy.signal.correlate2d.
    """
    if mode == "full":
        h_padding = kernel.shape[0] - 1  # Horizontal padding
        v_padding = kernel.shape[1] - 1  # Vertical padding
        input_array = np.pad(input_array, [(h_padding, h_padding), (v_padding, v_padding)], mode='constant')

    return np.einsum('ijkl,kl->ij', np.lib.stride_tricks.sliding_window_view(input_array, kernel.shape), kernel)


def convolve2d(input_array, kernel, mode="full"):
    """
    Equivalent to scipy.signal.convolve2d.
    """
    if mode == "full":
        h_padding = kernel.shape[0] - 1  # Horizontal padding
        v_padding = kernel.shape[1] - 1  # Vertical padding
        input_array = np.pad(input_array, [(h_padding, h_padding), (v_padding, v_padding)], mode='constant')

    kernel = np.flip(kernel)
    return np.einsum('ijkl,kl->ij', np.lib.stride_tricks.sliding_window_view(input_array, kernel.shape), kernel)


def parallel_iterator(function, *iterables):
    with ThreadPoolExecutor() as executor:
        executor.map(function, product(*iterables))
