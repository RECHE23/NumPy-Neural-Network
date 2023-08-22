from itertools import product
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def correlate2d(input_array, kernel, mode="full", boundary="constant", fillvalue=0):
    """
    Equivalent to scipy.signal.correlate2d.
    """
    padded_array = padding(input_array, kernel, mode, boundary, fillvalue)
    sliding_view = np.lib.stride_tricks.sliding_window_view(padded_array, kernel.shape)

    return np.einsum('ijkl,kl->ij', sliding_view, kernel)


def convolve2d(input_array, kernel, mode="full", boundary="constant", fillvalue=0):
    """
    Equivalent to scipy.signal.convolve2d.
    """
    padded_array = padding(input_array, kernel, mode, boundary, fillvalue, pad_after=True)
    sliding_view = np.lib.stride_tricks.sliding_window_view(padded_array, kernel.shape)

    return np.einsum('ijkl,kl->ij', sliding_view, np.flip(kernel))


def padding(array, kernel, mode, boundary="constant", fillvalue=0, pad_after=False):
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


def parallel_iterator(function, *iterables):
    with ThreadPoolExecutor() as executor:
        executor.map(function, product(*iterables))
