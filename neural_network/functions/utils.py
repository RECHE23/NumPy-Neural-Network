from typing import Callable, Tuple, Union
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from neural_network.tools import trace


@trace()
def convert_targets(targets: np.ndarray, to: str = None) -> np.ndarray:
    """
    Convert target vectors between different formats.

    Parameters:
    -----------
    targets : np.ndarray
        The target vectors to be converted.
    to : str or None, optional
        The target format to convert to. Possible values: "categorical", "one_hot", "binary", "labels", "probability".

    Returns:
    --------
    converted_targets : np.ndarray
        The converted target vectors.
    """
    if to is None:
        if targets.ndim == 1:
            # Convert to categorical (equivalent to keras.tools.to_categorical):
            num_classes = len(set(targets))
            converted_targets = np.eye(num_classes)[targets]
        else:
            # By default, no conversion needed
            converted_targets = targets
    elif to == "categorical":
        # Convert to categorical (equivalent to keras.tools.to_categorical):
        num_classes = len(set(targets))
        converted_targets = np.eye(num_classes)[targets]
    elif to == "one_hot":
        # Convert target vectors to one-hot vectors:
        idx = np.argmax(targets, axis=-1)
        converted_targets = np.zeros(targets.shape)
        converted_targets[np.arange(targets.shape[0]), idx] = 1
    elif to == "binary":
        # Convert a probability vector to a binary vector using a threshold of 0.5:
        converted_targets = np.where(targets >= 0.5, 1, 0)
    elif to == "labels":
        if targets.ndim != 1:
            # Convert target vectors to labels:
            converted_targets = np.argmax(targets, axis=-1)
        else:
            # No conversion needed, already in label form
            converted_targets = targets
    elif to == "probability":
        # Convert target vectors to probability distributions:
        from neural_network.functions.activation import softmax
        converted_targets = softmax(targets)
    else:
        raise ValueError(f"Unsupported target format: {to}")

    return converted_targets


def pair(args: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """
    Convert arguments to a pair format.

    Parameters:
    -----------
    args : int or tuple of int
        Arguments to be converted.

    Returns:
    --------
    result : tuple of int
        Converted arguments in a pair format.
    """
    assert isinstance(args, tuple) or isinstance(args, int)
    if isinstance(args, tuple):
        assert len(args) == 2
    else:
        args = (args, args)
    return args


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
