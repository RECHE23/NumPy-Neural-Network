from typing import Tuple, Optional
from abc import abstractmethod
import numpy as np


class Callback:
    """
    A base class for creating custom callbacks during training.

    This class provides a structure for implementing various callback functions
    that can be executed at different stages of training, such as before/after
    batches or epochs. Subclasses of this class should override the abstract
    methods to define specific behavior for each callback.

    Methods:
    --------
    __call__(model, epoch_info, batch_info, batch_samples, batch_labels, status):
        The entry point for callback execution.
    on_batch_begin(model, epoch_info, batch_info, batch_samples, batch_labels, status):
        Called at the beginning of each batch.
    on_batch_end(model, epoch_info, batch_info, batch_samples, batch_labels, status):
        Called at the end of each batch.
    on_epoch_begin(model, epoch_info, batch_info, batch_samples, batch_labels, status):
        Called at the beginning of each epoch.
    on_epoch_end(model, epoch_info, batch_info, batch_samples, batch_labels, status):
        Called at the end of each epoch.
    """

    def __init__(self):
        pass

    def __call__(self, model, epoch_info: Tuple[int, int], batch_info: Optional[Tuple[int, int]],
                 batch_samples: np.ndarray, batch_labels: np.ndarray, status: str) -> None:
        """
        The entry point for callback execution.

        This method determines the callback type based on the `status` argument
        and calls the corresponding callback method.

        Parameters:
        -----------
        model : NeuralNetwork
            The neural network being trained.
        epoch_info : Tuple[int, int]
            A tuple containing the current epoch number and the total number of epochs.
        batch_info : Optional[Tuple[int, int]]
            A tuple containing the current batch number and the total number of batches.
            This is set to None for epoch-level callbacks.
        batch_samples : np.ndarray
            Input samples for the current batch.
        batch_labels : np.ndarray
            Target labels for the current batch.
        status : str
            Indicates the current callback status ("batch_begin", "batch_end", "epoch_begin", "epoch_end").
        """
        assert status in ("batch_begin", "batch_end", "epoch_begin", "epoch_end")

        if status == "batch_begin":
            self.on_batch_begin(model, epoch_info, batch_info, batch_samples, batch_labels, status)
        elif status == "batch_end":
            self.on_batch_end(model, epoch_info, batch_info, batch_samples, batch_labels, status)
        elif status == "epoch_begin":
            self.on_epoch_begin(model, epoch_info, batch_info, batch_samples, batch_labels, status)
        else:
            self.on_epoch_end(model, epoch_info, batch_info, batch_samples, batch_labels, status)

    @abstractmethod
    def on_batch_begin(self, model, epoch_info: Tuple[int, int], batch_info: Optional[Tuple[int, int]],
                       batch_samples: np.ndarray, batch_labels: np.ndarray, status: str) -> None:
        """
        Called at the beginning of each batch.

        This method should be overridden in subclasses to define specific behavior
        to be executed at the beginning of each batch.

        Parameters:
        -----------
        model : NeuralNetwork
            The neural network being trained.
        epoch_info : Tuple[int, int]
            A tuple containing the current epoch number and the total number of epochs.
        batch_info : Optional[Tuple[int, int]]
            A tuple containing the current batch number and the total number of batches.
            This is set to None for epoch-level callbacks.
        batch_samples : np.ndarray
            Input samples for the current batch.
        batch_labels : np.ndarray
            Target labels for the current batch.
        status : str
            Indicates the current callback status ("batch_begin", "batch_end", "epoch_begin", "epoch_end").
        """
        raise NotImplementedError

    @abstractmethod
    def on_batch_end(self, model, epoch_info: Tuple[int, int], batch_info: Optional[Tuple[int, int]],
                     batch_samples: np.ndarray, batch_labels: np.ndarray, status: str) -> None:
        """
        Called at the end of each batch.

        This method should be overridden in subclasses to define specific behavior
        to be executed at the end of each batch.

        Parameters:
        -----------
        model : NeuralNetwork
            The neural network being trained.
        epoch_info : Tuple[int, int]
            A tuple containing the current epoch number and the total number of epochs.
        batch_info : Optional[Tuple[int, int]]
            A tuple containing the current batch number and the total number of batches.
            This is set to None for epoch-level callbacks.
        batch_samples : np.ndarray
            Input samples for the current batch.
        batch_labels : np.ndarray
            Target labels for the current batch.
        status : str
            Indicates the current callback status ("batch_begin", "batch_end", "epoch_begin", "epoch_end").
        """
        raise NotImplementedError

    @abstractmethod
    def on_epoch_begin(self, model, epoch_info: Tuple[int, int], batch_info: Optional[Tuple[int, int]],
                       batch_samples: np.ndarray, batch_labels: np.ndarray, status: str) -> None:
        """
        Called at the beginning of each epoch.

        This method should be overridden in subclasses to define specific behavior
        to be executed at the beginning of each epoch.

        Parameters:
        -----------
        model : NeuralNetwork
            The neural network being trained.
        epoch_info : Tuple[int, int]
            A tuple containing the current epoch number and the total number of epochs.
        batch_info : Optional[Tuple[int, int]]
            A tuple containing the current batch number and the total number of batches.
            This is set to None for epoch-level callbacks.
        batch_samples : np.ndarray
            Input samples for the current batch.
        batch_labels : np.ndarray
            Target labels for the current batch.
        status : str
            Indicates the current callback status ("batch_begin", "batch_end", "epoch_begin", "epoch_end").
        """
        raise NotImplementedError

    @abstractmethod
    def on_epoch_end(self, model, epoch_info: Tuple[int, int], batch_info: Optional[Tuple[int, int]],
                     batch_samples: np.ndarray, batch_labels: np.ndarray, status: str) -> None:
        """
        Called at the end of each epoch.

        This method should be overridden in subclasses to define specific behavior
        to be executed at the end of each epoch.

        Parameters:
        -----------
        model : NeuralNetwork
            The neural network being trained.
        epoch_info : Tuple[int, int]
            A tuple containing the current epoch number and the total number of epochs.
        batch_info : Optional[Tuple[int, int]]
            A tuple containing the current batch number and the total number of batches.
            This is set to None for epoch-level callbacks.
        batch_samples : np.ndarray
            Input samples for the current batch.
        batch_labels : np.ndarray
            Target labels for the current batch.
        status : str
            Indicates the current callback status ("batch_begin", "batch_end", "epoch_begin", "epoch_end").
        """
        raise NotImplementedError
