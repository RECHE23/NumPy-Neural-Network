from typing import Tuple, Optional
import time
import numpy as np
from .callback import Callback
from ..functions.utils import convert_targets


class ProgressCallback(Callback):
    """
    A callback for displaying progress during training epochs and batches.

    This callback provides a visual progress bar for each batch and calculates
    and displays the average error and accuracy for each epoch. Additionally,
    it records and displays the total training time.

    Attributes:
    -----------
    error : float
        Accumulated error across batches within an epoch.
    accuracy : float
        Accumulated accuracy across batches within an epoch.

    Methods:
    --------
    on_epoch_begin(model, epoch_info, batch_info, samples, labels, status):
        Called at the beginning of each epoch.
    on_batch_begin(model, epoch_info, batch_info, batch_samples, batch_labels, status):
        Called at the beginning of each batch.
    on_batch_end(model, epoch_info, batch_info, batch_samples, batch_labels, status):
        Called at the end of each batch.
    on_epoch_end(model, epoch_info, batch_info, samples, labels, status):
        Called at the end of each epoch.
    """

    def __init__(self, progress_bar_length: int = 74):
        super().__init__()
        self.error: float = 0
        self.accuracy: float = 0
        self.start_time = None
        self.progress_bar_length = progress_bar_length

    def on_epoch_begin(self, model, epoch_info: Tuple[int, int], batch_info: Optional[Tuple[int, int]],
                       samples: np.ndarray, labels: np.ndarray, status: str) -> None:
        """
        Called at the beginning of each epoch.

        Initializes the error and accuracy attributes and prints a header
        at the beginning of the first epoch. Records the start time for
        training duration calculation.

        Parameters:
        -----------
        model : NeuralNetwork
            The neural network being trained.
        epoch_info : Tuple[int, int]
            A tuple containing the current epoch number and the total number of epochs.
        batch_info : Optional[Tuple[int, int]]
            A tuple containing the current batch number and the total number of batches.
            This is set to None for epoch-level callbacks.
        samples : np.ndarray
            Input samples.
        labels : np.ndarray
            Target labels.
        status : str
            Indicates the current callback status ("epoch_begin", "epoch_end", etc.).
        """
        self.error = 0
        self.accuracy = 0
        if epoch_info[0] == 1:
            print(f"Training on {len(samples)} samples:")
            self.start_time = time.time()

    def on_batch_begin(self, model, epoch_info: Tuple[int, int], batch_info: Optional[Tuple[int, int]],
                       batch_samples: np.ndarray, batch_labels: np.ndarray, status: str) -> None:
        """
        Called at the beginning of each batch.

        Prints a progress bar indicating the current batch's progress.

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
            Indicates the current callback status ("batch_begin", "batch_end", etc.).
        """
        progress = (batch_info[0] * self.progress_bar_length) // batch_info[1]
        progress_bar = "█" * progress + "░" * (self.progress_bar_length - progress)
        print("\r", end=f"{progress_bar} Training on batch {batch_info[0]} of {batch_info[1]}.")

    def on_batch_end(self, model, epoch_info: Tuple[int, int], batch_info: Optional[Tuple[int, int]],
                     batch_samples: np.ndarray, batch_labels: np.ndarray, status: str) -> None:
        """
        Called at the end of each batch.

        Accumulates error and accuracy statistics for the current batch.

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
            Indicates the current callback status ("batch_begin", "batch_end", etc.).
        """
        self.error += model.layers[-1].loss(batch_labels, batch_samples)
        self.accuracy += np.sum(convert_targets(batch_samples, to="labels") == convert_targets(batch_labels, to="labels"))

    def on_epoch_end(self, model, epoch_info: Tuple[int, int], batch_info: Optional[Tuple[int, int]],
                     samples: np.ndarray, labels: np.ndarray, status: str) -> None:
        """
        Called at the end of each epoch.

        Calculates and displays the average error and accuracy for the epoch.
        Also calculates and displays the total training time.

        Parameters:
        -----------
        model : NeuralNetwork
            The neural network being trained.
        epoch_info : Tuple[int, int]
            A tuple containing the current epoch number and the total number of epochs.
        batch_info : Optional[Tuple[int, int]]
            A tuple containing the current batch number and the total number of batches.
            This is set to None for epoch-level callbacks.
        samples : np.ndarray
            Input samples.
        labels : np.ndarray
            Target labels.
        status : str
            Indicates the current callback status ("epoch_begin", "epoch_end", etc.).
        """
        total_samples = len(samples)
        avg_error = self.error / total_samples
        avg_accuracy = self.accuracy / total_samples
        print("\r", end=f"Epoch {epoch_info[0]:4d} of {epoch_info[1]:<4d}"
              f" \t Average Error = {avg_error:.6f} \t Average Accuracy = {avg_accuracy:.2%}" + 35 * " " + "\n")

        if epoch_info[0] == epoch_info[1]:
            end_time = time.time()
            formatted_time = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(end_time - self.start_time))
            print(f"Training time : {formatted_time}")
