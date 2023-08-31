from typing import Tuple, Optional
import numpy as np
from .callback import Callback


class BaseCallback(Callback):
    def on_batch_begin(self, model, epoch_info: Tuple[int, int], batch_info: Optional[Tuple[int, int]],
                       batch_samples: np.ndarray, batch_labels: np.ndarray, status: str) -> None:
        pass

    def on_batch_end(self, model, epoch_info: Tuple[int, int], batch_info: Optional[Tuple[int, int]],
                     batch_samples: np.ndarray, batch_labels: np.ndarray, status: str) -> None:
        pass

    def on_epoch_begin(self, model, epoch_info: Tuple[int, int], batch_info: Optional[Tuple[int, int]],
                       batch_samples: np.ndarray, batch_labels: np.ndarray, status: str) -> None:
        if epoch_info[0] == 1:
            model.is_training(True)

    def on_epoch_end(self, model, epoch_info: Tuple[int, int], batch_info: Optional[Tuple[int, int]],
                     batch_samples: np.ndarray, batch_labels: np.ndarray, status: str) -> None:
        for layer in model.layers:
            layer.optimizer.next_epoch()
            if hasattr(layer, 'weights'):
                assert not np.any(np.isnan(layer.weights)), "NaN values detected in layer weights"
                assert not np.any(np.isinf(layer.weights)), "Infinity values detected in layer weights"
            if hasattr(layer, 'bias'):
                assert not np.any(np.isnan(layer.bias)), "NaN values detected in layer weights"
                assert not np.any(np.isinf(layer.bias)), "Infinity values detected in layer weights"

        if epoch_info[0] == epoch_info[1]:
            model.is_training(False)

