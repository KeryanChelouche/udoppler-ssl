from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch


class SpectrogramDataset(ABC):
    """Abstract base class for all spectrogram datasets.

    Concrete subclasses are responsible for:
      - Locating data files on disk.
      - Defining the cross-validation strategy.
      - Loading and preprocessing a single sample into a tensor
        that is ready to be fed into a FeatureExtractor.

    All datasets produce **raw, linear-scale** spectrogram tensors of
    shape (1, F, T).  No log-compression or dB conversion is applied —
    each FeatureExtractor is responsible for its own magnitude scaling
    in ``extract()``.  For audio datasets the conversion from waveform
    to Mel spectrogram is done inside ``load_item()``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this dataset (used as cache key)."""
        ...

    @property
    @abstractmethod
    def n_classes(self) -> int:
        ...

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        ...

    @abstractmethod
    def items(self) -> Tuple[List[Path], np.ndarray]:
        """Return all dataset items and their integer class labels.

        Returns:
            items:  List of file paths (one per sample).
            labels: Integer class label array, shape (N,).
        """
        ...

    @abstractmethod
    def load_item(self, item: Path) -> torch.Tensor:
        """Load and preprocess one sample into a model-ready tensor.

        Returns:
            Spectrogram tensor of shape (1, F, T), float32.
        """
        ...

    @property
    def groups(self) -> Optional[np.ndarray]:
        """Per-sample group IDs for grouped CV (e.g. participant).

        Datasets with subject/participant structure should override this
        so that CV splits keep all samples from one group in the same fold.
        Returns None when no grouping is needed (default).
        """
        return None

    @abstractmethod
    def cv_splits(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) for each CV fold."""
        ...
