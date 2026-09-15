from abc import ABC, abstractmethod

import numpy as np
import torch


class FeatureExtractor(ABC):
    """Abstract base class for all feature extractors.

    Each subclass wraps a pretrained model and exposes a single
    extract() method that maps a batch of spectrograms to pooled
    feature vectors.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this model variant (used as cache key)."""
        ...

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimension of the output feature vectors."""
        ...

    @abstractmethod
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """Extract pooled features from a batch of spectrograms.

        Args:
            x: Input tensor, shape (B, F, T) or (B, 1, F, T).

        Returns:
            Float32 numpy array of shape (B, embed_dim).
        """
        ...

    def to(self, device: torch.device | str) -> "FeatureExtractor":
        """Move the underlying model to a device. Returns self."""
        return self
