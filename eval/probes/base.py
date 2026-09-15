from abc import ABC, abstractmethod

import numpy as np


class Probe(ABC):
    """Abstract probe (classifier) trained on frozen features."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Probe":
        """Fit on training features and labels. Returns self."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for test features."""
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean accuracy on (X, y)."""
        return float((self.predict(X) == y).mean())
