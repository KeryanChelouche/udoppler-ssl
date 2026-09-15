import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import Probe


class KNNProbe(Probe):
    """k-Nearest Neighbour probe with StandardScaler + cosine distance."""

    def __init__(self, k: int = 10, metric: str = "cosine") -> None:
        self.k = k
        self.metric = metric
        self._pipeline: Pipeline | None = None

    @property
    def name(self) -> str:
        return f"knn_k{self.k}"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNProbe":
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=self.k, metric=self.metric)),
        ])
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._pipeline is not None, "Call fit() first."
        return self._pipeline.predict(X)
