import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import Probe


class LinearProbe(Probe):
    """Logistic regression probe with StandardScaler preprocessing."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.C = C
        self.max_iter = max_iter
        self._pipeline: Pipeline | None = None

    @property
    def name(self) -> str:
        return f"linear_C{self.C}"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver="lbfgs",
            )),
        ])
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._pipeline is not None, "Call fit() first."
        return self._pipeline.predict(X)
