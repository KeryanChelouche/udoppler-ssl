"""Feature extraction with optional disk caching."""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from .datasets.base import SpectrogramDataset
from .models.base import FeatureExtractor

_CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / "cache" / "features"


def _cache_path(model: FeatureExtractor, dataset: SpectrogramDataset) -> Path:
    return _CACHE_DIR / f"{model.name}__{dataset.name}.npz"


def extract_and_cache(
    model: FeatureExtractor,
    dataset: SpectrogramDataset,
    device: torch.device,
    batch_size: int = 16,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for every item in *dataset* using *model*.

    Results are stored as a .npz file keyed by model + dataset name.
    Subsequent calls with use_cache=True skip extraction entirely.

    Returns:
        features: Float32 array of shape (N, embed_dim).
        labels:   Int64 array of shape (N,).
    """
    path = _cache_path(model, dataset)

    if use_cache and path.exists():
        logger.info(f"Loading cached features: {path.name}")
        data = np.load(path)
        return data["features"], data["labels"]

    items, labels = dataset.items()
    n = len(items)
    all_features: List[np.ndarray] = []

    logger.info(
        f"Extracting — model={model.name}  dataset={dataset.name}  "
        f"n={n}  batch_size={batch_size}"
    )

    for start in tqdm(range(0, n, batch_size), desc=f"{model.name}/{dataset.name}"):
        batch_items = items[start : start + batch_size]
        tensors = [dataset.load_item(item) for item in batch_items]
        x = torch.stack(tensors)          # (B, 1, F, T)
        feats = model.extract(x)          # (B, D)
        all_features.append(feats)

    features = np.concatenate(all_features, axis=0).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features, labels=labels)
    logger.info(f"Cached → {path}")

    return features, labels
