"""Cross-validation evaluation pipeline."""
from typing import Any, Dict, List

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score

from .datasets.base import SpectrogramDataset
from .features import extract_and_cache
from .models.base import FeatureExtractor
from .probes.base import Probe


def run_evaluation(
    model: FeatureExtractor,
    dataset: SpectrogramDataset,
    probes: List[Probe],
    device: torch.device,
    batch_size: int = 16,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run cross-validation for all probes and return a results dict.

    The returned dict is JSON-serialisable::

        {
          "model":     "dinov3_vits16",
          "dataset":   "glasgow",
          "n_samples": 2081,
          "n_classes": 6,
          "probes": {
            "knn_k10": {
              "fold_accs": [0.84, 0.86, ...],
              "mean_acc":  0.8467,
              "std_acc":   0.0136,
            },
            ...
          }
        }
    """
    model.to(device)
    features, labels = extract_and_cache(
        model, dataset, device, batch_size=batch_size, use_cache=use_cache
    )

    results: Dict[str, Any] = {
        "model": model.name,
        "dataset": dataset.name,
        "n_samples": int(len(labels)),
        "n_classes": dataset.n_classes,
        "probes": {},
    }

    for probe in probes:
        fold_accs: List[float] = []
        fold_f1s:  List[float] = []
        for fold_idx, (train_idx, test_idx) in enumerate(dataset.cv_splits()):
            X_train, y_train = features[train_idx], labels[train_idx]
            X_test, y_test   = features[test_idx],  labels[test_idx]

            probe.fit(X_train, y_train)
            y_pred = probe.predict(X_test)
            acc = float((y_pred == y_test).mean())
            f1  = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            fold_accs.append(acc)
            fold_f1s.append(f1)
            logger.debug(
                f"  {probe.name} fold {fold_idx + 1}: "
                f"acc={acc * 100:.2f}%  macro-F1={f1 * 100:.2f}%"
            )

        mean_acc = float(np.mean(fold_accs))
        std_acc  = float(np.std(fold_accs))
        mean_f1  = float(np.mean(fold_f1s))
        std_f1   = float(np.std(fold_f1s))
        logger.info(
            f"[{model.name}] [{dataset.name}] {probe.name}: "
            f"acc={mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%  "
            f"macro-F1={mean_f1 * 100:.2f}% ± {std_f1 * 100:.2f}%"
        )
        results["probes"][probe.name] = {
            "fold_accs": [float(a) for a in fold_accs],
            "fold_f1s":  [float(f) for f in fold_f1s],
            "mean_acc":  mean_acc,
            "std_acc":   std_acc,
            "mean_f1":   mean_f1,
            "std_f1":    std_f1,
        }

    return results
