"""Few-shot sample-efficiency evaluation on pre-extracted features.

For each n_shots value, the training split (within each CV fold) is
subsampled to exactly n_shots examples per class.  The probe is fitted
on that subset and evaluated on the full held-out test fold.  This is
repeated n_repeats times (with different random seeds) and the results
are aggregated across both folds and repeats.

A special "full" entry (n_shots=None) uses the entire training split and
gives the same numbers as run_evaluation().
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics import f1_score

from .datasets.base import SpectrogramDataset
from .features import extract_and_cache
from .models.base import FeatureExtractor
from .probes.base import Probe
from .probes.knn import KNNProbe
from .probes.linear import LinearProbe


# ── Probe cloning ─────────────────────────────────────────────────────────────

def _clone_probe(probe: Probe, n_train: int) -> Probe:
    """Return a fresh (unfitted) copy of *probe*, capping k for KNN."""
    if isinstance(probe, KNNProbe):
        return KNNProbe(k=min(probe.k, n_train), metric=probe.metric)
    if isinstance(probe, LinearProbe):
        return LinearProbe(C=probe.C, max_iter=probe.max_iter)
    raise TypeError(f"Cannot clone probe of type {type(probe).__name__}")


# ── Core evaluation ───────────────────────────────────────────────────────────

def run_few_shot_evaluation(
    model: FeatureExtractor,
    dataset: SpectrogramDataset,
    probes: List[Probe],
    device,
    batch_size: int = 16,
    use_cache: bool = True,
    n_shots_list: Sequence[int] = (1, 2, 5, 10, 20, 50, 100, 200),
    n_repeats: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run few-shot evaluation and return a JSON-serialisable results dict.

    Result structure::

        {
          "model":    "dinov3_vits16",
          "dataset":  "glasgow",
          "n_classes": 6,
          "n_folds":   5,
          "n_repeats": 10,
          "probes": {
            "knn_k10": [
              {
                "n_shots":       1,
                "n_total_train": 6,       # n_shots * n_classes
                "mean_acc":  0.52, "std_acc":  0.05,
                "mean_f1":   0.49, "std_f1":   0.05,
              },
              ...
              {
                "n_shots":       null,    # "full" – all training data
                "n_total_train": 1664,
                ...
              },
            ],
          }
        }
    """
    import torch

    model.to(device)
    features, labels = extract_and_cache(
        model, dataset, device, batch_size=batch_size, use_cache=use_cache
    )

    classes = np.unique(labels)
    n_classes = int(len(classes))
    folds = list(dataset.cv_splits())
    n_folds = len(folds)

    # Build the list of n_shots to evaluate, clamped to actual train-split size.
    # "None" means use the entire training split ("full" baseline).
    shots_to_run: List[Optional[int]] = list(n_shots_list) + [None]

    results: Dict[str, Any] = {
        "model":     model.name,
        "dataset":   dataset.name,
        "n_classes": n_classes,
        "n_folds":   n_folds,
        "n_repeats": n_repeats,
        "probes":    {},
    }

    for probe in probes:
        # raw_scores[n_shots] = list of (acc, f1) tuples
        raw_scores: Dict[Optional[int], List[Tuple[float, float]]] = {
            n: [] for n in shots_to_run
        }

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, y_train = features[train_idx], labels[train_idx]
            X_test,  y_test  = features[test_idx],  labels[test_idx]

            for n_shots in shots_to_run:
                repeats = 1 if n_shots is None else n_repeats
                for rep in range(repeats):
                    if n_shots is None:
                        sub_idx = np.arange(len(X_train))
                    else:
                        rng = np.random.default_rng(seed * 10_000 + fold_idx * 1_000 + rep)
                        sub_idx = _stratified_sample(y_train, classes, n_shots, rng)

                    X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]
                    p = _clone_probe(probe, len(sub_idx))
                    p.fit(X_sub, y_sub)
                    y_pred = p.predict(X_test)

                    acc = float((y_pred == y_test).mean())
                    f1  = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
                    raw_scores[n_shots].append((acc, f1))

            logger.debug(
                f"  [{probe.name}] fold {fold_idx + 1}/{n_folds} done "
                f"({len(shots_to_run)} shot levels × {n_repeats} repeats)"
            )

        # Aggregate across folds × repeats
        probe_data: List[Dict[str, Any]] = []
        for n_shots in shots_to_run:
            scores = raw_scores[n_shots]
            accs = [s[0] for s in scores]
            f1s  = [s[1] for s in scores]

            if n_shots is None:
                # actual mean train-split size across folds
                n_total = int(np.mean([len(t) for t, _ in folds]))
            else:
                n_total = n_shots * n_classes

            entry = {
                "n_shots":       n_shots,
                "n_total_train": n_total,
                "mean_acc":  float(np.mean(accs)),
                "std_acc":   float(np.std(accs)),
                "mean_f1":   float(np.mean(f1s)),
                "std_f1":    float(np.std(f1s)),
            }
            probe_data.append(entry)

            tag = "full" if n_shots is None else f"{n_shots}-shot"
            logger.info(
                f"[{model.name}] [{dataset.name}] {probe.name} {tag}: "
                f"acc={entry['mean_acc']*100:.2f}% ± {entry['std_acc']*100:.2f}%  "
                f"F1={entry['mean_f1']*100:.2f}% ± {entry['std_f1']*100:.2f}%"
            )

        results["probes"][probe.name] = probe_data

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stratified_sample(
    y: np.ndarray,
    classes: np.ndarray,
    n_shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return indices: up to n_shots examples per class."""
    sub_idx: List[int] = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        n = min(n_shots, len(cls_idx))
        sub_idx.extend(rng.choice(cls_idx, n, replace=False).tolist())
    return np.array(sub_idx)
