"""Cross-dataset evaluation pipeline.

Train a probe on features from one dataset and evaluate on another,
with label remapping to a shared label space.
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets.base import SpectrogramDataset
from .features import extract_and_cache
from .models.base import FeatureExtractor
from .probes.base import Probe
from .reporting import plot_confusion_matrix
from .supervised import _FoldDataset, _evaluate, _train_one_epoch


# ── Label alignment schemes ─────────────────────────────────────────────────
#
# Each scheme defines a shared label space and per-dataset remappings.
# A remap dict maps {original_label: shared_label}; labels absent from
# the dict are dropped.  None means identity (keep all labels as-is).

# MAD-10 class indices: 0 Walk, 1 Stand, 2 Sit, 3 Up Stairs, 4 Down Stairs,
#                       5 Pick, 6 Step Over, 7 Semi-Turn Around, 8 Fall,
#                       9 Walk (Arbitrary).
# Glasgow native indices: 0 Walking, 1 Sitting down, 2 Standing up,
#                         3 Picking up, 4 Drinking, 5 Falling.
# Glasgow_5 indices (Drinking removed): 0 Walking, 1 Sitting down,
#                                       2 Standing up, 3 Picking up, 4 Falling.

# 5-class subset → MAD-10 indexing.
_FIVE_TO_MAD10 = {0: 0, 1: 1, 2: 2, 3: 5, 4: 8}
# Glasgow native (6 cls) → MAD-10 (drops Drinking).
_GLASGOW6_TO_MAD10 = {0: 0, 1: 2, 2: 1, 3: 5, 5: 8}
# Glasgow_5 (5 cls) → MAD-10.
_GLASGOW5_TO_MAD10 = {0: 0, 1: 2, 2: 1, 3: 5, 4: 8}
# MAD-5 / mad_5_sub* → Glasgow native indexing.
_MAD5_TO_GLASGOW6 = {0: 0, 1: 2, 2: 1, 3: 3, 4: 5}
# Glasgow_5 → Glasgow native indexing.
_GLASGOW5_TO_GLASGOW6 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}


CROSS_LABEL_SCHEMES: Dict[str, Dict[str, Any]] = {
    "har5": {
        "class_names": ["Walk", "Stand", "Sit", "Pick", "Fall"],
        # 5-class space — sources and targets are both 5-class datasets.
        # For 5-class glasgow FT, use the glasgow_5 dataset directly (the
        # raw glasgow source belongs in glasgow6).
        "dataset_remaps": {
            "glasgow_5":    None,   # already 0-4 after exclude_classes
            "mad_5":       None,   # already 0-4 in shared order
            "mad_5_sub1":  None,
            "mad_5_sub2":  None,
            "mad_5_sub3":  None,
            "mad_5_sub12": None,
            "mad_5_sub23": None,
        },
    },
    "mad10": {
        "class_names": [
            "Walk", "Stand", "Sit", "Up Stairs", "Down Stairs",
            "Pick", "Step Over", "Semi-Turn Around", "Fall", "Walk (Arbitrary)",
        ],
        # 10-class MAD as source. 5-class datasets appear as targets only:
        # the model is FT'd on full MAD then evaluated against samples
        # whose true label is in a 5-class subset (open-prediction).
        "dataset_remaps": {
            "mad":          None,
            "mad_sub1":     None,
            "mad_sub2":     None,
            "mad_sub3":     None,
            "mad_sub12":    None,
            "mad_sub23":    None,
            "mad_5":         _FIVE_TO_MAD10,
            "mad_5_sub1":    _FIVE_TO_MAD10,
            "mad_5_sub2":    _FIVE_TO_MAD10,
            "mad_5_sub3":    _FIVE_TO_MAD10,
            "mad_5_sub12":   _FIVE_TO_MAD10,
            "mad_5_sub23":   _FIVE_TO_MAD10,
            "glasgow_5":     _GLASGOW5_TO_MAD10,
        },
    },
    "glasgow6": {
        "class_names": [
            "Walking", "Sitting down", "Standing up",
            "Picking up object", "Drinking", "Falling",
        ],
        "dataset_remaps": {
            # Native 6-class source.
            "glasgow":       None,
            # Cohort splits by acquisition campaign — same 6-class label space.
            "glasgow_young":  None,
            "glasgow_mature": None,
            # Glasgow_5 in native indexing (Drinking absent → no class 4).
            "glasgow_5":     _GLASGOW5_TO_GLASGOW6,
            # MAD-5 targets remapped into Glasgow's 6-class space.
            "mad_5":         _MAD5_TO_GLASGOW6,
            "mad_5_sub1":    _MAD5_TO_GLASGOW6,
            "mad_5_sub2":    _MAD5_TO_GLASGOW6,
            "mad_5_sub3":    _MAD5_TO_GLASGOW6,
            "mad_5_sub12":   _MAD5_TO_GLASGOW6,
            "mad_5_sub23":   _MAD5_TO_GLASGOW6,
        },
    },
}


def find_label_scheme(
    train_name: str,
    test_name: str,
    scheme: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Find a label scheme that covers both datasets.

    If *scheme* is given, it is used directly (and validated).  Otherwise
    the first registered scheme containing both datasets is returned.

    Raises ValueError if no scheme matches.
    """
    if scheme is not None:
        if scheme not in CROSS_LABEL_SCHEMES:
            raise ValueError(
                f"Unknown scheme '{scheme}'. "
                f"Available: {list(CROSS_LABEL_SCHEMES)}"
            )
        s = CROSS_LABEL_SCHEMES[scheme]
        remaps = s["dataset_remaps"]
        if train_name not in remaps or test_name not in remaps:
            raise ValueError(
                f"Scheme '{scheme}' does not cover "
                f"'{train_name}' ↔ '{test_name}'. "
                f"Members: {list(remaps)}"
            )
        return scheme, s

    matches = [
        name for name, s in CROSS_LABEL_SCHEMES.items()
        if train_name in s["dataset_remaps"] and test_name in s["dataset_remaps"]
    ]
    if not matches:
        raise ValueError(
            f"No label scheme found for '{train_name}' ↔ '{test_name}'. "
            f"Register one in CROSS_LABEL_SCHEMES."
        )
    if len(matches) > 1:
        logger.debug(
            f"Multiple schemes match {train_name} ↔ {test_name}: {matches}. "
            f"Using '{matches[0]}'. Pass scheme=... to override."
        )
    name = matches[0]
    return name, CROSS_LABEL_SCHEMES[name]


def _apply_remap(
    features: np.ndarray,
    labels: np.ndarray,
    remap: Optional[Dict[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter samples and remap labels.  None remap means identity."""
    if remap is None:
        return features, labels
    mask = np.isin(labels, list(remap.keys()))
    features = features[mask]
    labels = np.array([remap[int(l)] for l in labels[mask]], dtype=np.int64)
    return features, labels


def _remap_dataset_labels(
    labels: np.ndarray,
    remap: Optional[Dict[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices_to_keep, remapped_labels_array).

    The returned labels array has the same length as the original;
    only elements at the returned indices are valid.
    """
    if remap is None:
        return np.arange(len(labels)), labels
    mask = np.isin(labels, list(remap.keys()))
    indices = np.where(mask)[0]
    remapped = labels.copy()
    for old, new in remap.items():
        remapped[labels == old] = new
    return indices, remapped


def run_cross_evaluation(
    model: FeatureExtractor,
    train_dataset: SpectrogramDataset,
    test_dataset: SpectrogramDataset,
    probes: List[Probe],
    device: torch.device,
    batch_size: int = 16,
    use_cache: bool = True,
    center_per_dataset: bool = False,
    scheme: Optional[str] = None,
) -> Dict[str, Any]:
    """Train probes on one dataset's features and evaluate on another.

    Label alignment is looked up automatically from CROSS_LABEL_SCHEMES,
    or forced via *scheme*.

    Args:
        center_per_dataset: If True, subtract each dataset's own feature
            mean before probing.  This removes first-order distribution
            shift between the two domains.
        scheme: Force a specific entry from CROSS_LABEL_SCHEMES instead of
            auto-detecting.  Required when more than one scheme covers
            the (train, test) pair and the default isn't desired.

    Returns a JSON-serialisable results dict.
    """
    scheme_name, scheme_def = find_label_scheme(
        train_dataset.name, test_dataset.name, scheme,
    )
    remaps = scheme_def["dataset_remaps"]
    class_names = scheme_def["class_names"]
    n_classes = len(class_names)

    logger.info(
        f"Cross-eval: train={train_dataset.name} → test={test_dataset.name}  "
        f"scheme={scheme_name} ({n_classes} classes)"
        f"{'  [center-per-dataset]' if center_per_dataset else ''}"
    )

    # Extract features from both datasets (cached independently).
    model.to(device)
    train_feats, train_labels = extract_and_cache(
        model, train_dataset, device,
        batch_size=batch_size, use_cache=use_cache,
    )
    test_feats, test_labels = extract_and_cache(
        model, test_dataset, device,
        batch_size=batch_size, use_cache=use_cache,
    )

    # Apply label remapping / filtering.
    train_feats, train_labels = _apply_remap(
        train_feats, train_labels, remaps[train_dataset.name],
    )
    test_feats, test_labels = _apply_remap(
        test_feats, test_labels, remaps[test_dataset.name],
    )

    # Per-dataset centering: subtract each domain's own mean.
    if center_per_dataset:
        train_feats = train_feats - train_feats.mean(axis=0)
        test_feats = test_feats - test_feats.mean(axis=0)

    logger.info(
        f"  Train: {len(train_labels)} samples  |  "
        f"Test: {len(test_labels)} samples"
    )

    results: Dict[str, Any] = {
        "model": model.name,
        "train_dataset": train_dataset.name,
        "test_dataset": test_dataset.name,
        "label_scheme": scheme_name,
        "n_train": int(len(train_labels)),
        "n_test": int(len(test_labels)),
        "n_classes": n_classes,
        "class_names": class_names,
        "probes": {},
    }

    for probe in probes:
        probe.fit(train_feats, train_labels)
        y_pred = probe.predict(test_feats)
        acc = float((y_pred == test_labels).mean())
        # Macro-F1 averaged over classes actually present in y_true.
        # Avoids the sklearn-default behaviour of including a class in
        # the average iff it appears in y_pred — which would penalise
        # models that emit false positives on classes the test set
        # doesn't contain (those FPs already cost recall on the true
        # class; double-counting them is unfair across models).
        present_labels = sorted(set(test_labels.tolist()))
        f1 = float(f1_score(
            test_labels, y_pred,
            labels=present_labels, average="macro", zero_division=0,
        ))
        logger.info(
            f"  {probe.name}: acc={acc * 100:.2f}%  macro-F1={f1 * 100:.2f}%"
        )
        results["probes"][probe.name] = {"acc": acc, "f1": f1}

    return results


# ── Supervised cross-evaluation ─────────────────────────────────────────────


def _fine_tune_on_source(
    model: nn.Module,
    model_name: str,
    preprocess_fn: Callable[[torch.Tensor, torch.device], torch.Tensor],
    source_dataset: SpectrogramDataset,
    source_remap: Optional[Dict[int, int]],
    device: torch.device,
    batch_size: int,
    lr: float,
    epochs: int,
    weight_decay: float,
) -> int:
    """Fine-tune *model* in place on the (remapped) source dataset.

    Returns the number of source samples used.
    """
    src_items, src_labels = source_dataset.items()
    src_idx, src_labels_remapped = _remap_dataset_labels(src_labels, source_remap)

    src_ds = _FoldDataset(source_dataset, src_idx, src_items, src_labels_remapped)
    src_loader = DataLoader(
        src_ds, batch_size=batch_size, shuffle=True, num_workers=4,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc=f"{model_name} fine-tune", leave=False):
        loss = _train_one_epoch(
            model, src_loader, optimizer, criterion, preprocess_fn, device,
        )
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            logger.debug(
                f"  [{model_name}] epoch {epoch + 1}/{epochs}  loss={loss:.4f}"
            )

    del optimizer, scheduler
    torch.cuda.empty_cache()
    return int(len(src_idx))


def run_supervised_cross_evaluation(
    model_name: str,
    model_factory: Callable[[int], nn.Module],
    preprocess_fn: Callable[[torch.Tensor, torch.device], torch.Tensor],
    train_dataset: SpectrogramDataset,
    test_dataset: SpectrogramDataset,
    device: torch.device,
    batch_size: int = 16,
    lr: float = 1e-4,
    epochs: int = 30,
    weight_decay: float = 1e-4,
    checkpoint_dir: Optional[Path] = None,
    scheme: Optional[str] = None,
) -> Dict[str, Any]:
    """Fine-tune on source, evaluate on test with the trained head intact.

    The classification head trained on the source dataset is used directly
    on the test dataset — no head reset, no linear probe.  Source and test
    labels share a common space via the registered label scheme (auto-
    detected, or forced via *scheme*).

    If *checkpoint_dir* is given, the fine-tuned weights are cached as
    ``{model_name}_ft_{source}_{scheme}_ep{epochs}.pt`` and reused on
    subsequent runs instead of retraining.
    """
    scheme_name, scheme_def = find_label_scheme(
        train_dataset.name, test_dataset.name, scheme,
    )
    remaps = scheme_def["dataset_remaps"]
    class_names = scheme_def["class_names"]
    n_classes = len(class_names)

    logger.info(
        f"Supervised cross-eval: train={train_dataset.name} → "
        f"test={test_dataset.name}  scheme={scheme_name} ({n_classes} classes)"
    )

    test_items, test_labels = test_dataset.items()
    test_idx, test_labels_remapped = _remap_dataset_labels(
        test_labels, remaps[test_dataset.name],
    )
    test_ds = _FoldDataset(
        test_dataset, test_idx, test_items, test_labels_remapped,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4,
    )

    model = model_factory(n_classes).to(device)

    ckpt_path: Optional[Path] = None
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = (
            checkpoint_dir
            / f"{model_name}_ft_{train_dataset.name}_{scheme_name}_ep{epochs}.pt"
        )

    n_train: int
    if ckpt_path is not None and ckpt_path.exists():
        logger.info(f"  Loading fine-tuned checkpoint: {ckpt_path}")
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        # Recover sample count for reporting without retraining.
        _, src_labels = train_dataset.items()
        src_idx, _ = _remap_dataset_labels(
            src_labels, remaps[train_dataset.name],
        )
        n_train = int(len(src_idx))
    else:
        n_train = _fine_tune_on_source(
            model=model,
            model_name=model_name,
            preprocess_fn=preprocess_fn,
            source_dataset=train_dataset,
            source_remap=remaps[train_dataset.name],
            device=device,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            weight_decay=weight_decay,
        )
        if ckpt_path is not None:
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  Saved fine-tuned checkpoint → {ckpt_path}")

    logger.info(
        f"  Train: {n_train} samples  |  Test: {len(test_idx)} samples"
    )

    y_true, y_pred = _evaluate(model, test_loader, preprocess_fn, device)
    acc = float((y_pred == y_true).mean())
    # Macro-F1 averaged over classes present in y_true (see comment in
    # run_cross_evaluation for rationale).
    present_labels = sorted(set(y_true.tolist()))
    f1 = float(f1_score(
        y_true, y_pred,
        labels=present_labels, average="macro", zero_division=0,
    ))
    logger.info(
        f"  supervised: acc={acc * 100:.2f}%  macro-F1={f1 * 100:.2f}%"
    )

    pred_classes, pred_counts = np.unique(y_pred, return_counts=True)
    true_classes, true_counts = np.unique(y_true, return_counts=True)
    logger.info(
        "  pred dist: "
        + ", ".join(
            f"{class_names[c]}={n}" for c, n in zip(pred_classes, pred_counts)
        )
    )
    logger.info(
        "  true dist: "
        + ", ".join(
            f"{class_names[c]}={n}" for c, n in zip(true_classes, true_counts)
        )
    )

    cm_path = (
        Path("results/plots")
        / f"{model_name}__{train_dataset.name}_to_{test_dataset.name}"
          f"__{scheme_name}_cm.png"
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        f"{model_name} / {train_dataset.name} → {test_dataset.name} ({scheme_name})",
        cm_path,
        labels=list(range(n_classes)),
    )
    logger.info(f"  Confusion matrix → {cm_path}")

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "train_dataset": train_dataset.name,
        "test_dataset": test_dataset.name,
        "label_scheme": scheme_name,
        "n_train": n_train,
        "n_test": int(len(test_idx)),
        "n_classes": n_classes,
        "class_names": class_names,
        "probes": {
            "supervised": {"acc": acc, "f1": f1},
        },
    }
