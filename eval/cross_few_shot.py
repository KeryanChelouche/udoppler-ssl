"""Cross-dataset few-shot evaluation with a fine-tuned backbone.

Fine-tune a supervised model on a source dataset, discard the
classification head, then run few-shot probing on a target dataset
using the frozen backbone features.

This lets us compare:
  - SSL models: pretrained backbone → target few-shot
  - Supervised: ImageNet → source fine-tune → target few-shot
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cross_evaluation import _apply_remap, _remap_dataset_labels, find_label_scheme
from .datasets.base import SpectrogramDataset
from .few_shot import _clone_probe, _stratified_sample
from .probes.base import Probe
from .supervised import _FoldDataset, _train_one_epoch


@torch.inference_mode()
def _extract_backbone_features(
    model: nn.Module,
    dataset: SpectrogramDataset,
    preprocess_fn: Callable[[torch.Tensor, torch.device], torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward all samples through the model and return (features, labels)."""
    items, labels = dataset.items()
    indices = np.arange(len(items))
    fold_ds = _FoldDataset(dataset, indices, items, labels)
    loader = DataLoader(fold_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    all_feats = []
    for x, _ in tqdm(loader, desc="extract features", leave=False):
        x = preprocess_fn(x, device)
        feats = model(x)            # (B, D) with fc=Identity
        all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats), labels


def run_cross_few_shot_evaluation(
    model_name: str,
    model_factory: Callable[[int], nn.Module],
    preprocess_fn: Callable[[torch.Tensor, torch.device], torch.Tensor],
    target_dataset: SpectrogramDataset,
    probes: List[Probe],
    device: torch.device,
    source_dataset: Optional[SpectrogramDataset] = None,
    batch_size: int = 16,
    lr: float = 1e-4,
    epochs: int = 30,
    weight_decay: float = 1e-4,
    n_shots_list: Sequence[int] = (1, 2, 5, 10, 20, 50, 100, 200),
    n_repeats: int = 10,
    seed: int = 42,
    checkpoint_dir: Optional[Path] = None,
    scheme: Optional[str] = None,
    freeze_backbone: bool = False,
) -> Dict[str, Any]:
    """Few-shot probe on target, optionally after fine-tuning on source.

    If *source_dataset* is None, the model uses ImageNet weights only
    (no fine-tuning).  Otherwise it is fine-tuned on the full source
    dataset before feature extraction.

    If *freeze_backbone* is True, the fine-tuning step is skipped even
    when *source_dataset* is given — the source is used only to pick
    the label scheme and remap target labels.

    Returns a dict compatible with ``run_few_shot_evaluation()`` output.
    """
    # ── 1. Determine label scheme and n_classes ──────────────────────
    if source_dataset is not None:
        scheme_name, scheme_def = find_label_scheme(
            source_dataset.name, target_dataset.name, scheme,
        )
        remaps = scheme_def["dataset_remaps"]
        n_classes = len(scheme_def["class_names"])
        logger.info(
            f"Cross few-shot: source={source_dataset.name} → "
            f"target={target_dataset.name}  scheme={scheme_name} ({n_classes} classes)"
        )
    else:
        # No source — use target's own label space.
        n_classes = target_dataset.n_classes
        remaps = None
        scheme_name = None
        logger.info(
            f"Few-shot (ImageNet backbone): "
            f"target={target_dataset.name} ({n_classes} classes)"
        )

    # ── 2. Fine-tune on source (or skip) ─────────────────────────────
    model = model_factory(n_classes).to(device)

    if freeze_backbone:
        logger.info("  Frozen backbone — skipping fine-tuning")
    elif source_dataset is not None:
        # Check for cached checkpoint.
        ckpt_path = None
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = (
                checkpoint_dir
                / f"{model_name}_ft_{source_dataset.name}_{scheme_name}_ep{epochs}.pt"
            )

        if ckpt_path is not None and ckpt_path.exists():
            logger.info(f"  Loading cached backbone: {ckpt_path}")
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True)
            )
        else:
            src_items, src_labels = source_dataset.items()
            src_indices, src_labels_remapped = _remap_dataset_labels(
                src_labels, remaps[source_dataset.name],
            )
            logger.info(f"  Source: {len(src_indices)} samples for fine-tuning")

            src_ds = _FoldDataset(
                source_dataset, src_indices, src_items, src_labels_remapped,
            )
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

            if ckpt_path is not None:
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"  Saved fine-tuned backbone → {ckpt_path}")

    # ── 3. Strip head, extract target features ───────────────────────
    model.fc = nn.Identity()

    target_feats, target_labels = _extract_backbone_features(
        model, target_dataset, preprocess_fn, device, batch_size,
    )

    del model
    torch.cuda.empty_cache()

    # ── 4. Remap target labels (if cross-dataset) ────────────────────
    target_groups = target_dataset.groups  # None for datasets without groups
    if remaps is not None:
        remap = remaps[target_dataset.name]
        # Filter groups with the same mask _apply_remap will use.
        if remap is not None and target_groups is not None:
            mask = np.isin(target_labels, list(remap.keys()))
            target_groups = target_groups[mask]
        target_feats, target_labels = _apply_remap(
            target_feats, target_labels, remap,
        )
    logger.info(f"  Target: {len(target_labels)} samples for few-shot eval")

    # ── 5. Few-shot protocol on target features ──────────────────────
    classes = np.unique(target_labels)
    if target_groups is not None:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(sgkf.split(target_feats, target_labels, target_groups))
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(target_feats, target_labels))
    n_folds = len(folds)

    shots_to_run: List[Optional[int]] = list(n_shots_list) + [None]

    if freeze_backbone:
        suffix = source_dataset.name if source_dataset else "imagenet"
        display_name = f"{model_name}_frozen_{suffix}"
    elif source_dataset is not None:
        display_name = f"{model_name}_ft_{source_dataset.name}"
    else:
        display_name = f"{model_name}_imagenet"

    results: Dict[str, Any] = {
        "model": display_name,
        "source_dataset": source_dataset.name if source_dataset else None,
        "dataset": target_dataset.name,
        "label_scheme": scheme_name,
        "n_classes": int(len(classes)),
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "probes": {},
    }

    for probe in probes:
        # (acc, f1, fold_idx, rep) per probe fit. fold/rep are kept so that
        # downstream paired significance tests can align fits across methods:
        # the draw RNG is seeded by (seed, fold_idx, rep) only — independent
        # of the fine-tuning seed — so fit k of method A and fit k of method B
        # saw the *same* support set and the same test fold.
        raw_scores: Dict[Optional[int], List[Tuple[float, float, int, int]]] = {
            n: [] for n in shots_to_run
        }

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, y_train = target_feats[train_idx], target_labels[train_idx]
            X_test, y_test = target_feats[test_idx], target_labels[test_idx]

            for n_shots in shots_to_run:
                repeats = 1 if n_shots is None else n_repeats
                for rep in range(repeats):
                    if n_shots is None:
                        sub_idx = np.arange(len(X_train))
                    else:
                        rng = np.random.default_rng(
                            seed * 10_000 + fold_idx * 1_000 + rep,
                        )
                        sub_idx = _stratified_sample(y_train, classes, n_shots, rng)

                    X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]
                    p = _clone_probe(probe, len(sub_idx))
                    p.fit(X_sub, y_sub)
                    y_pred = p.predict(X_test)

                    acc = float((y_pred == y_test).mean())
                    # Macro-F1 over classes present in y_test (see
                    # eval/cross_evaluation.py for rationale).
                    f1 = float(f1_score(
                        y_test, y_pred,
                        labels=sorted(set(y_test.tolist())),
                        average="macro", zero_division=0,
                    ))
                    raw_scores[n_shots].append((acc, f1, fold_idx, rep))

            logger.debug(
                f"  [{probe.name}] fold {fold_idx + 1}/{n_folds} done "
                f"({len(shots_to_run)} shot levels × {n_repeats} repeats)"
            )

        # Aggregate
        probe_data: List[Dict[str, Any]] = []
        for n_shots in shots_to_run:
            scores = raw_scores[n_shots]
            accs = [s[0] for s in scores]
            f1s = [s[1] for s in scores]
            n_total = (
                int(np.mean([len(t) for t, _ in folds]))
                if n_shots is None
                # Nominal support = n_shots per class actually present in the
                # target (classes = np.unique(target_labels)), not the scheme's
                # label count: the few-shot probe only ever draws/fits on
                # present classes, so an absent class (e.g. Falling in the
                # older Glasgow cohort) must not inflate this count.
                else n_shots * len(classes)
            )
            entry = {
                "n_shots": n_shots,
                "n_total_train": n_total,
                "mean_acc": float(np.mean(accs)),
                "std_acc": float(np.std(accs)),
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s)),
                # Per-fit scores, in (fold, rep) order. Retained so paired
                # tests can be run post hoc without re-fitting: all methods
                # share the same folds and the same support draws.
                "fits": {
                    "fold": [int(s[2]) for s in scores],
                    "rep": [int(s[3]) for s in scores],
                    "acc": [round(float(s[0]), 6) for s in scores],
                    "f1": [round(float(s[1]), 6) for s in scores],
                },
            }
            probe_data.append(entry)
            tag = "full" if n_shots is None else f"{n_shots}-shot"
            logger.info(
                f"[{display_name}] [{target_dataset.name}] {probe.name} {tag}: "
                f"acc={entry['mean_acc']*100:.2f}% ± {entry['std_acc']*100:.2f}%  "
                f"F1={entry['mean_f1']*100:.2f}% ± {entry['std_f1']*100:.2f}%"
            )

        results["probes"][probe.name] = probe_data

    return results
