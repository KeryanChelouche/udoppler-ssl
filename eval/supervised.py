"""End-to-end supervised training and evaluation (e.g. fine-tuned ResNet).

Unlike the frozen-features pipeline, this module trains the full model
on each CV fold from scratch and evaluates on the held-out split.
Results are returned in the same dict format as ``run_evaluation()``
so that reporting and comparison tables work unchanged.
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .datasets.base import SpectrogramDataset
from .reporting import plot_confusion_matrix


# ── Dataset adapter ──────────────────────────────────────────────────────────

class _FoldDataset(Dataset):
    """Wraps a SpectrogramDataset for a specific set of indices."""

    def __init__(
        self,
        dataset: SpectrogramDataset,
        indices: np.ndarray,
        items: List[Path],
        labels: np.ndarray,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.items = items
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        x = self.dataset.load_item(self.items[real_idx])   # (1, F, T)
        y = int(self.labels[real_idx])
        return x, y


# ── Training helpers ─────────────────────────────────────────────────────────

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    preprocess_fn: Callable[[torch.Tensor, torch.device], torch.Tensor],
    device: torch.device,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x = preprocess_fn(x, device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    preprocess_fn: Callable[[torch.Tensor, torch.device], torch.Tensor],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) arrays."""
    model.eval()
    all_true, all_pred = [], []
    for x, y in loader:
        x = preprocess_fn(x, device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_true.append(y.numpy())
        all_pred.append(preds)
    return np.concatenate(all_true), np.concatenate(all_pred)


# ── Public API ───────────────────────────────────────────────────────────────

def run_supervised_evaluation(
    model_name: str,
    model_factory: Callable[[int], nn.Module],
    preprocess_fn: Callable[[torch.Tensor, torch.device], torch.Tensor],
    dataset: SpectrogramDataset,
    device: torch.device,
    batch_size: int = 16,
    lr: float = 1e-4,
    epochs: int = 30,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    """Fine-tune a model on each CV fold and return results.

    Args:
        model_name:     Identifier for results/caching.
        model_factory:  ``fn(n_classes) -> nn.Module`` — called fresh per fold.
        preprocess_fn:  ``fn(batch_tensor, device) -> preprocessed_tensor``.
        dataset:        A SpectrogramDataset.
        device:         Torch device.
        batch_size:     Batch size for train and eval.
        lr:             Learning rate for AdamW.
        epochs:         Training epochs per fold.
        weight_decay:   AdamW weight decay.

    Returns:
        Dict in the same shape as ``run_evaluation()`` output, with
        a single probe entry named ``"supervised"``.
    """
    items, labels = dataset.items()
    criterion = nn.CrossEntropyLoss()

    fold_accs: List[float] = []
    fold_f1s:  List[float] = []
    all_y_true = np.empty(len(labels), dtype=np.int64)
    all_y_pred = np.empty(len(labels), dtype=np.int64)

    for fold_idx, (train_idx, test_idx) in enumerate(dataset.cv_splits()):
        model = model_factory(dataset.n_classes).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs,
        )

        train_ds = _FoldDataset(dataset, train_idx, items, labels)
        test_ds  = _FoldDataset(dataset, test_idx,  items, labels)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=4,
        )

        desc = f"fold {fold_idx + 1}"
        for epoch in tqdm(range(epochs), desc=desc, leave=False):
            loss = _train_one_epoch(
                model, train_loader, optimizer, criterion, preprocess_fn, device,
            )
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                logger.debug(
                    f"  [{model_name}] fold {fold_idx + 1} "
                    f"epoch {epoch + 1}/{epochs}  loss={loss:.4f}"
                )

        y_true, y_pred = _evaluate(model, test_loader, preprocess_fn, device)
        all_y_true[test_idx] = y_true
        all_y_pred[test_idx] = y_pred
        acc = float((y_pred == y_true).mean())
        f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        fold_accs.append(acc)
        fold_f1s.append(f1)
        logger.info(
            f"  [{model_name}] fold {fold_idx + 1}: "
            f"acc={acc * 100:.2f}%  macro-F1={f1 * 100:.2f}%"
        )

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    mean_f1  = float(np.mean(fold_f1s))
    std_f1   = float(np.std(fold_f1s))

    logger.info(
        f"[{model_name}] [{dataset.name}] supervised: "
        f"acc={mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%  "
        f"macro-F1={mean_f1 * 100:.2f}% ± {std_f1 * 100:.2f}%"
    )

    plots_dir = Path("results/plots")
    cm_path = plots_dir / f"{model_name}__{dataset.name}_cm_supervised.png"
    plot_confusion_matrix(
        all_y_true, all_y_pred, dataset.class_names,
        f"{model_name} / {dataset.name} (supervised)", cm_path,
    )
    logger.info(f"Confusion matrix → {cm_path}")

    return {
        "model":     model_name,
        "dataset":   dataset.name,
        "n_samples": int(len(labels)),
        "n_classes": dataset.n_classes,
        "probes": {
            "supervised": {
                "fold_accs": [float(a) for a in fold_accs],
                "fold_f1s":  [float(f) for f in fold_f1s],
                "mean_acc":  mean_acc,
                "std_acc":   std_acc,
                "mean_f1":   mean_f1,
                "std_f1":    std_f1,
            },
        },
    }


# ── Few-shot supervised evaluation ──────────────────────────────────────────

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


def run_supervised_few_shot_evaluation(
    model_name: str,
    model_factory: Callable[[int], nn.Module],
    preprocess_fn: Callable[[torch.Tensor, torch.device], torch.Tensor],
    dataset: SpectrogramDataset,
    device: torch.device,
    batch_size: int = 16,
    lr: float = 1e-4,
    epochs: int = 30,
    weight_decay: float = 1e-4,
    n_shots_list: Sequence[int] = (1, 2, 5, 10, 20, 50, 100, 200),
    n_repeats: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Few-shot evaluation for a supervised model: fine-tune on N-shot
    subsets per fold and evaluate on the full held-out split.

    Returns a dict in the same shape as ``run_few_shot_evaluation()``.
    """
    items, labels = dataset.items()
    classes = np.unique(labels)
    n_classes = int(len(classes))
    folds = list(dataset.cv_splits())
    n_folds = len(folds)
    criterion = nn.CrossEntropyLoss()

    shots_to_run: List[Optional[int]] = list(n_shots_list) + [None]
    raw_scores: Dict[Optional[int], List[Tuple[float, float]]] = {
        n: [] for n in shots_to_run
    }

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        test_ds = _FoldDataset(dataset, test_idx, items, labels)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=4,
        )

        for n_shots in shots_to_run:
            repeats = 1 if n_shots is None else n_repeats
            for rep in range(repeats):
                if n_shots is None:
                    sub_idx = train_idx
                else:
                    rng = np.random.default_rng(
                        seed * 10_000 + fold_idx * 1_000 + rep,
                    )
                    sub_idx = train_idx[
                        _stratified_sample(labels[train_idx], classes, n_shots, rng)
                    ]

                train_ds = _FoldDataset(dataset, sub_idx, items, labels)
                train_loader = DataLoader(
                    train_ds, batch_size=min(batch_size, len(train_ds)),
                    shuffle=True, num_workers=4,
                )

                model = model_factory(n_classes).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=weight_decay,
                )
                ep = epochs
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=ep,
                )

                for epoch in range(ep):
                    _train_one_epoch(
                        model, train_loader, optimizer, criterion,
                        preprocess_fn, device,
                    )
                    scheduler.step()

                y_true, y_pred = _evaluate(
                    model, test_loader, preprocess_fn, device,
                )
                acc = float((y_pred == y_true).mean())
                f1  = float(f1_score(
                    y_true, y_pred, average="macro", zero_division=0,
                ))
                raw_scores[n_shots].append((acc, f1))

                del model, optimizer, scheduler
                torch.cuda.empty_cache()

        tag_str = ", ".join(
            "full" if s is None else f"{s}-shot" for s in shots_to_run
        )
        logger.debug(
            f"  [{model_name}] fold {fold_idx + 1}/{n_folds} done ({tag_str})"
        )

    # Aggregate
    probe_data: List[Dict[str, Any]] = []
    for n_shots in shots_to_run:
        scores = raw_scores[n_shots]
        accs = [s[0] for s in scores]
        f1s  = [s[1] for s in scores]
        n_total = (
            int(np.mean([len(t) for t, _ in folds]))
            if n_shots is None
            else n_shots * n_classes
        )
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
            f"[{model_name}] [{dataset.name}] supervised {tag}: "
            f"acc={entry['mean_acc']*100:.2f}% ± {entry['std_acc']*100:.2f}%  "
            f"F1={entry['mean_f1']*100:.2f}% ± {entry['std_f1']*100:.2f}%"
        )

    return {
        "model":     model_name,
        "dataset":   dataset.name,
        "n_classes": n_classes,
        "n_folds":   n_folds,
        "n_repeats": n_repeats,
        "probes": {
            "supervised": probe_data,
        },
    }
