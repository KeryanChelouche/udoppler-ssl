import re
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from .base import SpectrogramDataset

CLASS_NAMES = [
    "Walking",
    "Sitting down",
    "Standing up",
    "Picking up object",
    "Drinking",
    "Falling",
]

# 0-indexed classes to drop for the har5 variant (Drinking=4).
HAR5_EXCLUDE = [4]

# Filename prefixes (D01..D07) group recordings by acquisition campaign
# (different locations / dates / participant cohorts).  Files start with
# "D0X_" where X is 1..7.
_DATASET_PREFIX_RE = re.compile(r"^D0(\d)_")

# Young-aged participants (<40) recorded in the mature campaigns D6-D7.
# Listed as (campaign_id, participant_id_within_campaign) tuples.
#
#   (6,  8): age 33 — same physical person as D05 pid 8 (genuine duplicate).
#   (6, 22): age 33 — pid only present in D6-7.
#   (6, 25): age 25 — pid only present in D6-7.
#   (6, 31): age 24 — same physical person as D03 pid 31 (genuine duplicate).
#   (7, 56): age 25 — pid 56 also exists in D01 but as a different 32-yo person
#                     (within-campaign pid collision, not a duplicate).
#
# Dropping these from glasgow_mature gives a clean strictly-older test set
# with no leakage from glasgow_young and no young contaminants.
GLASGOW_MATURE_YOUNG_EXCLUDE = [(6, 8), (6, 22), (6, 25), (6, 31), (7, 56)]


class GlasgowDataset(SpectrogramDataset):
    """Glasgow micro-Doppler human activity recognition dataset.

    2081 spectrograms, 6 activity classes, stored as .npy files of
    shape (1024, 365), float32, Fortran-order, in class-indexed
    subdirectories (1–6).

    Preprocessing: ascontiguousarray → (1, 1024, 365) tensor (raw linear scale).
    CV: StratifiedGroupKFold(5) grouped by participant (no subject leakage).

    Parameters
    ----------
    exclude_classes : 0-indexed class IDs to drop (e.g. [4] for Drinking).
        Remaining labels are remapped to 0..N-1.
    datasets : acquisition-campaign IDs to keep (1..7), parsed from the
        ``D0X_`` prefix in each filename.  None means all campaigns.
        Different IDs correspond to different recording locations / dates,
        so this lets you build distribution-shift splits (e.g. young
        cohort D1-5 vs. mature cohort D6-7).
    subset_name : human-readable label appended to ``name`` (e.g. "young"
        → ``glasgow_young``).  Used as cache key and for label-scheme
        lookup, so it must match the registry entry.
    exclude_dpids : iterable of ``(campaign_id, participant_id)`` tuples
        whose recordings should be dropped.  Used to remove specific
        subjects (e.g. young-aged participants leaking into the mature
        cohort) without touching ``metadata.csv``.
    """

    def __init__(
        self,
        root: str | Path,
        exclude_classes: Optional[Sequence[int]] = None,
        datasets: Optional[Sequence[int]] = None,
        subset_name: Optional[str] = None,
        exclude_dpids: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        self.root = Path(root)
        self._exclude = set(exclude_classes) if exclude_classes is not None else set()
        self._datasets = sorted(datasets) if datasets is not None else None
        if self._datasets is not None and any(
            d < 1 or d > 7 for d in self._datasets
        ):
            raise ValueError(f"datasets must be in 1..7, got {self._datasets}")
        self._subset_name = subset_name
        self._exclude_dpids = (
            set(map(tuple, exclude_dpids)) if exclude_dpids is not None else set()
        )
        self._files: List[Path] = []
        self._labels: np.ndarray = np.array([], dtype=np.int64)
        self._groups: np.ndarray = np.array([], dtype=np.int64)
        self._kept_classes: List[int] = []  # original 0-indexed IDs kept
        self._build_index()

    def _build_index(self) -> None:
        # Determine which original classes are kept and build remap.
        self._kept_classes = [c for c in range(6) if c not in self._exclude]
        remap = {orig: new for new, orig in enumerate(self._kept_classes)}
        allowed_datasets = set(self._datasets) if self._datasets is not None else None

        files, labels, groups = [], [], []
        for class_id in range(1, 7):
            orig_label = class_id - 1  # 0-indexed
            if orig_label in self._exclude:
                continue
            class_dir = self.root / str(class_id)
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            for f in sorted(class_dir.glob("*.npy")):
                m = re.search(r"P(\d+)", f.stem)
                if m is None:
                    raise ValueError(f"Cannot parse participant ID from {f.name}")
                pid = int(m.group(1))
                d_id: Optional[int] = None
                if allowed_datasets is not None or self._exclude_dpids:
                    dm = _DATASET_PREFIX_RE.match(f.stem)
                    if dm is None:
                        raise ValueError(
                            f"Cannot parse dataset prefix from {f.name}"
                        )
                    d_id = int(dm.group(1))
                if allowed_datasets is not None and d_id not in allowed_datasets:
                    continue
                if self._exclude_dpids and (d_id, pid) in self._exclude_dpids:
                    continue
                files.append(f)
                labels.append(remap[orig_label])
                groups.append(pid)
        self._files = files
        self._labels = np.array(labels, dtype=np.int64)
        self._groups = np.array(groups, dtype=np.int64)

    @property
    def name(self) -> str:
        base = "glasgow_5" if self._exclude else "glasgow"
        if self._subset_name is not None:
            return f"{base}_{self._subset_name}"
        if self._datasets is not None:
            return base + "_d" + "".join(str(d) for d in self._datasets)
        return base

    @property
    def n_classes(self) -> int:
        return len(self._kept_classes)

    @property
    def class_names(self) -> List[str]:
        return [CLASS_NAMES[c] for c in self._kept_classes]

    @property
    def groups(self) -> np.ndarray:
        return self._groups

    def items(self) -> Tuple[List[Path], np.ndarray]:
        return self._files, self._labels

    def load_item(self, path: Path) -> torch.Tensor:
        spec = np.load(path)
        spec = np.ascontiguousarray(spec).astype(np.float32)
        return torch.from_numpy(spec).unsqueeze(0)  # (1, 1024, 365)

    def cv_splits(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        yield from sgkf.split(
            np.zeros(len(self._labels)), self._labels, self._groups
        )
