from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from .base import SpectrogramDataset

CLASS_NAMES = [
    "Walk",              # activity 01
    "Stand",             # activity 02
    "Sit",               # activity 03
    "Up Stairs",         # activity 04
    "Down Stairs",       # activity 05
    "Pick",              # activity 06
    "Step Over",         # activity 07
    "Semi-Turn Around",  # activity 08
    "Fall",              # activity 09
    "Walk (Arbitrary)",  # activity 10
]

# Activities that overlap with the Glasgow dataset (1-indexed).
# Walk, Stand, Sit, Pick, Fall.
GLASGOW_OVERLAP_ACTIVITIES = [1, 2, 3, 6, 9]


def _parse_filename(stem: str):
    """Parse MAD filename: AY1Y2PXXXRZSUDV_complex.

    Returns (activity, subcategory, participant, repetition, antenna).
    """
    name = stem.replace("_complex", "")
    activity = int(name[1:3])       # Y1: 01-10
    subcategory = int(name[3])      # Y2: 1-5
    participant = int(name[5:8])    # XXX: 001-100
    repetition = int(name[9])       # Z: 1-3
    antenna = int(name[13])         # V: 3 or 4
    return activity, subcategory, participant, repetition, antenna


class MADDataset(SpectrogramDataset):
    """MAD micro-Doppler human activity recognition dataset.

    Complex STFT spectrograms from FMCW radar (9.8 GHz), shape (256, 750),
    stored as .npy complex64 files in participant-indexed subdirectories.

    Two antennas available: D3 (Rx1 spectrogram) and D4 (Rx2 spectrogram).
    By default both are used, doubling the sample count.  Participant-grouped
    CV ensures no leakage across antennas of the same recording.

    Preprocessing: |spec| → (1, 256, 750) tensor (raw linear magnitude).
    CV: StratifiedGroupKFold(5) grouped by participant.
    """

    def __init__(
        self,
        root: str | Path,
        antenna: str = "both",
        activities: Optional[Sequence[int]] = None,
        subcategories: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        root : path to data/MAD/ containing participant subdirectories.
        antenna : "D3" (Rx1), "D4" (Rx2), or "both".
        activities : 1-indexed activity numbers to keep (e.g. [1,2,3,6,9]).
            None means all 10 activities.  Labels are remapped to 0..N-1.
        subcategories : subcategory numbers to keep (e.g. [1] or [2,3]).
            None means all subcategories.
        """
        self.root = Path(root)
        if antenna not in ("D3", "D4", "both"):
            raise ValueError(f"antenna must be 'D3', 'D4', or 'both', got '{antenna}'")
        self.antenna = antenna
        self._activities = sorted(activities) if activities is not None else None
        self._subcategories = sorted(subcategories) if subcategories is not None else None
        self._files: List[Path] = []
        self._labels: np.ndarray = np.array([], dtype=np.int64)
        self._groups: np.ndarray = np.array([], dtype=np.int64)
        self._build_index()

    def _build_index(self) -> None:
        patterns = []
        if self.antenna in ("D3", "both"):
            patterns.append("*D3_complex.npy")
        if self.antenna in ("D4", "both"):
            patterns.append("*D4_complex.npy")

        allowed_activities = set(self._activities) if self._activities is not None else None
        allowed_subcats = set(self._subcategories) if self._subcategories is not None else None

        # Map 1-indexed activities to contiguous 0-indexed labels.
        if self._activities is not None:
            activity_to_label = {a: i for i, a in enumerate(self._activities)}
        else:
            activity_to_label = {a: a - 1 for a in range(1, 11)}

        files, labels, groups = [], [], []
        for pattern in patterns:
            for f in sorted(self.root.glob(f"**/{pattern}")):
                activity, subcategory, participant, _, _ = _parse_filename(f.stem)
                if allowed_activities is not None and activity not in allowed_activities:
                    continue
                if allowed_subcats is not None and subcategory not in allowed_subcats:
                    continue
                files.append(f)
                labels.append(activity_to_label[activity])
                groups.append(participant)

        self._files = files
        self._labels = np.array(labels, dtype=np.int64)
        self._groups = np.array(groups, dtype=np.int64)

    @property
    def name(self) -> str:
        parts = ["mad"]
        if self._activities is not None:
            parts.append(str(len(self._activities)))
        if self.antenna != "both":
            parts.append(self.antenna.lower())
        if self._subcategories is not None:
            parts.append("sub" + "".join(str(s) for s in self._subcategories))
        return "_".join(parts)

    @property
    def n_classes(self) -> int:
        if self._activities is not None:
            return len(self._activities)
        return 10

    @property
    def class_names(self) -> List[str]:
        if self._activities is not None:
            return [CLASS_NAMES[a - 1] for a in self._activities]
        return CLASS_NAMES

    @property
    def groups(self) -> np.ndarray:
        return self._groups

    def items(self) -> Tuple[List[Path], np.ndarray]:
        return self._files, self._labels

    def load_item(self, path: Path) -> torch.Tensor:
        spec = np.load(path)  # (256, 750) complex64
        spec = np.abs(spec).astype(np.float32)
        return torch.from_numpy(spec).unsqueeze(0)  # (1, 256, 750)

    def cv_splits(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        yield from sgkf.split(
            np.zeros(len(self._labels)), self._labels, self._groups
        )
