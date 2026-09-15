"""Frozen-backbone nn.Module wrapper for cross_few_shot.

Wraps a FeatureExtractor (which has its own internal preprocessing and
returns numpy from `.extract()`) in a thin nn.Module so it can flow
through `run_cross_few_shot_evaluation(freeze_backbone=True)` alongside
the LoRA/DoRA/ResNet entries.

Use the identity preprocess (``_identity_preprocess`` below) when
registering, because the extractor handles its own log1p / resize /
normalise.
"""
from typing import Any

import torch
import torch.nn as nn

from .base import FeatureExtractor


def identity_preprocess(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device)


class FrozenBackboneModule(nn.Module):
    def __init__(self, extractor: FeatureExtractor) -> None:
        super().__init__()
        self.extractor = extractor
        self.fc = nn.Identity()

    def to(self, device: Any, *args: Any, **kwargs: Any) -> "FrozenBackboneModule":
        self.extractor.to(device)
        return super().to(device, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arr = self.extractor.extract(x)
        return torch.from_numpy(arr).to(x.device)

    def train(self, mode: bool = True) -> "FrozenBackboneModule":
        super().train(False)
        return self

    def parameters(self, recurse: bool = True):
        return iter(())
