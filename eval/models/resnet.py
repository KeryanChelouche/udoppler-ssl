"""Supervised ResNet baseline — fine-tuned end-to-end on target datasets."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_VARIANTS = {
    "resnet18":  (models.resnet18,  models.ResNet18_Weights.DEFAULT,  512),
    "resnet50":  (models.resnet50,  models.ResNet50_Weights.DEFAULT,  2048),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048),
}


def build_resnet(variant: str, n_classes: int) -> nn.Module:
    """Return a pretrained ResNet with its fc head replaced for *n_classes*."""
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {list(_VARIANTS)}, got {variant!r}")
    factory, weights, dim = _VARIANTS[variant]
    net = factory(weights=weights)
    net.fc = nn.Linear(dim, n_classes)
    return net


def preprocess_batch(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Log-compress, resize to 224x224, and apply ImageNet normalisation.

    Args:
        x: Raw linear-scale spectrogram, (B, 1, F, T) or (B, F, T).
        device: target device.

    Returns:
        (B, 3, 224, 224) tensor ready for a torchvision ResNet.
    """
    x = x.to(device)
    if x.ndim == 3:
        x = x.unsqueeze(1)

    # Log compression (reduces dynamic range before min-max)
    x = torch.log1p(x.clamp(min=0))

    x = F.interpolate(x.float(), size=(224, 224), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)

    # Per-sample min-max normalisation to [0, 1]
    b = x.shape[0]
    x_flat = x.view(b, -1)
    lo = x_flat.min(dim=1).values.view(b, 1, 1, 1)
    hi = x_flat.max(dim=1).values.view(b, 1, 1, 1)
    x = (x - lo) / (hi - lo + 1e-8)

    # ImageNet channel normalisation
    mean = _IMAGENET_MEAN.to(device)
    std  = _IMAGENET_STD.to(device)
    x = (x - mean) / std

    return x
