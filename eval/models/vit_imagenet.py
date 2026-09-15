"""ImageNet-supervised ViT (AugReg recipe) as a frozen spectrogram extractor.

Supervised counterpart to ``DINOv3Extractor``: same architecture at both
sizes, but trained with class supervision on ImageNet-21k → ImageNet-1k
(AugReg) instead of self-distillation.  Lets us isolate the contribution
of SSL pretraining at matched architecture and matched parameter count.

CLS token is returned to mirror DINOv3's readout convention.
"""
from typing import Optional

import numpy as np
import timm
import torch
import torch.nn.functional as F

from .base import FeatureExtractor

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Both variants share the AugReg recipe (IN-21k pretrain → IN-1k FT) so
# the only thing that differs across sizes is parameter count.
_VARIANTS = {
    "vits16": ("vit_small_patch16_224.augreg_in21k_ft_in1k", 384),
    "vitb16": ("vit_base_patch16_224.augreg_in21k_ft_in1k", 768),
}


class ViTImageNetExtractor(FeatureExtractor):
    """Supervised ImageNet ViT-S/16 or ViT-B/16, AugReg recipe via timm.

    Preprocessing mirrors ``DINOv3Extractor``: log1p → resize 224×224 →
    replicate to 3 channels → per-sample min-max → ImageNet normalise.
    """

    def __init__(
        self,
        variant: str = "vitb16",
        device: str = "cpu",
    ) -> None:
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {list(_VARIANTS)}, got {variant!r}")
        timm_name, dim = _VARIANTS[variant]
        size_tag = "small" if variant == "vits16" else "base"
        self._name = f"vit_{size_tag}_imagenet"
        self._embed_dim = dim
        self._device = torch.device(device)
        # num_classes=0 makes timm replace the classification head with
        # nn.Identity; forward(x) then returns the CLS token (B, D).
        self.model = timm.create_model(timm_name, pretrained=True, num_classes=0)
        self.model.eval().to(self._device)

    @property
    def name(self) -> str:
        return self._name

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def to(self, device) -> "ViTImageNetExtractor":
        self._device = torch.device(device)
        self.model = self.model.to(self._device)
        return self

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self._device)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = torch.log1p(x.clamp(min=0))
        x = F.interpolate(
            x.float(), size=(224, 224),
            mode="bilinear", align_corners=False,
        )
        x = x.repeat(1, 3, 1, 1)

        b = x.shape[0]
        x_flat = x.view(b, -1)
        lo = x_flat.min(dim=1).values.view(b, 1, 1, 1)
        hi = x_flat.max(dim=1).values.view(b, 1, 1, 1)
        x = (x - lo) / (hi - lo + 1e-8)

        mean = _IMAGENET_MEAN.to(self._device)
        std  = _IMAGENET_STD.to(self._device)
        x = (x - mean) / std

        return self.model(x).cpu().numpy()
