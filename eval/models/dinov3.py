import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from .base import FeatureExtractor

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_VARIANTS = {
    "vits16": ("facebook/dinov3-vits16-pretrain-lvd1689m", 384),
    "vitb16": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
}


class DINOv3Extractor(FeatureExtractor):
    """DINOv3 ViT-B/16 pretrained on LVD-1689M.

    Datasets provide raw linear-scale spectrograms.  We apply log1p
    compression, resize to 224x224, replicate to 3 channels, per-sample
    min-max normalise to [0, 1], then ImageNet-normalise.

    The [CLS] token (768-d) is returned rather than mean-pooled patches:
    DINO's self-distillation objective directly trains the CLS token to be
    a compact global representation, making it the standard readout for
    this family of models.

    Args:
        variant: Model variant key (currently only "vitb16").
        device:  Torch device string, e.g. "cuda" or "cpu".
    """

    def __init__(
        self,
        variant: str = "vitb16",
        device: str = "cpu",
        token: Optional[str] = None,
    ) -> None:
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {list(_VARIANTS)}, got {variant!r}")
        hub_id, dim = _VARIANTS[variant]
        self._name = f"dinov3_{variant}"
        self._embed_dim = dim
        self._device = torch.device(device)
        hf_token = token or os.environ.get("HF_TOKEN")
        self.model = AutoModel.from_pretrained(hub_id, token=hf_token)
        self.model.eval().to(self._device)

    @property
    def name(self) -> str:
        return self._name

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def to(self, device) -> "DINOv3Extractor":
        self._device = torch.device(device)
        self.model = self.model.to(self._device)
        return self

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """
        Args:
            x: Raw linear-scale spectrogram, (B, F, T) or (B, 1, F, T).

        Returns:
            CLS token features, shape (B, 768), float32 numpy array.
        """
        x = x.to(self._device)
        if x.ndim == 3:
            x = x.unsqueeze(1)                              # (B, 1, F, T)

        # Log compression (reduces dynamic range before min-max)
        x = torch.log1p(x.clamp(min=0))

        x = F.interpolate(                                  # (B, 1, 224, 224)
            x.float(), size=(224, 224),
            mode="bilinear", align_corners=False,
        )
        x = x.repeat(1, 3, 1, 1)                           # (B, 3, 224, 224)

        # Per-sample min-max normalisation to [0, 1]
        b = x.shape[0]
        x_flat = x.view(b, -1)
        lo = x_flat.min(dim=1).values.view(b, 1, 1, 1)
        hi = x_flat.max(dim=1).values.view(b, 1, 1, 1)
        x = (x - lo) / (hi - lo + 1e-8)

        # ImageNet channel normalisation
        mean = _IMAGENET_MEAN.to(self._device)
        std  = _IMAGENET_STD.to(self._device)
        x = (x - mean) / std

        out = self.model(pixel_values=x)
        # last_hidden_state: (B, N_patches + 1, D) — index 0 is the CLS token
        return out.last_hidden_state[:, 0].cpu().numpy()   # (B, 768)
