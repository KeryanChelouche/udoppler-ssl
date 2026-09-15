"""DINOv3 ViT with full fine-tuning (no PEFT) and a classification head.

Reuses the DINOv3 backbone + CLS-token + linear head pattern from
``dinov3_lora.py`` but skips the PEFT wrapping, leaving all backbone
parameters trainable.  Used as the "w/o PEFT" ablation row.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModel

_VARIANTS = {
    "vits16": ("facebook/dinov3-vits16-pretrain-lvd1689m", 384),
    "vitb16": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
}


class DINOv3FullFT(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        cls_token = out.last_hidden_state[:, 0]
        return self.fc(cls_token)


def build_dinov3_full_ft(
    n_classes: int,
    variant: str = "vits16",
    token: Optional[str] = None,
) -> DINOv3FullFT:
    if variant not in _VARIANTS:
        raise ValueError(
            f"variant must be one of {list(_VARIANTS)}, got {variant!r}"
        )
    hub_id, embed_dim = _VARIANTS[variant]
    hf_token = token or os.environ.get("HF_TOKEN")
    backbone = AutoModel.from_pretrained(hub_id, token=hf_token)
    for p in backbone.parameters():
        p.requires_grad = True

    model = DINOv3FullFT(backbone, embed_dim, n_classes)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"DINOv3-{variant} Full-FT: {trainable:,} trainable / {total:,} "
        f"total params ({100 * trainable / total:.2f}%)"
    )
    return model
