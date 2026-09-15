"""ImageNet-supervised ViT (AugReg) with LoRA adapters for supervised fine-tuning.

Supervised counterpart to ``dinov3_lora.py`` — same architecture and
sizes, but the backbone is trained with class supervision on ImageNet
instead of DINO self-distillation.

Compatible with ``cross_few_shot.py``'s interface:
  - ``build_vit_imagenet_lora(n_classes) -> nn.Module`` (has ``.fc`` head)
  - After fine-tuning, ``model.fc = nn.Identity()`` makes ``forward(x)``
    return CLS-token features of shape ``(B, embed_dim)`` for probing.
"""

from typing import List, Union

import timm
import torch
import torch.nn as nn
from loguru import logger
from peft import LoraConfig, get_peft_model

_VARIANTS = {
    "vits16": ("vit_small_patch16_224.augreg_in21k_ft_in1k", 384),
    "vitb16": ("vit_base_patch16_224.augreg_in21k_ft_in1k", 768),
}

# timm ViT linear module suffixes (per block):
#   attention: attn.qkv (fused Q,K,V), attn.proj
#   mlp:       mlp.fc1, mlp.fc2
# Suffix matching is used to avoid PatchEmbed.proj (Conv2d) which shares
# the bare name "proj".
_ATTN_MODULES = ["attn.qkv", "attn.proj"]
_MLP_MODULES  = ["mlp.fc1", "mlp.fc2"]


class ViTImageNetLoRA(nn.Module):
    """ImageNet ViT backbone with LoRA adapters and a classification head.

    Accepts already-preprocessed images ``(B, 3, 224, 224)`` — use
    ``eval.models.resnet.preprocess_batch`` upstream.
    """

    def __init__(self, backbone: nn.Module, embed_dim: int, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)     # (B, D) — CLS token via num_classes=0
        return self.fc(features)


def build_vit_imagenet_lora(
    n_classes: int,
    variant: str = "vitb16",
    rank: int = 16,
    alpha: int = 32,
    use_dora: bool = False,
    target_modules: Union[str, List[str]] = "attn",
    init_lora_weights: Union[bool, str] = True,
) -> ViTImageNetLoRA:
    """Build a supervised ImageNet ViT with LoRA (or DoRA / PiSSA) + classifier.

    Args:
        n_classes: Number of output classes for the head.
        variant:   One of "vits16", "vitb16".
        rank:      LoRA rank.
        alpha:     LoRA alpha (scaling = alpha / rank).
        use_dora:  If True, use DoRA (weight-decomposed LoRA).
        target_modules: Which linear modules to adapt.
                   "attn" → attn.qkv, attn.proj (default)
                   "all"  → attention + MLP (mlp.fc1, mlp.fc2)
                   list[str] → explicit suffix list passed to PEFT.
        init_lora_weights: ``True`` for standard LoRA (zero-init B),
                   ``"pissa"`` to initialise A,B from SVD of W0 (PiSSA).
                   PiSSA is incompatible with use_dora=True.
    """
    if variant not in _VARIANTS:
        raise ValueError(
            f"variant must be one of {list(_VARIANTS)}, got {variant!r}"
        )
    if use_dora and isinstance(init_lora_weights, str) and init_lora_weights == "pissa":
        raise ValueError("DoRA + PiSSA is not a supported combination.")

    if isinstance(target_modules, str):
        if target_modules == "attn":
            tm = list(_ATTN_MODULES)
        elif target_modules == "all":
            tm = list(_ATTN_MODULES) + list(_MLP_MODULES)
        else:
            raise ValueError(
                f"target_modules string must be 'attn' or 'all'; got {target_modules!r}"
            )
    else:
        tm = list(target_modules)

    timm_name, embed_dim = _VARIANTS[variant]
    backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=tm,
        lora_dropout=0.05,
        bias="none",
        use_dora=use_dora,
        init_lora_weights=init_lora_weights,
    )
    backbone = get_peft_model(backbone, lora_config)

    trainable, total = 0, 0
    for p in backbone.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    if use_dora:
        method = "DoRA"
    elif isinstance(init_lora_weights, str) and init_lora_weights == "pissa":
        method = "PiSSA"
    else:
        method = "LoRA"
    logger.info(
        f"ViT-{variant} ImageNet {method} (r={rank}, alpha={alpha}, targets={tm}): "
        f"{trainable:,} trainable / {total:,} total params "
        f"({100 * trainable / total:.2f}%)"
    )

    return ViTImageNetLoRA(backbone, embed_dim, n_classes)
