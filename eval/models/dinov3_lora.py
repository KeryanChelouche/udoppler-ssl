"""DINOv3 ViT-B/16 with LoRA adapters for supervised fine-tuning.

Used in the cross-dataset few-shot pipeline: LoRA-adapt on a source
dataset, then freeze the backbone and fit a linear probe on target
few-shot data.

Compatible with ``cross_few_shot.py``'s interface:
  - ``build_dinov3_lora(n_classes) -> nn.Module`` (has ``.fc`` head)
  - After fine-tuning, set ``model.fc = nn.Identity()`` to extract
    CLS token features of shape ``(B, 768)``.
"""

import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

_VARIANTS = {
    "vits16": ("facebook/dinov3-vits16-pretrain-lvd1689m", 384),
    "vitb16": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
}

# DINOv3 (HF) linear module suffixes:
#   attention: q_proj, k_proj, v_proj, o_proj
#   mlp:       up_proj, down_proj
_ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
_QV_MODULES = ["q_proj", "v_proj"]   # original LoRA paper's recommendation
_MLP_MODULES = ["up_proj", "down_proj"]


class DINOv3LoRA(nn.Module):
    """DINOv3 backbone with LoRA adapters and a classification head."""

    def __init__(self, backbone: nn.Module, embed_dim: int, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        cls_token = out.last_hidden_state[:, 0]
        return self.fc(cls_token)


def build_dinov3_lora(
    n_classes: int,
    variant: str = "vitb16",
    rank: int = 16,
    alpha: Optional[int] = 32,
    token: Optional[str] = None,
    use_dora: bool = False,
    target_modules: Union[str, List[str]] = "attn",
    init_lora_weights: Union[bool, str] = True,
) -> DINOv3LoRA:
    """Build a DINOv3 model with LoRA (or DoRA / PiSSA) adapters and a head.

    Only adapter weights and the classification head are trainable; the
    pretrained backbone is frozen.

    Args:
        n_classes: Number of output classes for the head.
        variant:   Model variant ("vits16" or "vitb16").
        rank:      LoRA rank.
        alpha:     LoRA alpha (scaling = alpha / rank). If ``None``,
                   defaults to ``2 * rank`` (the alpha=2r convention from
                   the LoRA / DoRA papers). Existing callers passing
                   ``alpha=32`` get the original behaviour.
        token:     HuggingFace token (falls back to HF_TOKEN env var).
        use_dora:  If True, use DoRA (weight-decomposed LoRA).
        target_modules: Which linear modules to adapt.
                   "attn"   → q_proj, k_proj, v_proj, o_proj (default)
                   "all"    → attention + MLP (up_proj, down_proj)
                   list[str] → explicit suffix list passed to PEFT.
        init_lora_weights: Passed to LoraConfig. ``True`` for default
                   LoRA (zero-init B), ``"pissa"`` to initialise from SVD
                   of W0 (PiSSA). PiSSA is incompatible with use_dora=True.
    """
    if variant not in _VARIANTS:
        raise ValueError(
            f"variant must be one of {list(_VARIANTS)}, got {variant!r}"
        )
    if use_dora and isinstance(init_lora_weights, str) and init_lora_weights == "pissa":
        raise ValueError("DoRA + PiSSA is not a supported combination.")

    if alpha is None:
        alpha = 2 * rank

    if isinstance(target_modules, str):
        if target_modules == "attn":
            tm = list(_ATTN_MODULES)
        elif target_modules == "qv":
            tm = list(_QV_MODULES)
        elif target_modules == "all":
            tm = list(_ATTN_MODULES) + list(_MLP_MODULES)
        else:
            raise ValueError(
                f"target_modules string must be one of 'attn', 'qv', 'all'; got {target_modules!r}"
            )
    else:
        tm = list(target_modules)

    hub_id, embed_dim = _VARIANTS[variant]
    hf_token = token or os.environ.get("HF_TOKEN")
    backbone = AutoModel.from_pretrained(hub_id, token=hf_token)

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
        f"DINOv3-{variant} {method} (r={rank}, alpha={alpha}, targets={tm}): "
        f"{trainable:,} trainable / {total:,} total params "
        f"({100 * trainable / total:.2f}%)"
    )

    return DINOv3LoRA(backbone, embed_dim, n_classes)
