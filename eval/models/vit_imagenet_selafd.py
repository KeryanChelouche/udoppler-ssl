"""SelaFD reimplementation: ViT-S/B + LoRA(q,v) + serial/parallel adapters.

Reproduces the architecture from:
  Wang et al., "SelaFD: Seamless Adaptation of Vision Transformer Fine-tuning
  for Radar-based Human Activity Recognition", arXiv:2502.04740 (2025).
  Reference code: https://github.com/wangyijunlyy/SelaFD

Differences from the reference repo (intentional):
  * Default parallel-adapter scale ``s = 0.2`` (paper value).  The repo
    hard-codes ``s = 1.0`` despite a stale ``# 0.2`` comment.
  * Standard ``preprocess_batch`` (log1p + min-max + ImageNet-norm) is used
    upstream — no RandomResizedCrop / HFlip, to keep preprocessing identical
    across all PEFT methods in this framework.
  * ViT-S backbone is exposed in addition to ViT-B for head-to-head
    comparison against DINOv3-vits16 PEFT runs.
  * Optimizer / loss / batch size are owned by the cross-few-shot driver.
    The repo's label_smoothing=0.1 is dropped here — the rest of this
    framework trains with plain CE, and keeping the loss identical across
    methods is what makes the comparison vs DINOv3+PiSSA fair.

Compatible with ``cross_few_shot.py``'s interface:
  - ``build_vit_imagenet_selafd(n_classes) -> nn.Module`` (has ``.fc`` head)
  - After fine-tuning, ``model.fc = nn.Identity()`` returns CLS-token
    features of shape ``(B, embed_dim)`` for probing.
"""

import math
import types

import timm
import torch
import torch.nn as nn
from loguru import logger

_VARIANTS = {
    "vits16": ("vit_small_patch16_224.augreg_in21k_ft_in1k", 384),
    "vitb16": ("vit_base_patch16_224.augreg_in21k_ft_in1k", 768),
}


class _LoRAQKV(nn.Module):
    """Wrap a fused ``qkv`` Linear and add LoRA deltas to the q and v slices only.

    The base ``qkv`` weight stays frozen.  Two rank-r LoRA pairs are
    learned, one for q and one for v.  k is left untouched, matching the
    SelaFD paper (LoRA on q,v only).
    """

    def __init__(self, qkv: nn.Linear, r: int, alpha: int) -> None:
        super().__init__()
        self.qkv = qkv
        self.r = r
        self.alpha = alpha
        # Match the SelaFD repo's integer scaling (alpha // r).  With the
        # paper's r=alpha=4 this evaluates to 1, identical to a float
        # alpha/r.  Kept as integer to avoid silent drift from the
        # reference implementation.
        self.scale = alpha // r if alpha % r == 0 else alpha / r

        in_dim = qkv.in_features
        out_dim = qkv.out_features
        if out_dim != 3 * in_dim:
            raise ValueError(
                f"_LoRAQKV expects a fused qkv linear (out=3*in); "
                f"got in={in_dim}, out={out_dim}"
            )
        self.dim = in_dim

        self.linear_a_q = nn.Linear(in_dim, r, bias=False)
        self.linear_b_q = nn.Linear(r, in_dim, bias=False)
        self.linear_a_v = nn.Linear(in_dim, r, bias=False)
        self.linear_b_v = nn.Linear(r, in_dim, bias=False)

        nn.init.kaiming_uniform_(self.linear_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.zeros_(self.linear_b_v.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)                                       # (B, N, 3D)
        delta_q = self.scale * self.linear_b_q(self.linear_a_q(x))
        delta_v = self.scale * self.linear_b_v(self.linear_a_v(x))
        q_part, k_part, v_part = qkv.chunk(3, dim=-1)
        return torch.cat([q_part + delta_q, k_part, v_part + delta_v], dim=-1)


class _Adapter(nn.Module):
    """Bottleneck MLP adapter (ratio = hidden/dim, ReLU).

    ``D_fc2`` is zero-initialised so the adapter starts as an identity
    (when ``skip_connect=True``, serial branch) or as a no-op (when
    ``skip_connect=False``, parallel branch).  This keeps the pretrained
    forward pass intact at step 0.
    """

    def __init__(
        self,
        dim: int,
        ratio: float = 0.5,
        skip_connect: bool = True,
    ) -> None:
        super().__init__()
        hidden = max(1, int(dim * ratio))
        self.skip_connect = skip_connect
        self.D_fc1 = nn.Linear(dim, hidden)
        self.act = nn.ReLU()
        self.D_fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.D_fc2.weight)
        nn.init.zeros_(self.D_fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.D_fc2(self.act(self.D_fc1(x)))
        return x + h if self.skip_connect else h


def _selafd_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    # Serial adapter sits in-line on the attention output; parallel
    # adapter is a side-branch off LN2 summed with the MLP output.
    x = x + self.drop_path1(self.ls1(self.adapter1(self.attn(self.norm1(x)))))
    h = self.norm2(x)
    x = x + self.drop_path2(
        self.ls2(self.mlp(h) + self.parallel_scale * self.adapter2(h))
    )
    return x


class ViTImageNetSelaFD(nn.Module):
    """ImageNet ViT backbone with SelaFD adapters and a classification head."""

    def __init__(self, backbone: nn.Module, embed_dim: int, n_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)             # (B, D), CLS via num_classes=0
        return self.fc(features)


def build_vit_imagenet_selafd(
    n_classes: int,
    variant: str = "vitb16",
    lora_rank: int = 4,
    lora_alpha: int = 4,
    adapter_ratio: float = 0.5,
    parallel_scale: float = 0.2,
) -> ViTImageNetSelaFD:
    """Build a SelaFD-style ViT: LoRA(q,v) + serial + parallel adapters.

    Args:
        n_classes:      Number of output classes for the head.
        variant:        ``"vits16"`` (384-dim) or ``"vitb16"`` (768-dim).
        lora_rank:      LoRA rank for the q,v adapters (paper: 4).
        lora_alpha:     LoRA alpha; scaling is ``alpha // r`` (paper: 4).
        adapter_ratio:  Adapter bottleneck ratio = hidden / embed_dim
                        (paper: 0.5).
        parallel_scale: Scalar multiplier on the parallel adapter branch
                        (paper: 0.2; repo: 1.0).
    """
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {list(_VARIANTS)}, got {variant!r}")
    timm_name, embed_dim = _VARIANTS[variant]
    backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)

    # Freeze the entire pretrained backbone.  Newly inserted LoRA /
    # adapter modules will get requires_grad=True by default.
    for p in backbone.parameters():
        p.requires_grad = False

    for block in backbone.blocks:
        block.attn.qkv = _LoRAQKV(block.attn.qkv, r=lora_rank, alpha=lora_alpha)
        block.adapter1 = _Adapter(embed_dim, ratio=adapter_ratio, skip_connect=True)
        block.adapter2 = _Adapter(embed_dim, ratio=adapter_ratio, skip_connect=False)
        block.parallel_scale = parallel_scale
        block.forward = types.MethodType(_selafd_block_forward, block)

    trainable, total = 0, 0
    for p in backbone.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    logger.info(
        f"ViT-{variant} SelaFD (r={lora_rank}, alpha={lora_alpha}, "
        f"ratio={adapter_ratio}, s={parallel_scale}): "
        f"{trainable:,} trainable / {total:,} total params "
        f"({100 * trainable / total:.2f}%)"
    )

    return ViTImageNetSelaFD(backbone, embed_dim, n_classes)
