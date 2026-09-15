#!/usr/bin/env python3
"""PEFT method sweep for DINOv3 on cross-dataset few-shot transfer.

Sweeps LoRA / PiSSA across rank, target-modules, learning-rate, and seed
on a single (source, target) pair, then aggregates the linear probe
scores at the few-shot levels of interest (5/10/20).

Stage 1 (LoRA only):
    python scripts/run_peft_sweep.py \\
        --methods lora \\
        --lrs 1e-4 3e-4 \\
        --ranks 8 16 \\
        --targets attn all \\
        --seeds 0 1 2

Stage 2 (PiSSA at LRs derived from stage 1):
    python scripts/run_peft_sweep.py \\
        --methods pissa \\
        --lrs <best-lora-lr> <best-lora-lr-/2> \\
        --ranks 8 16 \\
        --targets attn all \\
        --seeds 0 1 2

Each (method, lr, rank, targets, seed) combination produces one JSON
result file.  Files already on disk are skipped, so the script can be
resumed.
"""
import argparse
import itertools
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from eval.cross_few_shot import run_cross_few_shot_evaluation  # noqa: E402
from eval.datasets.glasgow import (  # noqa: E402
    GLASGOW_MATURE_YOUNG_EXCLUDE,
    GlasgowDataset,
    HAR5_EXCLUDE,
)
from eval.datasets.mad import GLASGOW_OVERLAP_ACTIVITIES, MADDataset  # noqa: E402
from eval.models.dinov3_full_ft import build_dinov3_full_ft  # noqa: E402
from eval.models.dinov3_lora import build_dinov3_lora  # noqa: E402
from eval.models.resnet import preprocess_batch  # noqa: E402
from eval.models.vit_imagenet_lora import build_vit_imagenet_lora  # noqa: E402
from eval.models.vit_imagenet_selafd import build_vit_imagenet_selafd  # noqa: E402
from eval.probes.knn import KNNProbe  # noqa: E402
from eval.probes.linear import LinearProbe  # noqa: E402
from loguru import logger  # noqa: E402


# Methods:
#   lora/pissa → DINOv3 backbone, --variant picks size, --targets/--ranks honoured.
#   selafd     → ImageNet ViT, paper-pinned (q,v + adapters), only (lr, seed, variant) vary.
#   pissa_in   → ImageNet ViT + PiSSA, honours --targets/--ranks (ablation row).
#   lora_in    → ImageNet ViT + LoRA (zero-init), honours --targets/--ranks (ablation row).
#   full_ft    → DINOv3 backbone, no PEFT (all params trainable). --ranks/--targets ignored.
#                Isolates backbone from adaptation strategy: same backbone as
#                lora/pissa, same training recipe as ResNet50's full FT.
_METHODS = ("lora", "pissa", "selafd", "pissa_in", "lora_in", "full_ft")
_TARGETS = ("attn", "qv", "all")
_DEFAULT_SHOTS = [1, 2, 5, 10, 20, 50, 100, 200]

# SelaFD hyperparameters are pinned to the paper values; the sweep only
# varies (variant, lr, seed) for this method.  --ranks / --targets are
# ignored when method == "selafd".
_SELAFD_FIXED = {
    "lora_rank": 4,
    "lora_alpha": 4,
    "adapter_ratio": 0.5,
    "parallel_scale": 0.2,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_lr(lr: float) -> str:
    s = f"{lr:.0e}"  # e.g. "1e-04"
    mant, exp = s.split("e")
    exp = int(exp)
    return f"{mant.rstrip('0').rstrip('.') or '1'}e{exp:+d}".replace("+", "p").replace("-", "m")


def make_factory(
    method: str,
    variant: str,
    rank: int,
    target_modules: str,
):
    """Return a callable(n_classes) -> nn.Module pinned to (method, rank, targets).

    For ``method == "selafd"``: returns an ImageNet ViT with LoRA(q,v) +
    serial/parallel adapters, hyperparameters pinned to ``_SELAFD_FIXED``
    (paper values). ``rank`` and ``target_modules`` are ignored.

    For ``method == "pissa_in"``: returns an ImageNet ViT with PiSSA
    adapters. Honours ``rank`` and ``target_modules``.

    For ``method == "lora_in"``: returns an ImageNet ViT with standard
    LoRA adapters (zero-init B). Honours ``rank`` and ``target_modules``.

    For ``method == "full_ft"``: returns a DINOv3 backbone with all
    parameters trainable (no PEFT). ``rank`` and ``target_modules`` are
    ignored.
    """
    if method == "full_ft":
        def factory(n_classes: int):
            return build_dinov3_full_ft(n_classes, variant=variant)
        return factory

    if method == "selafd":
        def factory(n_classes: int):
            return build_vit_imagenet_selafd(
                n_classes,
                variant=variant,
                **_SELAFD_FIXED,
            )
        return factory

    if method == "pissa_in":
        def factory(n_classes: int):
            return build_vit_imagenet_lora(
                n_classes,
                variant=variant,
                rank=rank,
                alpha=2 * rank,   # match build_dinov3_lora's alpha=None path
                target_modules=target_modules,
                init_lora_weights="pissa",
            )
        return factory

    if method == "lora_in":
        def factory(n_classes: int):
            return build_vit_imagenet_lora(
                n_classes,
                variant=variant,
                rank=rank,
                alpha=2 * rank,
                target_modules=target_modules,
                init_lora_weights=True,   # standard LoRA zero-init
            )
        return factory

    if method == "lora":
        kw = dict(use_dora=False, init_lora_weights=True)
    elif method == "pissa":
        kw = dict(use_dora=False, init_lora_weights="pissa")
    else:
        raise ValueError(method)

    def factory(n_classes: int):
        return build_dinov3_lora(
            n_classes,
            variant=variant,
            rank=rank,
            alpha=None,          # → 2 * rank
            target_modules=target_modules,
            **kw,
        )

    return factory


def config_name(method: str, lr: float, rank: int, targets: str, seed: int) -> str:
    if method == "selafd":
        # rank/targets meaningless for selafd; omit so configs collapse.
        return f"vit_selafd_lr{_format_lr(lr)}_s{seed}"
    if method == "full_ft":
        # rank/targets meaningless for full FT; omit so configs collapse.
        return f"dinov3_full_ft_lr{_format_lr(lr)}_s{seed}"
    if method == "pissa_in":
        return f"vit_in_pissa_lr{_format_lr(lr)}_r{rank}_{targets}_s{seed}"
    if method == "lora_in":
        return f"vit_in_lora_lr{_format_lr(lr)}_r{rank}_{targets}_s{seed}"
    return f"dinov3_vits16_{method}_lr{_format_lr(lr)}_r{rank}_{targets}_s{seed}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--methods", nargs="+", choices=_METHODS, default=["lora"])
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4])
    p.add_argument("--ranks", nargs="+", type=int, default=[8, 16])
    p.add_argument("--targets", nargs="+", choices=_TARGETS, default=list(_TARGETS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument(
        "--variant", nargs="+", default=["vits16"], choices=["vits16", "vitb16"],
        help="ViT variant(s) to sweep over.",
    )
    p.add_argument("--source", default="glasgow_5")
    p.add_argument("--target", default="mad_5")
    p.add_argument("--scheme", default=None,
                   help="Label scheme name (e.g. har5, glasgow6, mad10). "
                        "If None, auto-detected from source/target pair.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-shots", nargs="+", type=int, default=_DEFAULT_SHOTS)
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument(
        "--probes", nargs="+", choices=["knn", "linear"],
        default=["knn", "linear"],
        help="Probes to fit on the target features. The paper tables read "
             "only linear_C1.0, so `--probes linear` roughly halves runtime "
             "when regenerating results.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output-dir", default=str(REPO_ROOT / "results" / "cross_distribution"),
    )
    p.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT / "results" / "cache" / "checkpoints"),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the configurations that would run and exit.",
    )
    return p.parse_args()


_MAD_ROOT = REPO_ROOT / "data" / "MAD"
_GLASGOW_ROOT = REPO_ROOT / "data" / "Glasgow"

_DATASETS = {
    "glasgow_5":      lambda: GlasgowDataset(_GLASGOW_ROOT, exclude_classes=HAR5_EXCLUDE),
    "glasgow_young":  lambda: GlasgowDataset(_GLASGOW_ROOT, datasets=[1, 2, 3, 4, 5], subset_name="young"),
    "glasgow_mature": lambda: GlasgowDataset(
        _GLASGOW_ROOT,
        datasets=[6, 7],
        exclude_dpids=GLASGOW_MATURE_YOUNG_EXCLUDE,
        subset_name="mature",
    ),
    "mad_5":     lambda: MADDataset(_MAD_ROOT, activities=GLASGOW_OVERLAP_ACTIVITIES),
    "mad_sub12": lambda: MADDataset(_MAD_ROOT, subcategories=[1, 2]),
    "mad_sub3":  lambda: MADDataset(_MAD_ROOT, subcategories=[3]),
}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir)

    log_path = output_dir / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_path, level="DEBUG")

    if args.source not in _DATASETS or args.target not in _DATASETS:
        raise ValueError(
            f"--source / --target must be in {list(_DATASETS)} (extend _DATASETS if needed)"
        )

    # Sweep grid.  For "selafd", rank/targets are pinned (see _SELAFD_FIXED),
    # so iterate over (variant, lr, seed) only and dedupe by config_name.
    raw_configs = list(itertools.product(
        args.methods, args.variant, args.lrs, args.ranks, args.targets, args.seeds,
    ))
    seen: set = set()
    configs: list = []
    for cfg in raw_configs:
        method, variant, lr, rank, targets, seed = cfg
        key = (method, variant, config_name(method, lr, rank, targets, seed))
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)

    logger.info(
        f"Sweep: {len(configs)} configs  |  variants={args.variant}  "
        f"|  {args.source} → {args.target}  |  log → {log_path}"
    )

    if args.dry_run:
        for i, (m, v, lr, r, t, s) in enumerate(configs, 1):
            print(f"[{i:>3}/{len(configs)}] {v}  {config_name(m, lr, r, t, s)}")
        return

    source_ds = _DATASETS[args.source]()
    target_ds = _DATASETS[args.target]()
    _PROBES = {
        "knn": lambda: KNNProbe(k=10, metric="cosine"),
        "linear": lambda: LinearProbe(C=1.0),
    }
    probes = [_PROBES[name]() for name in args.probes]

    for i, (method, variant, lr, rank, targets, seed) in enumerate(configs, 1):
        name = f"{variant}_{config_name(method, lr, rank, targets, seed)}"
        out_path = output_dir / f"{name}__{args.source}__{args.target}.json"

        if out_path.exists():
            logger.info(f"[{i:>3}/{len(configs)}] skip (exists): {out_path.name}")
            continue

        logger.info(f"[{i:>3}/{len(configs)}] running {name}")
        seed_everything(seed)

        factory = make_factory(method, variant, rank, targets)
        try:
            result = run_cross_few_shot_evaluation(
                model_name=name,
                model_factory=factory,
                preprocess_fn=preprocess_batch,
                target_dataset=target_ds,
                probes=probes,
                device=device,
                source_dataset=source_ds,
                batch_size=args.batch_size,
                lr=lr,
                epochs=args.epochs,
                n_shots_list=args.n_shots,
                n_repeats=args.n_repeats,
                seed=42,                # fixed few-shot draws across configs
                checkpoint_dir=ckpt_dir,
                scheme=args.scheme,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"[{i}] config {name} crashed: {exc}")
            continue

        # Tag the result with the swept hyperparameters for later aggregation.
        if method == "selafd":
            result["sweep"] = {
                "method": method,
                "lr": lr,
                "seed": seed,
                "variant": variant,
                "epochs": args.epochs,
                **_SELAFD_FIXED,
            }
        elif method == "full_ft":
            result["sweep"] = {
                "method": method,
                "lr": lr,
                "seed": seed,
                "variant": variant,
                "epochs": args.epochs,
            }
        else:
            result["sweep"] = {
                "method": method,
                "lr": lr,
                "rank": rank,
                "alpha": 2 * rank,
                "targets": targets,
                "seed": seed,
                "variant": variant,
                "epochs": args.epochs,
            }
        out_path.write_text(json.dumps(result, indent=2))
        logger.info(f"  saved → {out_path}")

    logger.info("sweep done")


if __name__ == "__main__":
    main()
