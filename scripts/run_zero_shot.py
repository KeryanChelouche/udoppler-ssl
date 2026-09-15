#!/usr/bin/env python3
"""Zero-shot transfer: source-trained head applied directly to the target,
with no target-side probe refit.

Context: Table 2 measures 10-shot transfer, which is a "lab-to-field"
protocol in the sense that no target-side fine-tuning happens, but it does
refit a linear probe on 10 labeled target samples per class. This script
answers the stricter question — what if the target has *zero* labels? —
by reusing the exact backbone fine-tuned on the source (same checkpoint
run_final_method_comparison.sh / run_ablation.sh
already produced) and evaluating it on the target with its classification
head intact.

This script NEVER fine-tunes. It only evaluates already-cached checkpoints
(built by the scripts above) and skips any (method, pair, seed) whose
checkpoint isn't on disk — it will not silently kick off an expensive
fresh fine-tune. Zero-shot is only meaningful for methods with a
source-trained classification head, so frozen linear-probe baselines are
not included here.

Usage
-----
    # everything (all methods x all 3 shifts x 3 seeds) — the paper default
    python scripts/run_zero_shot.py

    # restrict scope
    python scripts/run_zero_shot.py --methods pissa full_ft --pairs cross-dataset
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from dotenv import load_dotenv
from loguru import logger

load_dotenv(REPO_ROOT / ".env")

from eval.cross_evaluation import find_label_scheme, run_supervised_cross_evaluation  # noqa: E402
from eval.datasets.glasgow import (  # noqa: E402
    GLASGOW_MATURE_YOUNG_EXCLUDE,
    GlasgowDataset,
    HAR5_EXCLUDE,
)
from eval.datasets.mad import GLASGOW_OVERLAP_ACTIVITIES, MADDataset  # noqa: E402
from eval.models.dinov3_full_ft import build_dinov3_full_ft  # noqa: E402
from eval.models.dinov3_lora import build_dinov3_lora  # noqa: E402
from eval.models.resnet import build_resnet, preprocess_batch  # noqa: E402
from eval.models.vit_imagenet_lora import build_vit_imagenet_lora  # noqa: E402
from eval.models.vit_imagenet_selafd import build_vit_imagenet_selafd  # noqa: E402

_MAD_ROOT = REPO_ROOT / "data" / "MAD"
_GLASGOW_ROOT = REPO_ROOT / "data" / "Glasgow"

DATASETS = {
    "glasgow_young":  lambda: GlasgowDataset(_GLASGOW_ROOT, datasets=[1, 2, 3, 4, 5], subset_name="young"),
    "glasgow_mature": lambda: GlasgowDataset(
        _GLASGOW_ROOT, datasets=[6, 7],
        exclude_dpids=GLASGOW_MATURE_YOUNG_EXCLUDE, subset_name="mature",
    ),
    "glasgow_5":  lambda: GlasgowDataset(_GLASGOW_ROOT, exclude_classes=HAR5_EXCLUDE),
    "mad_5":      lambda: MADDataset(_MAD_ROOT, activities=GLASGOW_OVERLAP_ACTIVITIES),
    "mad_sub12":  lambda: MADDataset(_MAD_ROOT, subcategories=[1, 2]),
    "mad_sub3":   lambda: MADDataset(_MAD_ROOT, subcategories=[3]),
}

# Headline transfer pairs — same three shifts as Table 2/3.
PAIRS = {
    "cross-age":         ("glasgow_young", "glasgow_mature"),
    "cross-orientation": ("mad_sub12", "mad_sub3"),
    "cross-dataset":     ("glasgow_5", "mad_5"),
}

# method -> (model_factory(n_classes), preprocess_fn, checkpoint model_name
# prefix). Prefixes must exactly match the model_name the checkpoint was
# saved under by run_peft_sweep.py / run_cross_few_shot.py.
METHODS = {
    "pissa": (
        lambda n: build_dinov3_lora(n, variant="vits16", rank=8, alpha=None, target_modules="all", init_lora_weights="pissa"),
        preprocess_batch, "vits16_dinov3_vits16_pissa_lr1em4_r8_all",
        "DINOv3-S + PiSSA (Ours)"),
    "lora_in": (
        lambda n: build_vit_imagenet_lora(n, variant="vits16", rank=8, alpha=16, target_modules="attn"),
        preprocess_batch, "vits16_vit_in_lora_lr1em4_r8_attn",
        "ImageNet ViT-S + LoRA"),
    "selafd": (
        lambda n: build_vit_imagenet_selafd(n, variant="vits16", lora_rank=4, lora_alpha=4, adapter_ratio=0.5, parallel_scale=0.2),
        preprocess_batch, "vits16_vit_selafd_lr1em4",
        "SelaFD-S"),
    "resnet50": (
        lambda n: build_resnet("resnet50", n), preprocess_batch, "resnet50",
        "ResNet50 (full FT)"),
    "full_ft": (
        lambda n: build_dinov3_full_ft(n, variant="vits16"), preprocess_batch, "vits16_dinov3_full_ft_lr1em4",
        "DINOv3-S (full FT)"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    p.add_argument("--pairs", nargs="+", choices=list(PAIRS), default=list(PAIRS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=60,
                   help="Must match the epochs the checkpoint was fine-tuned for.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "zero_shot"))
    p.add_argument("--checkpoint-dir", default=str(REPO_ROOT / "results" / "cache" / "checkpoints"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir)

    n_run, n_skip_ckpt, n_skip_exists = 0, 0, 0

    for pair_name in args.pairs:
        src_name, tgt_name = PAIRS[pair_name]
        for method in args.methods:
            factory, preprocess, prefix, label = METHODS[method]
            for seed in args.seeds:
                model_name = f"{prefix}_s{seed}"

                # Determine the checkpoint path exactly as the fine-tuning
                # scripts named it, without loading any data yet.
                scheme_name, _ = find_label_scheme(src_name, tgt_name)
                ckpt_path = ckpt_dir / f"{model_name}_ft_{src_name}_{scheme_name}_ep{args.epochs}.pt"
                out_path = output_dir / f"{model_name}__{src_name}__{tgt_name}.json"

                if out_path.exists():
                    n_skip_exists += 1
                    continue
                if not ckpt_path.exists():
                    logger.warning(f"skip (no checkpoint): {ckpt_path.name}")
                    n_skip_ckpt += 1
                    continue

                result = run_supervised_cross_evaluation(
                    model_name=model_name,
                    model_factory=factory,
                    preprocess_fn=preprocess,
                    train_dataset=DATASETS[src_name](),
                    test_dataset=DATASETS[tgt_name](),
                    device=device,
                    checkpoint_dir=ckpt_dir,
                    epochs=args.epochs,
                )
                result["shift"] = pair_name
                result["method_label"] = label
                out_path.write_text(json.dumps(result, indent=2))
                acc = result["probes"]["supervised"]["acc"] * 100
                f1 = result["probes"]["supervised"]["f1"] * 100
                logger.info(f"[{pair_name:18s}] {label:24s} s{seed}  acc={acc:5.2f}%  f1={f1:5.2f}%  -> {out_path.name}")
                n_run += 1

    logger.info(f"done: {n_run} run, {n_skip_exists} already existed, {n_skip_ckpt} skipped (no checkpoint)")


if __name__ == "__main__":
    main()
