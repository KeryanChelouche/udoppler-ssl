#!/usr/bin/env python3
"""Evaluate one or more models on one or more datasets.

Examples
--------
# k-NN + linear probe on Glasgow with a frozen DINOv3 ViT-B
python scripts/run_eval.py --model dinov3_vitb16 --dataset glasgow

# Grid: two models × two datasets
python scripts/run_eval.py \\
    --model dinov3_vits16 vit_small_imagenet \\
    --dataset glasgow mad \\
    --probes knn linear

# Force re-extraction (ignore cache)
python scripts/run_eval.py --model dinov3_vits16 --dataset glasgow --no-cache
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.datasets.glasgow import GlasgowDataset
from eval.datasets.mad import MADDataset
from eval.evaluation import run_evaluation
from eval.models.dinov3_full_ft import build_dinov3_full_ft
from eval.models.dinov3_lora import build_dinov3_lora
from eval.models.resnet import build_resnet, preprocess_batch
from eval.models.vit_imagenet import ViTImageNetExtractor
from eval.models.vit_imagenet_lora import build_vit_imagenet_lora
from eval.models.vit_imagenet_selafd import build_vit_imagenet_selafd
from eval.models.dinov3 import DINOv3Extractor
from eval.probes.knn import KNNProbe
from eval.probes.linear import LinearProbe
from eval.reporting import print_results, save_results
from eval.supervised import run_supervised_evaluation
from loguru import logger

# ── Registries ────────────────────────────────────────────────────────────────
# Add new models / datasets / probes here without touching the rest of the code.

MODEL_REGISTRY = {
    "dinov3_vits16":        lambda: DINOv3Extractor(variant="vits16"),
    "dinov3_vitb16":        lambda: DINOv3Extractor(variant="vitb16"),
    "vit_small_imagenet":   lambda: ViTImageNetExtractor(variant="vits16"),
    "vit_base_imagenet":    lambda: ViTImageNetExtractor(variant="vitb16"),
}

# Supervised models: fine-tuned end-to-end (bypass feature extraction + probes).
# Each entry is (model_factory_taking_n_classes, preprocess_fn).
SUPERVISED_REGISTRY = {
    "resnet50":                (lambda n: build_resnet("resnet50", n),                                  preprocess_batch),
    # Final-spec entries used by the in-distribution + cross-dataset tables:
    # PiSSA → r=8, alpha=16 (2*r via alpha=None), all linears, SVD init.
    # LoRA  → r=8, alpha=16, attn linears only, zero-B init.
    # Note: build_dinov3_lora's default alpha=32 is *not* what we want; passing
    # alpha=None routes through the 2*rank path that run_peft_sweep.py uses.
    "dinov3_vits16_pissa":     (lambda n: build_dinov3_lora(n, variant="vits16", rank=8, alpha=None, target_modules="all",  init_lora_weights="pissa"), preprocess_batch),
    "dinov3_vitb16_pissa":     (lambda n: build_dinov3_lora(n, variant="vitb16", rank=8, alpha=None, target_modules="all",  init_lora_weights="pissa"), preprocess_batch),
    # Fully fine-tuned DINOv3-S (no PEFT) — same backbone as the PiSSA row,
    # isolating adaptation strategy from backbone choice (Table 1/2, Fig. 1).
    "dinov3_vits16_full_ft":   (lambda n: build_dinov3_full_ft(n, variant="vits16"),                     preprocess_batch),
    "vit_base_imagenet_lora":  (lambda n: build_vit_imagenet_lora(n, variant="vitb16"),                 preprocess_batch),
    # ImageNet ViT-S + LoRA attn at the same final spec (r=8, alpha=16,
    # attn-only) for table-1/2 comparability.
    "vit_small_imagenet_lora": (lambda n: build_vit_imagenet_lora(n, variant="vits16", rank=8, alpha=16, target_modules="attn"),  preprocess_batch),
    "vit_small_imagenet_selafd": (lambda n: build_vit_imagenet_selafd(n, variant="vits16"),             preprocess_batch),
    "vit_base_imagenet_selafd":  (lambda n: build_vit_imagenet_selafd(n, variant="vitb16"),             preprocess_batch),
}

_MAD_ROOT = REPO_ROOT / "data" / "MAD"

DATASET_REGISTRY = {
    "glasgow":  lambda: GlasgowDataset(REPO_ROOT / "data" / "Glasgow"),
    "mad":      lambda: MADDataset(_MAD_ROOT),
}

PROBE_REGISTRY = {
    "knn":    lambda: KNNProbe(k=10, metric="cosine"),
    "linear": lambda: LinearProbe(C=1.0),
}

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    all_models = list(MODEL_REGISTRY) + list(SUPERVISED_REGISTRY)
    p.add_argument(
        "--model", nargs="+", choices=all_models, required=True,
        metavar="MODEL", help=f"One or more of: {all_models}",
    )
    p.add_argument(
        "--dataset", nargs="+", choices=DATASET_REGISTRY, required=True,
        metavar="DATASET", help=f"One or more of: {list(DATASET_REGISTRY)}",
    )
    p.add_argument(
        "--probes", nargs="+", choices=PROBE_REGISTRY,
        default=["knn", "linear"],
        metavar="PROBE", help=f"One or more of: {list(PROBE_REGISTRY)}",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Epochs for supervised (FT) models. Ignored for frozen models.",
    )
    p.add_argument(
        "--lr", type=float, default=1e-4,
        help="AdamW learning rate for supervised models. Ignored for frozen.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed for FT init/data shuffle (supervised models only).",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Re-extract features even if a cache file exists.",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "in_distribution"),
    )
    p.add_argument(
        "--skip-if-exists", action="store_true",
        help="Use a deterministic, hyperparam-encoded filename and skip the "
             "run if it already exists. Lets the script be re-run safely as "
             "a reproducibility entry point.",
    )
    return p.parse_args()


def _format_lr(lr: float) -> str:
    s = f"{lr:.0e}"
    mant, exp = s.split("e")
    exp = int(exp)
    return (
        f"{mant.rstrip('0').rstrip('.') or '1'}e{exp:+d}"
        .replace("+", "p").replace("-", "m")
    )


def _deterministic_name(
    model_name: str,
    dataset_name: str,
    is_supervised: bool,
    epochs: int,
    lr: float,
    seed: int,
) -> str:
    if is_supervised:
        return (
            f"{model_name}_ep{epochs}_lr{_format_lr(lr)}_s{seed}"
            f"__{dataset_name}.json"
        )
    return f"{model_name}__{dataset_name}.json"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_path, level="DEBUG")
    logger.info(f"Device: {device}  |  log → {log_path}")

    for model_name in args.model:
        for dataset_name in args.dataset:
            is_supervised = model_name in SUPERVISED_REGISTRY

            # Skip-if-exists: deterministic filename keyed by hyperparams so a
            # stale ep=30 run won't shadow a fresh ep=60 one.
            if args.skip_if_exists:
                det_path = output_dir / _deterministic_name(
                    model_name, dataset_name, is_supervised,
                    args.epochs, args.lr, args.seed,
                )
                if det_path.exists():
                    logger.info(f"skip (exists): {det_path.name}")
                    continue

            dataset = DATASET_REGISTRY[dataset_name]()

            if is_supervised:
                factory, preprocess_fn = SUPERVISED_REGISTRY[model_name]
                _seed_everything(args.seed)
                result = run_supervised_evaluation(
                    model_name=model_name,
                    model_factory=factory,
                    preprocess_fn=preprocess_fn,
                    dataset=dataset,
                    device=device,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    epochs=args.epochs,
                )
                # Tag the result with the hyperparams used so aggregation
                # scripts can filter by config.
                result["config"] = {
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                }
            else:
                model  = MODEL_REGISTRY[model_name]()
                probes = [PROBE_REGISTRY[p]() for p in args.probes]
                result = run_evaluation(
                    model=model,
                    dataset=dataset,
                    probes=probes,
                    device=device,
                    batch_size=args.batch_size,
                    use_cache=not args.no_cache,
                )

            print_results(result)

            if args.skip_if_exists:
                out = output_dir / _deterministic_name(
                    model_name, dataset_name, is_supervised,
                    args.epochs, args.lr, args.seed,
                )
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = output_dir / f"{model_name}__{dataset_name}__{ts}.json"
            save_results(result, out)
            logger.info(f"Saved → {out}")


if __name__ == "__main__":
    main()
