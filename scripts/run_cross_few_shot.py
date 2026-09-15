#!/usr/bin/env python3
"""Cross-dataset few-shot: fine-tune on source, few-shot probe on target.

Compare supervised backbones (ImageNet → source fine-tune) against frozen
foundation-model backbones that skip source fine-tuning entirely.

Examples
--------
# ResNet50 fine-tuned on Glasgow-5, few-shot on MAD-5
python scripts/run_cross_few_shot.py \\
    --model resnet50 \\
    --source-dataset glasgow_5 \\
    --target-dataset mad_5

# Both directions
python scripts/run_cross_few_shot.py \\
    --model resnet50 \\
    --source-dataset glasgow_5 \\
    --target-dataset mad_5 \\
    --bidirectional
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

from eval.cross_evaluation import CROSS_LABEL_SCHEMES
from eval.cross_few_shot import run_cross_few_shot_evaluation
from eval.datasets.glasgow import (
    GLASGOW_MATURE_YOUNG_EXCLUDE,
    GlasgowDataset,
    HAR5_EXCLUDE,
)
from eval.datasets.mad import GLASGOW_OVERLAP_ACTIVITIES, MADDataset
from eval.models.dinov3 import DINOv3Extractor
from eval.models.frozen import FrozenBackboneModule, identity_preprocess
from eval.models.resnet import build_resnet, preprocess_batch
from eval.models.vit_imagenet import ViTImageNetExtractor
from eval.probes.knn import KNNProbe
from eval.probes.linear import LinearProbe
from loguru import logger

# ── Registries ────────────────────────────────────────────────────────────────

_MAD_ROOT = REPO_ROOT / "data" / "MAD"

SUPERVISED_REGISTRY = {
    "resnet50": (lambda n: build_resnet("resnet50", n), preprocess_batch),
    # Frozen backbones (pair with --freeze; no fine-tuning).  Each wraps
    # a FeatureExtractor that handles its own preprocessing, so we use
    # identity_preprocess to avoid double-preprocessing.
    "dinov3_vits16_frozen":      (lambda n: FrozenBackboneModule(DINOv3Extractor(variant="vits16")),    identity_preprocess),
    "vit_small_imagenet_frozen": (lambda n: FrozenBackboneModule(ViTImageNetExtractor(variant="vits16")), identity_preprocess),
}

DATASET_REGISTRY = {
    "glasgow":     lambda: GlasgowDataset(REPO_ROOT / "data" / "Glasgow"),
    "glasgow_5":    lambda: GlasgowDataset(REPO_ROOT / "data" / "Glasgow", exclude_classes=HAR5_EXCLUDE),
    "glasgow_young":  lambda: GlasgowDataset(REPO_ROOT / "data" / "Glasgow", datasets=[1, 2, 3, 4, 5], subset_name="young"),
    # glasgow_mature = D6-7 minus 5 young-aged participants (see
    # GLASGOW_MATURE_YOUNG_EXCLUDE).  Two of those pids are also present
    # in glasgow_young (genuine duplicates), the other three are unique
    # to D6-7 but young; both groups inflate cohort-shift scores when
    # left in.  Drop them explicitly here so the dataset name keeps
    # meaning "older cohort, no leakage".
    "glasgow_mature": lambda: GlasgowDataset(
        REPO_ROOT / "data" / "Glasgow",
        datasets=[6, 7],
        exclude_dpids=GLASGOW_MATURE_YOUNG_EXCLUDE,
        subset_name="mature",
    ),
    "mad":         lambda: MADDataset(_MAD_ROOT),
    "mad_5":       lambda: MADDataset(_MAD_ROOT, activities=GLASGOW_OVERLAP_ACTIVITIES),
    "mad_sub12":   lambda: MADDataset(_MAD_ROOT, subcategories=[1, 2]),
    "mad_sub3":    lambda: MADDataset(_MAD_ROOT, subcategories=[3]),
}

PROBE_REGISTRY = {
    "knn":    lambda: KNNProbe(k=10, metric="cosine"),
    "linear": lambda: LinearProbe(C=1.0),
}

_DEFAULT_N_SHOTS = [1, 2, 5, 10, 20, 50, 100, 200]

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", nargs="+", choices=SUPERVISED_REGISTRY, required=True,
        metavar="MODEL",
    )
    p.add_argument(
        "--source-dataset", choices=DATASET_REGISTRY, default=None,
        metavar="DATASET",
        help="Source dataset for fine-tuning. Omit for ImageNet-only baseline.",
    )
    p.add_argument(
        "--target-dataset", nargs="+", choices=DATASET_REGISTRY, required=True,
        metavar="DATASET",
    )
    p.add_argument(
        "--probes", nargs="+", choices=PROBE_REGISTRY,
        default=["knn", "linear"], metavar="PROBE",
    )
    p.add_argument(
        "--n-shots", nargs="+", type=int, default=_DEFAULT_N_SHOTS,
        metavar="N",
    )
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument(
        "--bidirectional", action="store_true",
        help="Also run the reverse direction (target→source).",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="One or more FT seeds. Each seed produces a separate JSON; "
             "the model_name encodes the seed when more than one is given. "
             "Ignored when --freeze is set (frozen runs are deterministic).",
    )
    p.add_argument(
        "--freeze", action="store_true",
        help="Skip fine-tuning and use the backbone as-is. Intended for the "
             "frozen registry entries (e.g. dinov3_vits16_frozen).",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "cross_distribution"),
    )
    p.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT / "results" / "cache" / "checkpoints"),
        help="Directory to cache fine-tuned backbone weights.",
    )
    p.add_argument(
        "--scheme", choices=list(CROSS_LABEL_SCHEMES), default=None,
        help="Force a specific label scheme. Default: first scheme covering "
             "both datasets.",
    )
    return p.parse_args()


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
    logger.info(f"Shot levels: {args.n_shots}  |  Repeats: {args.n_repeats}")

    # Build (source, target) pairs.
    pairs = [
        (args.source_dataset, t) for t in args.target_dataset
    ]
    if args.bidirectional and args.source_dataset is not None:
        pairs += [
            (t, args.source_dataset) for t in args.target_dataset
        ]

    # Frozen runs are deterministic; collapse --seeds to a single value
    # so we don't waste compute re-running identical configs.
    seeds = [args.seeds[0]] if args.freeze else list(args.seeds)
    encode_seed = len(seeds) > 1

    for model_name in args.model:
        factory, preprocess_fn = SUPERVISED_REGISTRY[model_name]

        for source_name, target_name in pairs:
            source_ds = DATASET_REGISTRY[source_name]() if source_name else None
            target_ds = DATASET_REGISTRY[target_name]()

            for seed in seeds:
                # Encode seed in model_name so the FT checkpoint cache is
                # keyed per-seed (downstream code uses model_name as the
                # checkpoint filename prefix).
                run_model_name = f"{model_name}_s{seed}" if encode_seed else model_name

                if args.freeze:
                    tag = f"frozen_{source_name}" if source_name else "frozen_imagenet"
                else:
                    tag = f"ft_{source_name}" if source_name else "imagenet"

                # Deterministic, hyperparam-keyed filename so the script is
                # idempotent: an already-computed (model, seed, pair) is a
                # no-op. Lets the reproducer be re-launched safely and skip
                # runs whose inputs (e.g. an unchanged Glasgow split) are done.
                out = output_dir / f"{run_model_name}_{tag}__{target_name}.json"
                if out.exists():
                    logger.info(f"skip (exists): {out.name}")
                    continue

                _seed_everything(seed)

                probes = [PROBE_REGISTRY[p]() for p in args.probes]
                result = run_cross_few_shot_evaluation(
                    model_name=run_model_name,
                    model_factory=factory,
                    preprocess_fn=preprocess_fn,
                    target_dataset=target_ds,
                    probes=probes,
                    device=device,
                    source_dataset=source_ds,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    epochs=args.epochs,
                    n_shots_list=args.n_shots,
                    n_repeats=args.n_repeats,
                    checkpoint_dir=Path(args.checkpoint_dir),
                    scheme=args.scheme,
                    freeze_backbone=args.freeze,
                )

                out.write_text(json.dumps(result, indent=2))
                logger.info(f"Saved → {out}")


if __name__ == "__main__":
    main()
