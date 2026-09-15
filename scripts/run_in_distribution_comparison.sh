#!/usr/bin/env bash
# In-distribution method comparison: train + test on the same dataset with
# subject-disjoint StratifiedGroupKFold(5) CV.  One number per (model, dataset)
# reported as mean ± std across 5 folds.
#
# Final hyperparameters:
#   FT methods:  ep=60, lr=1e-4, seed=42 (single)
#     PiSSA   → DINOv3 ViT-S, r=8, alpha=16, all linears, SVD init
#     LoRA    → DINOv3 ViT-S, r=8, alpha=16, attn linears only
#     SelaFD  → ImageNet ViT-S + LoRA(q,v) r=alpha=4 + serial/parallel adapters
#     ResNet50 → standard supervised FT
#     DINOv3 full FT → same backbone as PiSSA, no PEFT (all weights trainable);
#       isolates adaptation strategy from backbone choice (Table 1/2, Fig. 1)
#   Frozen:      one deterministic linear/k-NN probe per fold
#     DINOv3 ViT-S, ViT-S ImageNet — features cached in results/cache/features/
#
# Idempotent: run_eval.py --skip-if-exists uses a deterministic filename keyed
# by (model, dataset, epochs, lr, seed) and skips a config whose JSON already
# exists.  So this script is safe to re-launch — already-done runs are no-ops.
#
# Output naming (under results/in_distribution/):
#   FT:     {model}_ep60_lr1em4_s42__{dataset}.json
#   Frozen: {model}__{dataset}.json
#
# Note: stale legacy files like {model}__{dataset}__{timestamp}.json from
# earlier runs will be ignored by the skip check (different naming).  Remove
# them by hand if you want the directory tidy.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
OUT=results/in_distribution

DATASETS=(glasgow mad)

# ── 1. FT methods at ep=60 / lr=1e-4 / seed=42  ─────────────────────────
# vit_base_imagenet_selafd is included to demonstrate that the original
# SelaFD architecture (ViT-B) does not outperform our ViT-S variant on
# these datasets — supports the paper's use of ViT-S throughout.
for ds in "${DATASETS[@]}"; do
  for model in dinov3_vits16_pissa dinov3_vitb16_pissa vit_small_imagenet_lora vit_small_imagenet_selafd vit_base_imagenet_selafd resnet50 dinov3_vits16_full_ft; do
    echo ">>> $model on $ds  (ep=60, lr=1e-4, seed=42)"
    $PY scripts/run_eval.py \
        --model "$model" \
        --dataset "$ds" \
        --epochs 60 --lr 1e-4 --seed 42 \
        --output-dir "$OUT" \
        --skip-if-exists
  done
done

# ── 2. Frozen baselines (linear + k-NN probes)  ────────────────────────
# ViT-B frozen rows are included to show that backbone scale alone does
# not close the gap to our method — supports the paper's ViT-S choice.
# Features are cached in results/cache/features/, so these runs only probe and
# finish in seconds each.
for ds in "${DATASETS[@]}"; do
  for model in dinov3_vits16 vit_small_imagenet dinov3_vitb16 vit_base_imagenet; do
    echo ">>> $model on $ds  (frozen)"
    $PY scripts/run_eval.py \
        --model "$model" \
        --dataset "$ds" \
        --probes linear knn \
        --output-dir "$OUT" \
        --skip-if-exists
  done
done

echo
echo "Done.  Aggregate with the in-distribution analysis script."
