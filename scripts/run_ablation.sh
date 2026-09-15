#!/usr/bin/env bash
# Ablation table reproducer — 2 rows × 4 columns.
#
# Layout:
#                    | Cross-orientation                    | Cross-dataset
#                    | (mad_sub12 → mad_sub3)               | (glasgow_5 → mad_5)
#                    | ImageNet ViT-S | DINOv3 ViT-S        | ImageNet ViT-S | DINOv3 ViT-S
#   Row 1: LoRA      | lora_in (attn) | lora (attn)         | lora_in (attn) | lora (attn)
#   Row 2: + PiSSA   | pissa_in (all) | pissa (all, Ours)   | pissa_in (all) | pissa (all, Ours)
#
# Shared hyperparameters: ep=60, lr=1e-4, r=8, alpha=16, 3 seeds, n_repeats=10.
# Cell numbers come from results/cross_distribution/vits16_*_s*__{SRC}__{TGT}.json
# (aggregated across the 3 seeds at 10-shot).
#
# Idempotent: run_peft_sweep.py checks out_path.exists() before running,
# so cells already produced by run_final_method_comparison.sh (all except
# `pissa_in all` on mad_sub12 → mad_sub3) are no-ops.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python

# Shot levels: override with e.g. `N_SHOTS=10 bash scripts/<this>.sh` to
# regenerate only the 10-shot level the paper reports (much faster — the
# probe refits dominate runtime once checkpoints are cached).
N_SHOTS="${N_SHOTS:-1 2 5 10 20 50 100 200}"

# Probes: override with `PROBES=linear` to skip the k-NN probe. No paper table
# reads it, and it is a large fraction of the post-checkpoint runtime.
PROBES="${PROBES:-knn linear}"
EP60_OUT=results/cross_distribution
mkdir -p "$EP60_OUT"

# Two transfer pairs × two backbones × two adaptation methods = 8 cells.
PAIRS=(
  "mad_sub12  mad_sub3"      # Cross-orientation
  "glasgow_5  mad_5"         # Cross-dataset
)

# Each config is a (method, targets) pair.  The `_in` suffix distinguishes
# the ImageNet backbone from DINOv3 (run_peft_sweep.py maps the method name
# to the appropriate factory).
DINOV3_CONFIGS=(
  "lora   attn"              # Row 1, DINOv3 columns
  "pissa  all"               # Row 2, DINOv3 columns (Ours at cross-dataset)
)
IMAGENET_CONFIGS=(
  "lora_in   attn"           # Row 1, ImageNet columns
  "pissa_in  all"            # Row 2, ImageNet columns
)

for pair in "${PAIRS[@]}"; do
  read -r SRC TGT <<<"$pair"

  # DINOv3 backbone
  for cfg in "${DINOV3_CONFIGS[@]}"; do
    read -r METHOD TARGETS <<<"$cfg"
    echo ">>> DINOv3    | $METHOD ($TARGETS)  $SRC -> $TGT"
    $PY scripts/run_peft_sweep.py \
        --methods "$METHOD" --targets "$TARGETS" \
        --ranks 8 --lrs 1e-4 \
        --epochs 60 --seeds 0 1 2 \
        --variant vits16 \
        --source "$SRC" --target "$TGT" \
        --n-shots $N_SHOTS \
        --probes $PROBES \
      --output-dir "$EP60_OUT"
  done

  # ImageNet backbone
  for cfg in "${IMAGENET_CONFIGS[@]}"; do
    read -r METHOD TARGETS <<<"$cfg"
    echo ">>> ImageNet  | $METHOD ($TARGETS)  $SRC -> $TGT"
    $PY scripts/run_peft_sweep.py \
        --methods "$METHOD" --targets "$TARGETS" \
        --ranks 8 --lrs 1e-4 \
        --epochs 60 --seeds 0 1 2 \
        --variant vits16 \
        --source "$SRC" --target "$TGT" \
        --n-shots $N_SHOTS \
        --probes $PROBES \
      --output-dir "$EP60_OUT"
  done
done

echo
echo "Ablation runs complete."
