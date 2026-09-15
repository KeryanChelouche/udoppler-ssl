#!/usr/bin/env bash
# Controlled 2x2 adapter ablation: initialization x layer coverage.
#
# Table 3 in the paper moves initialization and coverage together (attention-only
# LoRA -> all-layer PiSSA). This script fills in the two off-diagonal cells so the
# factors can be read separately, on the DINOv3-S backbone throughout:
#
#                 | attention-only        | all layers
#   LoRA  (B=0)   | reference             | coverage only
#   PiSSA (SVD)   | initialization only   | our recipe
#
# The two diagonal cells (LoRA attn, PiSSA all) already exist -- they are Table 3's
# DINOv3 column and step 1/2 of run_final_method_comparison.sh. run_peft_sweep.py
# skips any (config, seed, pair) whose JSON exists, so only the missing off-diagonal
# cells are computed.
#
# Scope: cross-dataset only. That is the shift where the recipe's advantage is
# largest and where the paper's claim about the two factors matters most, and
# restricting to it halves the cost.
#
# COST: unlike the other reproducers, this one FINE-TUNES. No checkpoints exist for
# `lora --targets all` or `pissa --targets attn`, so the two off-diagonal cells
# train 60 epochs x 3 seeds each: 6 fine-tuning runs. Budget several GPU-hours.
#
# Output: results/cross_distribution/vits16_dinov3_vits16_{lora,pissa}_*_s{0,1,2}__*.json
#         rendered as Table 4 by paper_tables.ipynb.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
OUT=results/cross_distribution
mkdir -p "$OUT"

# Shot levels / probes: the tables read only the 10-shot linear probe, so the
# defaults here are narrower than the other scripts to keep the run tractable.
N_SHOTS="${N_SHOTS:-10}"
PROBES="${PROBES:-linear}"

# Cross-dataset only (see Scope above). Add the cross-orientation line back to
# fill in that half of Table 4 as well.
PAIRS=(
  "glasgow_5  mad_5"         # Cross-dataset
  # "mad_sub12  mad_sub3"    # Cross-orientation
)

# All four (method, targets) combinations. The diagonal pair is included so the
# script is self-contained; those runs are no-ops when the JSONs already exist.
CONFIGS=(
  "lora   attn"      # reference
  "lora   all"       # coverage only
  "pissa  attn"      # initialization only
  "pissa  all"       # our recipe
)

for pair in "${PAIRS[@]}"; do
  read -r SRC TGT <<<"$pair"
  for cfg in "${CONFIGS[@]}"; do
    read -r METHOD TARGETS <<<"$cfg"
    echo ">>> DINOv3-S | $METHOD ($TARGETS)  $SRC -> $TGT"
    $PY scripts/run_peft_sweep.py \
        --methods "$METHOD" --targets "$TARGETS" \
        --ranks 8 --lrs 1e-4 \
        --epochs 60 --seeds 0 1 2 \
        --variant vits16 \
        --source "$SRC" --target "$TGT" \
        --n-shots $N_SHOTS \
        --probes $PROBES \
        --output-dir "$OUT"
  done
done

echo
echo "Controlled 2x2 ablation complete -> Table 4 in paper_tables.ipynb"
