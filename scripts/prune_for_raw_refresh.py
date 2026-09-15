#!/usr/bin/env python3
"""Preflight for regenerating Tables 2 & 3 with raw per-fit scores.

`eval/cross_few_shot.py` now persists every probe fit (`probes.<probe>[i].fits`)
so that paired significance tests can be run post hoc. Result files written
before that change only carry `mean_acc`/`std_acc`, so they must be
regenerated — but the reproducer scripts are idempotent and skip any run whose
JSON already exists, so the stale files have to be moved out of the way first.

Regeneration is cheap *only* if the fine-tuned backbone is already cached: the
run then reduces to a checkpoint load, one forward pass over the target, and
the probe refits. This script therefore moves aside only the files whose
checkpoint is on disk, and loudly reports any whose checkpoint is missing —
those are left in place so that re-running the reproducers cannot silently
kick off a 60-epoch fine-tune.

    python scripts/prune_for_raw_refresh.py            # dry run
    python scripts/prune_for_raw_refresh.py --apply
"""
import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CROSS_DIR = REPO_ROOT / "results" / "cross_distribution"
CKPT_DIR = REPO_ROOT / "results" / "cache" / "checkpoints"
BACKUP_DIR = CROSS_DIR / "_pre_raw_backup"
EPOCHS = 60


def has_fits(d: dict) -> bool:
    for probe in d.get("probes", {}).values():
        if isinstance(probe, list) and probe and probe[0].get("fits") is not None:
            return True
    return False


def checkpoint_for(fp: Path, d: dict) -> Path | None:
    """Checkpoint a rerun of this result would need, or None if it needs none.

    Derived from the *filename*, not the JSON's ``model`` field: some older
    results were written before ``run_peft_sweep.py`` prefixed the run name
    with the ViT variant, so their stored ``model`` no longer matches the
    checkpoint a rerun would look for. The filename is authoritative because
    the reproducers key their skip-if-exists check on it.

    Filename shapes:
      ``{name}__{src}__{tgt}.json``   (run_peft_sweep)   -> {name}_ft_{src}_...
      ``{display}__{tgt}.json``       (run_cross_few_shot) -> {display}_...
        where ``display`` already ends in ``_ft_{src}`` or ``_frozen_{src}``.
    """
    parts = fp.stem.split("__")
    if len(parts) == 3:
        name, src, _tgt = parts
        stem = f"{name}_ft_{src}"
    else:
        stem = parts[0]
    if "_frozen_" in stem:
        return None            # frozen runs never fine-tune
    return CKPT_DIR / f"{stem}_{d.get('label_scheme')}_ep{EPOCHS}.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true",
                   help="Actually move the files (default is a dry run).")
    args = p.parse_args()

    to_move, blocked, already = [], [], []
    for fp in sorted(CROSS_DIR.glob("*.json")):
        d = json.loads(fp.read_text())
        if has_fits(d):
            already.append(fp)
            continue
        ckpt = checkpoint_for(fp, d)
        if ckpt is None or ckpt.exists():
            to_move.append((fp, ckpt))
        else:
            blocked.append((fp, ckpt))

    print(f"{len(already):>4} already have raw fits (left alone)")
    print(f"{len(to_move):>4} regenerable from cache (no training)")
    print(f"{len(blocked):>4} BLOCKED — checkpoint missing, would retrain")

    for fp, ckpt in blocked:
        print(f"  ! {fp.name}\n      needs {ckpt.name}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to move the regenerable files to "
              f"{BACKUP_DIR.relative_to(REPO_ROOT)}/")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for fp, _ in to_move:
        shutil.move(str(fp), str(BACKUP_DIR / fp.name))
    print(f"\nMoved {len(to_move)} file(s) to {BACKUP_DIR.relative_to(REPO_ROOT)}/")
    print("Now re-run, in this order:")
    print("  N_SHOTS=10 bash scripts/run_final_method_comparison.sh")
    print("  N_SHOTS=10 bash scripts/run_ablation.sh")


if __name__ == "__main__":
    main()
