#!/usr/bin/env python3
"""Paired significance tests over the shared folds / shared few-shot draws.

Every model in this paper is evaluated on *identical* splits:

  * Table 1  — ``StratifiedGroupKFold(5, shuffle=True, random_state=42)``,
    so all rows share the same 5 subject-disjoint test folds.
  * Tables 2 & 3 — the same 5 target folds, plus few-shot support draws
    seeded by ``seed*10_000 + fold_idx*1_000 + rep`` with ``seed=42`` pinned
    in both entry points (``run_peft_sweep.py``, ``run_cross_few_shot.py``).
    The draw RNG does *not* depend on the fine-tuning seed, so fit *k* of
    method A and fit *k* of method B saw the same support set and the same
    test fold.

That shared structure lets us test differences pairwise instead of comparing
two independent means, which is much better powered at these sample sizes.

Two levels are reported, because they answer different questions:

  fold-level (n=5)   Folds are the only truly independent unit — subjects are
                     disjoint across them. Repeats are *nested* resampling
                     inside a fold, so this is the conservative, defensible
                     test. It is also badly underpowered: with n=5 the
                     two-sided Wilcoxon floor is p=0.0625, i.e. it can never
                     reach 0.05 no matter how consistent the difference is.

  draw-level (n=50)  One point per (fold, repeat), averaged over the 3
                     fine-tuning seeds. Higher resolution and it lets the
                     single-seed frozen rows participate, but the 10 repeats
                     within a fold are correlated, so its p-values are
                     anti-conservative. Treat as descriptive.

The win/loss count over the 50 draws is the most robust summary: it makes no
distributional assumption and shows whether a gap is consistent or driven by
one fold.

Usage
-----
    .venv/bin/python scripts/paired_tests.py              # all three tables
    .venv/bin/python scripts/paired_tests.py --tables 1
    .venv/bin/python scripts/paired_tests.py --shot 10 --metric acc
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIST_DIR = REPO_ROOT / "results" / "in_distribution"
CROSS_DIR = REPO_ROOT / "results" / "cross_distribution"

EPOCHS, LR_TAG, RANK = 60, "1em4", 8
FT_SEEDS = (0, 1, 2)

CROSS_PAIRS = {
    "Cross-age":         ("glasgow_young", "glasgow_mature"),
    "Cross-orientation": ("mad_sub12",     "mad_sub3"),
    "Cross-dataset":     ("glasgow_5",     "mad_5"),
}

# ── Table 1: per-fold accuracies ────────────────────────────────────────────

T1_FT = {
    "ResNet50 (ft)":            "resnet50",
    "DINOv3-S (full FT)":       "dinov3_vits16_full_ft",
    "SelaFD ViT-S":             "vit_small_imagenet_selafd",
    "ViT-S + LoRA":             "vit_small_imagenet_lora",
    "Ours-S (DINOv3-S+PiSSA)":  "dinov3_vits16_pissa",
    "SelaFD ViT-B":             "vit_base_imagenet_selafd",
    "Ours-B (DINOv3-B+PiSSA)":  "dinov3_vitb16_pissa",
}
T1_FROZEN = {
    "ViT-S ImageNet (frozen)":  "vit_small_imagenet",
    "DINOv3-S (frozen)":        "dinov3_vits16",
    "ViT-B ImageNet (frozen)":  "vit_base_imagenet",
    "DINOv3-B (frozen)":        "dinov3_vitb16",
}

# The claims the paper actually makes, as one Holm family per dataset.
T1_COMPARISONS = [
    ("Ours-S (DINOv3-S+PiSSA)", "SelaFD ViT-B"),
    ("Ours-S (DINOv3-S+PiSSA)", "SelaFD ViT-S"),
    ("Ours-S (DINOv3-S+PiSSA)", "ViT-S + LoRA"),
    ("Ours-B (DINOv3-B+PiSSA)", "SelaFD ViT-B"),
    ("Ours-B (DINOv3-B+PiSSA)", "Ours-S (DINOv3-S+PiSSA)"),
    ("Ours-B (DINOv3-B+PiSSA)", "ResNet50 (ft)"),
    ("Ours-B (DINOv3-B+PiSSA)", "DINOv3-S (full FT)"),
    ("DINOv3-S (full FT)",      "ResNet50 (ft)"),
]


def _t1_folds(label: str, dataset: str, metric: str) -> Optional[np.ndarray]:
    """Per-fold scores for one Table-1 row, or None if the run is missing."""
    field = "fold_accs" if metric == "acc" else "fold_f1s"
    if label in T1_FT:
        path = IN_DIST_DIR / f"{T1_FT[label]}_ep{EPOCHS}_lr{LR_TAG}_s42__{dataset}.json"
        probe = "supervised"
    else:
        path = IN_DIST_DIR / f"{T1_FROZEN[label]}__{dataset}.json"
        probe = "linear_C1.0"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    p = d.get("probes", {}).get(probe)
    return None if p is None or field not in p else np.asarray(p[field], dtype=float)


# ── Tables 2 & 3: per-fit few-shot scores ───────────────────────────────────

def _fits(path: Path, shot: int, metric: str) -> Optional[Dict[Tuple[int, int], float]]:
    """{(fold, rep) -> score} for one run at one shot level.

    Returns None if the file predates the raw-score persistence added to
    eval/cross_few_shot.py (i.e. only aggregate mean/std were kept).
    """
    d = json.loads(path.read_text())
    lp = d.get("probes", {}).get("linear_C1.0")
    if not isinstance(lp, list):
        return None
    for e in lp:
        if e["n_shots"] != shot:
            continue
        f = e.get("fits")
        if f is None:
            return None
        return {
            (int(fo), int(rp)): float(v)
            for fo, rp, v in zip(f["fold"], f["rep"], f[metric])
        }
    return None


def _cross_paths(kind: str, spec, src: str, tgt: str) -> List[Path]:
    """Result files for one Table-2/3 row and transfer pair (one per seed)."""
    if kind == "frozen":
        return sorted(CROSS_DIR.glob(f"{spec}_frozen_frozen_{src}__{tgt}.json"))
    if kind == "resnet":
        return sorted(CROSS_DIR.glob(f"resnet50_s*_ft_{src}__{tgt}.json"))
    if kind == "full_ft":
        return sorted(CROSS_DIR.glob(f"vits16_dinov3_full_ft_lr{LR_TAG}_s*__{src}__{tgt}.json"))
    if kind == "selafd":
        return sorted(CROSS_DIR.glob(f"vits16_vit_selafd_lr{LR_TAG}_s*__{src}__{tgt}.json"))
    # peft: file naming reorders method tokens for the *_in variants, so glob
    # permissively and filter on the sweep tag (same trick as the notebook).
    method, targets = spec
    out = []
    for fp in sorted(CROSS_DIR.glob(f"vits16_*_s*__{src}__{tgt}.json")):
        sw = json.loads(fp.read_text()).get("sweep", {})
        if (sw.get("method") == method and sw.get("epochs") == EPOCHS
                and sw.get("lr") == 1e-4 and sw.get("rank") == RANK
                and sw.get("targets") == targets):
            out.append(fp)
    return out


def _cross_draws(
    kind: str, spec, src: str, tgt: str, shot: int, metric: str,
) -> Tuple[Optional[Dict[Tuple[int, int], float]], str]:
    """{(fold, rep) -> score}, averaged over available fine-tuning seeds.

    Second element is a status string for diagnostics.
    """
    paths = _cross_paths(kind, spec, src, tgt)
    if not paths:
        return None, "no result files"
    acc: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    n_ok = 0
    for p in paths:
        fits = _fits(p, shot, metric)
        if fits is None:
            continue
        n_ok += 1
        for k, v in fits.items():
            acc[k].append(v)
    if n_ok == 0:
        return None, f"{len(paths)} file(s) but no raw per-fit scores (re-run needed)"
    return {k: float(np.mean(v)) for k, v in acc.items()}, f"{n_ok} seed(s)"


T23_ROWS = {
    "DINOv3-S (frozen)":     ("frozen",  "dinov3_vits16"),
    "ViT-S ImageNet (frozen)": ("frozen", "vit_small_imagenet"),
    "ResNet50 (full FT)":    ("resnet",  None),
    "DINOv3-S (full FT)":    ("full_ft", None),
    "SelaFD (ViT-S)":        ("selafd",  None),
    "ImageNet ViT-S + LoRA": ("peft",    ("lora_in", "attn")),
    "Ours (DINOv3-S+PiSSA)": ("peft",    ("pissa",   "all")),
    # Table 3 ablation cells
    "DINOv3-S + LoRA attn":  ("peft",    ("lora",     "attn")),
    "ImageNet ViT-S + PiSSA all": ("peft", ("pissa_in", "all")),
}

T2_COMPARISONS = [
    ("Ours (DINOv3-S+PiSSA)", "SelaFD (ViT-S)"),
    ("Ours (DINOv3-S+PiSSA)", "ImageNet ViT-S + LoRA"),
    ("Ours (DINOv3-S+PiSSA)", "ResNet50 (full FT)"),
    ("Ours (DINOv3-S+PiSSA)", "DINOv3-S (full FT)"),
    ("Ours (DINOv3-S+PiSSA)", "DINOv3-S (frozen)"),
    ("Ours (DINOv3-S+PiSSA)", "ViT-S ImageNet (frozen)"),
]

# Table 3 is a 2x2: {LoRA attn, PiSSA all} x {ImageNet ViT-S, DINOv3 ViT-S}.
# The interesting contrasts are the two adaptation effects (one per backbone)
# and the two backbone effects (one per adaptation).
T3_COMPARISONS = [
    ("Ours (DINOv3-S+PiSSA)",      "DINOv3-S + LoRA attn"),        # PiSSA effect, DINOv3
    ("ImageNet ViT-S + PiSSA all", "ImageNet ViT-S + LoRA"),        # PiSSA effect, ImageNet
    ("DINOv3-S + LoRA attn",       "ImageNet ViT-S + LoRA"),        # backbone effect, LoRA
    ("Ours (DINOv3-S+PiSSA)",      "ImageNet ViT-S + PiSSA all"),   # backbone effect, PiSSA
]
T3_PAIRS = ["Cross-orientation", "Cross-dataset"]


# ── Testing machinery ───────────────────────────────────────────────────────

def _paired(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Paired t-test + Wilcoxon on a-b, guarding degenerate inputs."""
    d = a - b
    out = {"delta": float(d.mean()), "n": int(len(d))}
    if len(d) < 2 or np.allclose(d, 0):
        out["t_p"] = out["w_p"] = float("nan")
        return out
    out["t_p"] = float(stats.ttest_rel(a, b).pvalue)
    try:
        out["w_p"] = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        out["w_p"] = float("nan")
    return out


def _holm(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaNs pass through)."""
    idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    adj = [float("nan")] * len(pvals)
    if not idx:
        return adj
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)      # enforce monotonicity
        adj[i] = running
    return adj


def _fold_means(draws: Dict[Tuple[int, int], float]) -> np.ndarray:
    """Collapse (fold, rep) scores to one mean per fold."""
    per_fold: Dict[int, List[float]] = defaultdict(list)
    for (fold, _rep), v in draws.items():
        per_fold[fold].append(v)
    return np.asarray([np.mean(per_fold[f]) for f in sorted(per_fold)], dtype=float)


def _report(title: str, rows: List[Dict], scale: float = 100.0) -> None:
    """Print one Holm family. `rows` carry raw stats; Holm is applied here."""
    print(f"\n### {title}")
    usable = [r for r in rows if r.get("skip") is None]
    for r in rows:
        if r.get("skip"):
            print(f"  {r['a']} vs {r['b']}: SKIPPED — {r['skip']}")
    if not usable:
        return
    holm_fold = _holm([r["fold"]["t_p"] for r in usable])
    holm_draw = _holm([r["draw"]["t_p"] for r in usable]) if usable[0]["draw"] else None

    hdr = (f"  {'comparison':<52}{'Δ':>7}{'W-L':>8}"
           f"{'fold t':>9}{'fold Holm':>11}")
    if holm_draw is not None:
        hdr += f"{'draw t':>10}{'draw Holm':>11}{'draw W-L':>10}"
    print(hdr)
    for r, hf in zip(usable, holm_fold if holm_draw is None else holm_fold):
        name = f"{r['a']} vs {r['b']}"
        line = (f"  {name:<52}{r['fold']['delta']*scale:>+7.2f}{r['fold_wl']:>8}"
                f"{r['fold']['t_p']:>9.4f}{hf:>11.4f}")
        if holm_draw is not None:
            hd = holm_draw[usable.index(r)]
            line += (f"{r['draw']['t_p']:>10.4f}{hd:>11.4f}{r['draw_wl']:>10}")
        star = ""
        if not np.isnan(hf) and hf < 0.05:
            star = "  *fold-Holm<.05"
        elif holm_draw is not None:
            hd = holm_draw[usable.index(r)]
            if not np.isnan(hd) and hd < 0.05:
                star = "  (draw-Holm<.05 only)"
        print(line + star)


def _wl(a: np.ndarray, b: np.ndarray) -> str:
    return f"{int((a > b).sum())}-{int((a < b).sum())}"


# ── Drivers ─────────────────────────────────────────────────────────────────

def run_table1(metric: str) -> None:
    print("\n" + "=" * 100)
    print("TABLE 1 — in-distribution, paired over the 5 shared subject-disjoint folds")
    print("=" * 100)
    print("n=5: the paired t-test is the only test with any power here; the two-sided")
    print("Wilcoxon floor at n=5 is p=0.0625, so it can never reach 0.05.")
    for dataset in ("glasgow", "mad"):
        rows = []
        for a_lab, b_lab in T1_COMPARISONS:
            a, b = _t1_folds(a_lab, dataset, metric), _t1_folds(b_lab, dataset, metric)
            if a is None or b is None:
                miss = a_lab if a is None else b_lab
                rows.append({"a": a_lab, "b": b_lab, "skip": f"missing run: {miss}"})
                continue
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            rows.append({
                "a": a_lab, "b": b_lab, "skip": None,
                "fold": _paired(a, b), "fold_wl": _wl(a, b),
                "draw": None, "draw_wl": None,
            })
        _report(f"{dataset.upper()} ({metric})", rows)


def run_cross(title: str, comparisons, pair_names, shot: int, metric: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for pair_label in pair_names:
        src, tgt = CROSS_PAIRS[pair_label]
        cache: Dict[str, Optional[Dict[Tuple[int, int], float]]] = {}
        status: Dict[str, str] = {}
        rows = []
        for a_lab, b_lab in comparisons:
            for lab in (a_lab, b_lab):
                if lab not in cache:
                    kind, spec = T23_ROWS[lab]
                    cache[lab], status[lab] = _cross_draws(
                        kind, spec, src, tgt, shot, metric,
                    )
            A, B = cache[a_lab], cache[b_lab]
            if A is None or B is None:
                miss = a_lab if A is None else b_lab
                rows.append({"a": a_lab, "b": b_lab,
                             "skip": f"{miss}: {status[miss]}"})
                continue
            keys = sorted(set(A) & set(B))
            if not keys:
                rows.append({"a": a_lab, "b": b_lab, "skip": "no shared (fold, rep) draws"})
                continue
            av = np.asarray([A[k] for k in keys])
            bv = np.asarray([B[k] for k in keys])
            afold = _fold_means({k: A[k] for k in keys})
            bfold = _fold_means({k: B[k] for k in keys})
            rows.append({
                "a": a_lab, "b": b_lab, "skip": None,
                "fold": _paired(afold, bfold), "fold_wl": _wl(afold, bfold),
                "draw": _paired(av, bv), "draw_wl": _wl(av, bv),
            })
        _report(f"{pair_label}  ({src} → {tgt}, {shot}-shot, {metric})", rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
    p.add_argument("--shot", type=int, default=10)
    p.add_argument("--metric", choices=["acc", "f1"], default="acc")
    args = p.parse_args()

    if "1" in args.tables:
        run_table1(args.metric)
    if "2" in args.tables:
        run_cross(
            "TABLE 2 — cross-distribution, paired over shared folds and shared "
            f"{args.shot}-shot draws",
            T2_COMPARISONS, list(CROSS_PAIRS), args.shot, args.metric,
        )
    if "3" in args.tables:
        run_cross(
            "TABLE 3 — ablation (LoRA/PiSSA x ImageNet/DINOv3), same shared draws",
            T3_COMPARISONS, T3_PAIRS, args.shot, args.metric,
        )
    print()


if __name__ == "__main__":
    main()
