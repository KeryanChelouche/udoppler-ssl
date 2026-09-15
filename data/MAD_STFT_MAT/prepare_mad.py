#!/usr/bin/env python3
"""Produce ``data/MAD/`` (npy) from ``data/MAD_STFT_MAT/`` (.mat).

The MAD dataset ships as MATLAB ``.mat`` files under this directory, one
per (activity, subcategory, participant, repetition, antenna). Every entry
contains the complex STFT ``spec_complex`` at (256, 750) complex128 plus
metadata (radar_params, time_axis, doppler_axis, vel_axis) that the
downstream evaluation pipeline does not need.

This script converts the .mat files into the compact layout the
``eval.datasets.mad.MADDataset`` reader expects:

    data/MAD/{PPP}/{AYY1PPPXXXRZS5DV_complex.npy}   # complex64, (256, 750)

Only ``D3`` (Rx1) and ``D4`` (Rx2) files are converted; ``D0`` .bin
files are the raw pre-STFT data, which the pipeline does not consume.

Idempotent: skips any target file that already exists (safe to re-run).

Usage
-----
    # from repo root
    python data/MAD_STFT_MAT/prepare_mad.py            # full extract
    python data/MAD_STFT_MAT/prepare_mad.py --dry-run  # report only
    python data/MAD_STFT_MAT/prepare_mad.py --verify   # cross-check one file

    # optional: restrict to a subset of participants
    python data/MAD_STFT_MAT/prepare_mad.py --participants 073 074 075
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.io as sio

SRC_ROOT = Path(__file__).resolve().parent               # data/MAD_STFT_MAT/
DST_ROOT = SRC_ROOT.parent / "MAD"                       # data/MAD/


def convert_one(src_mat: Path, dst_npy: Path) -> None:
    """Load spec_complex from .mat, cast to complex64, save as .npy."""
    dst_npy.parent.mkdir(parents=True, exist_ok=True)
    data = sio.loadmat(src_mat, variable_names=["spec_complex"])
    spec = data["spec_complex"].astype(np.complex64)
    np.save(dst_npy, spec)


def verify_sample_roundtrip() -> None:
    """Convert one .mat in memory, compare to its existing .npy sibling.

    Picks the first participant with both a .mat and a matching .npy on
    disk; useful to check that a re-extraction would be bit-identical to
    what's already been used in downstream results.
    """
    for pdir in sorted(SRC_ROOT.iterdir()):
        if not pdir.is_dir() or not pdir.name.isdigit():
            continue
        mats = sorted(pdir.glob("*D[34]_complex.mat"))
        for mat_path in mats:
            npy_path = DST_ROOT / pdir.name / (mat_path.stem + ".npy")
            if not npy_path.exists():
                continue
            from_mat = sio.loadmat(mat_path)["spec_complex"].astype(np.complex64)
            from_npy = np.load(npy_path)
            diff = float(np.abs(from_mat - from_npy).max())
            print(f"[verify] {mat_path.relative_to(SRC_ROOT.parent)}  "
                  f"shape={from_mat.shape}  max abs diff = {diff:.2e}")
            if diff > 1e-3:
                print(f"  ⚠ non-trivial difference — check dtype conventions")
            return
    print("[verify] no participant found with both .mat and existing .npy")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be converted and exit.")
    p.add_argument("--verify", action="store_true",
                   help="Verify one .mat/.npy round-trip against an existing pair, then exit.")
    p.add_argument("--participants", nargs="+", metavar="PID",
                   help="Restrict to specific zero-padded participant IDs (e.g. 073 074).")
    args = p.parse_args()

    if args.verify:
        verify_sample_roundtrip()
        return

    if not SRC_ROOT.is_dir():
        print(f"error: source directory not found: {SRC_ROOT}", file=sys.stderr)
        sys.exit(1)

    all_participants = sorted(d for d in SRC_ROOT.iterdir()
                              if d.is_dir() and d.name.isdigit())
    if args.participants:
        keep = set(args.participants)
        participants = [d for d in all_participants if d.name in keep]
        missing = keep - {d.name for d in participants}
        if missing:
            print(f"warning: participant(s) not found in {SRC_ROOT}: {sorted(missing)}",
                  file=sys.stderr)
    else:
        participants = all_participants

    action_prefix = "would convert" if args.dry_run else "converted"
    print(f"[prepare] {len(participants)} participants under {SRC_ROOT}")
    print(f"          destination: {DST_ROOT}")

    total_files = converted = skipped = 0
    for pdir in participants:
        pid = pdir.name
        dst_dir = DST_ROOT / pid
        mats = sorted(pdir.glob("*D[34]_complex.mat"))
        per_pid_converted = 0
        for mat_path in mats:
            total_files += 1
            npy_path = dst_dir / (mat_path.stem + ".npy")
            if npy_path.exists():
                skipped += 1
                continue
            if not args.dry_run:
                convert_one(mat_path, npy_path)
            converted += 1
            per_pid_converted += 1
        print(f"  {pid}: {action_prefix} {per_pid_converted:>3d}/{len(mats):>3d} .mat files")

    print()
    print(f"[summary] total .mat files scanned: {total_files}")
    print(f"          {action_prefix}: {converted}")
    print(f"          already-existing (skipped): {skipped}")


if __name__ == "__main__":
    main()
