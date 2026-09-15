#!/usr/bin/env python3
"""Produce ``data/Glasgow/`` from ``data/Glasgow_STFT_MAT/`` (the UoG release).

Place what you downloaded from the University of Glasgow in this directory,
next to this script, then run it from the repository root. The release ("Radar signatures of human activities",
DOI 10.5525/gla.researchdata.848) is split into seven acquisition campaigns,
each distributed as a folder or ``.zip`` archive:

    1 December 2017 Dataset        4 July 2018 Dataset
    2 March 2017 Dataset           5 February 2019 UoG Dataset
    3 June 2017 Dataset            6 February 2019 NG Homes Dataset
                                   7 March 2019 West Cumbria Dataset

Every recording ``KPXXAYYRZ.dat`` comes with a precomputed micro-Doppler
spectrogram ``KPXXAYYRZ.dat_MD_prm_TWL_256_OF_95_PF_4.mat`` holding the
complex matrix ``Data_spec_MTI2`` (1024 Doppler bins x time). This script
converts those spectrograms into the layout ``eval.datasets.glasgow``
expects:

    data/Glasgow/{K}/D{CC}_{code}.npy    # (1024, 365) float32 magnitude

where ``K`` is the activity (1 walking, 2 sitting down, 3 standing up,
4 picking up an object, 5 drinking, 6 falling) and ``CC`` the campaign.
The ``.dat`` raw radar files are not needed.

Processing, per spectrogram:

1. Magnitude: ``|Data_spec_MTI2|``, stored as float32.
2. Naming: run ids are zero-padded (``R1`` -> ``R01``) and the campaign is
   prefixed (``D03_``), since participant ids repeat across campaigns.
3. Uniform duration of 365 time bins (5 s):
   - 750-bin walking clips (10 s) -> first and last 365 bins, saved as
     runs ``R0z`` and ``R1z``;
   - 1519-bin recordings (three campaign-6 walking files) -> four evenly
     spaced 365-bin windows, runs ``R0z``..``R3z``;
   - 750-bin ``2P36A02R01`` (a sitting-down recording) -> first 365 bins;
   - 365-bin recordings -> unchanged.

Expected result: 2,081 spectrograms (640 / 312 / 311 / 310 / 310 / 198 per
activity) from 1,753 source files: the release's 1,754 minus one duplicate
(see ``_DUPLICATE_CODES``).

Idempotent: files that already exist in the destination are skipped, never
overwritten.

Usage
-----
    # from repo root, with the campaign folders/archives in data/Glasgow_STFT_MAT/
    python data/Glasgow_STFT_MAT/prepare_glasgow.py            # writes data/Glasgow/
    python data/Glasgow_STFT_MAT/prepare_glasgow.py --dry-run  # report only
    python data/Glasgow_STFT_MAT/prepare_glasgow.py --campaigns 6 7

    # Check a conversion against an existing tree, entirely in memory
    # (writes nothing): compares the serialized bytes of every file.
    python data/Glasgow_STFT_MAT/prepare_glasgow.py --verify data/Glasgow

    # the release can also be read from elsewhere
    python data/Glasgow_STFT_MAT/prepare_glasgow.py --src "/path/to/Dataset Glasgow"
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import scipy.io as sio

SRC_ROOT = Path(__file__).resolve().parent               # data/Glasgow_STFT_MAT/
DST_ROOT = SRC_ROOT.parent / "Glasgow"                   # data/Glasgow/

N_DOPPLER = 1024
TARGET_WIDTH = 365       # 5 s
WALK_WIDTH = 750         # 10 s walking clips
EXTRA_WIDE_WIDTH = 1519  # three campaign-6 walking recordings

_CAMPAIGN_RE = re.compile(r"^(\d+)\s")
# Some release filenames have stray spaces before "_MD_" (e.g. "R1.dat    _MD_").
_MD_NAME_RE = re.compile(r"^(.*?)\.dat\s*_MD_.*\.mat$")
_RUN_RE = re.compile(r"^(.*R)(\d+)$")

# Two release files are sample-for-sample duplicates of another recording:
#   - "4P64A04R3 - Copy" (July 2018) duplicates 4P64A04R3 -> skipped.
#   - "4P30A04R3 (2)" (June 2017) duplicates 4P30A04R3 -> kept, under its
#     original name, because it is one of the 2,081 spectrograms the paper's
#     results were computed on. Folds are grouped by subject, so both copies
#     always fall on the same side of every split.
_DUPLICATE_CODES = {"4P64A04R3 - Copy"}


@dataclass(frozen=True)
class Source:
    """One ``_MD_`` spectrogram inside a campaign folder or archive."""
    campaign: int
    name: str                        # basename, used to derive the code
    label: str                       # for error messages
    read: Callable[[], dict]         # returns the loadmat dict


# ── Discovery ────────────────────────────────────────────────────────────────

def _is_md_mat(name: str) -> bool:
    """True for an MD spectrogram that is not a skipped duplicate."""
    m = _MD_NAME_RE.match(name)
    return m is not None and m.group(1) not in _DUPLICATE_CODES


def discover(src: Path, campaigns: set[int] | None) -> list[Source]:
    """List every MD spectrogram, preferring extracted folders over archives."""
    folders: dict[int, Path] = {}
    archives: dict[int, Path] = {}
    for entry in sorted(src.iterdir()):
        m = _CAMPAIGN_RE.match(entry.name)
        if m is None:
            continue
        cid = int(m.group(1))
        if entry.is_dir():
            folders[cid] = entry
        elif entry.suffix.lower() == ".zip":
            archives[cid] = entry

    sources: list[Source] = []
    for cid in sorted(set(folders) | set(archives)):
        if campaigns is not None and cid not in campaigns:
            continue
        folder = folders.get(cid)
        mats = sorted(p for p in folder.rglob("*.mat") if _is_md_mat(p.name)) if folder else []
        if mats:
            for p in mats:
                sources.append(Source(cid, p.name, str(p),
                                      lambda p=p: sio.loadmat(p)))
        elif cid in archives:
            zpath = archives[cid]
            with zipfile.ZipFile(zpath) as zf:
                members = sorted(n for n in zf.namelist() if _is_md_mat(Path(n).name))
            for member in members:
                def _read(zpath=zpath, member=member) -> dict:
                    with zipfile.ZipFile(zpath) as zf:
                        return sio.loadmat(io.BytesIO(zf.read(member)))
                sources.append(Source(cid, Path(member).name, f"{zpath.name}:{member}", _read))
    return sources


# ── Conversion ───────────────────────────────────────────────────────────────

def _code(name: str) -> str:
    """``1P36A01R1.dat_MD_...mat`` -> ``1P36A01R01``."""
    code = _MD_NAME_RE.match(name).group(1)
    m = _RUN_RE.match(code)
    return f"{m.group(1)}{int(m.group(2)):02d}" if m else code


def _offset_run(code: str, offset: int) -> str:
    """``...R02`` with offset 10 -> ``...R12`` (names the extra windows)."""
    if offset == 0:
        return code
    m = _RUN_RE.match(code)
    if m is None:
        raise ValueError(f"cannot offset run id of {code}")
    prefix, run = m.groups()
    return f"{prefix}{int(run) + offset:0{max(2, len(run))}d}"


def _matrix(mat: dict, label: str) -> np.ndarray:
    keys = [k for k in mat if not k.startswith("__")]
    if len(keys) != 1:
        keys = [k for k in keys if np.asarray(mat[k]).ndim >= 2]
    if len(keys) != 1:
        raise ValueError(f"{label}: expected one matrix variable, found {keys}")
    return np.asarray(mat[keys[0]])


def windows(code: str, mag: np.ndarray, label: str) -> list[tuple[str, np.ndarray]]:
    """Cut one magnitude spectrogram into (output code, 365-bin window) pairs."""
    rows, cols = mag.shape
    if rows != N_DOPPLER:
        raise ValueError(f"{label}: expected {N_DOPPLER} Doppler bins, got {rows}")
    if cols == TARGET_WIDTH:
        return [(code, mag)]
    if cols == WALK_WIDTH and code.startswith("1P"):
        return [(code, mag[:, :TARGET_WIDTH]),
                (_offset_run(code, 10), mag[:, -TARGET_WIDTH:])]
    if cols == WALK_WIDTH and code == "2P36A02R01":
        return [(code, mag[:, :TARGET_WIDTH])]
    if cols == EXTRA_WIDE_WIDTH:
        starts = np.linspace(0, cols - TARGET_WIDTH, num=4, dtype=int)
        return [(_offset_run(code, 10 * i), mag[:, s:s + TARGET_WIDTH])
                for i, s in enumerate(starts)]
    raise ValueError(f"{label}: unsupported shape {mag.shape}")


def convert(src: Source) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (relative output path, array) for one source spectrogram."""
    code = _code(src.name)
    mag = np.abs(_matrix(src.read(), src.label)).astype(np.float32, copy=False)
    for out_code, arr in windows(code, mag, src.label):
        yield f"{code[0]}/D{src.campaign:02d}_{out_code}.npy", arr.astype(np.float32, copy=False)


def _serialize(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=SRC_ROOT,
                   help="Folder containing the campaign folders and/or .zip "
                        "archives (default: this script's folder, data/Glasgow_STFT_MAT/).")
    p.add_argument("--dst", type=Path, default=DST_ROOT,
                   help="Output root (default: data/Glasgow/).")
    p.add_argument("--campaigns", nargs="+", type=int, metavar="N",
                   help="Restrict to campaign ids 1-7.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be written and exit.")
    p.add_argument("--verify", type=Path, metavar="DIR",
                   help="Convert in memory and compare byte-for-byte against an "
                        "existing tree in DIR. Writes nothing.")
    args = p.parse_args()

    if not args.src.is_dir():
        sys.exit(f"error: source folder not found: {args.src}")
    sources = discover(args.src, set(args.campaigns) if args.campaigns else None)
    if not sources:
        sys.exit(f"error: no *_MD_*.mat spectrograms found under {args.src}")

    per_campaign = Counter(s.campaign for s in sources)
    print(f"[prepare] {len(sources)} source spectrograms under {args.src}")
    for cid, n in sorted(per_campaign.items()):
        print(f"          campaign {cid}: {n}")

    if args.verify is not None:
        _verify(sources, args.verify, full_tree=args.campaigns is None)
        return

    mode = "would write" if args.dry_run else "wrote"
    print(f"          destination: {args.dst}")
    written = skipped = 0
    per_class: Counter[str] = Counter()
    seen: set[str] = set()
    for i, src in enumerate(sources, 1):
        for rel, arr in convert(src):
            if rel in seen:
                raise RuntimeError(f"two sources map to the same output {rel}")
            seen.add(rel)
            per_class[rel.split("/")[0]] += 1
            out = args.dst / rel
            if out.exists():
                skipped += 1
                continue
            if not args.dry_run:
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, arr)
            written += 1
        if i % 250 == 0:
            print(f"  {i}/{len(sources)} sources processed")

    print()
    print(f"[summary] outputs: {len(seen)}  ({mode}: {written}, already present: {skipped})")
    print("          per activity: " + "  ".join(f"{k}:{per_class[k]}" for k in sorted(per_class)))


def _verify(sources: list[Source], ref: Path, full_tree: bool) -> None:
    """Compare every converted file's bytes to the matching file under ``ref``."""
    if not ref.is_dir():
        sys.exit(f"error: reference folder not found: {ref}")
    match, mismatch, missing = 0, [], []
    produced: set[str] = set()
    for i, src in enumerate(sources, 1):
        for rel, arr in convert(src):
            produced.add(rel)
            target = ref / rel
            if not target.exists():
                missing.append(rel)
            elif hashlib.sha256(_serialize(arr)).digest() == hashlib.sha256(target.read_bytes()).digest():
                match += 1
            else:
                mismatch.append(rel)
        if i % 250 == 0:
            print(f"  {i}/{len(sources)} sources verified")

    extra = sorted({p.relative_to(ref).as_posix() for p in ref.glob("*/*.npy")} - produced) \
        if full_tree else []
    print()
    print(f"[verify] against {ref}")
    print(f"         byte-identical: {match}")
    print(f"         different:      {len(mismatch)}  {mismatch[:5]}")
    print(f"         missing in ref: {len(missing)}  {missing[:5]}")
    if full_tree:
        print(f"         only in ref:    {len(extra)}  {extra[:5]}")
    if mismatch or missing or extra:
        sys.exit(1)


if __name__ == "__main__":
    main()
