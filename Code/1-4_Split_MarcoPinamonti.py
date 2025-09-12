#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patient-level train/test split for the Marco dataset (ImageFolder-style).

- Assumes data root has class subdirs like:
    Kaggle_MarcoPinamonti_Alzheimers_MRI/
      Mild_Impaired/
      Moderate_Impaired/
      No_Impairment/
      Very_Mild_Impaired/

- Filenames begin with a patient ID (leading integer), e.g.:
    "11 (6).jpg", "30.jpg", "3 (10).jpg"

- Splits **per class** at the **patient level** (no leakage), 80/20 by default.

Usage:
  python split_marco_patientwise.py \
    --root ~/Alzheimers/Data/Kaggle_MarcoPinamonti_Alzheimers_MRI \
    --test_ratio 0.2 \
    --seed 42 \
    --mode copy \
    --force 0
"""

import argparse
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def extract_patient_id(filename: str) -> str:
    """
    Extract leading integer as patient ID.
    Examples:
      '11 (6).jpg' -> '11'
      '30.jpg'     -> '30'
    """
    name = filename.strip()
    m = re.match(r"^(\d+)\b", name)
    if not m:
        raise ValueError(f"Cannot parse patient id from filename: {filename!r}")
    return m.group(1)

def find_class_dirs(root: Path) -> List[Path]:
    """Return immediate subdirs that look like class folders (non-empty)."""
    return [p for p in root.iterdir() if p.is_dir() and p.name.lower() not in {"train","test"}]

def collect_by_patient(class_dir: Path) -> Dict[str, List[Path]]:
    """
    For a class dir, return {patient_id: [list of image paths]}.
    """
    buckets: Dict[str, List[Path]] = {}
    for p in sorted(class_dir.glob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            pid = extract_patient_id(p.name)
            buckets.setdefault(pid, []).append(p)
    return buckets

def split_patients(pids: List[str], test_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    pids_shuffled = pids[:]
    rng.shuffle(pids_shuffled)
    n_total = len(pids_shuffled)
    n_test = max(1, int(round(n_total * test_ratio))) if n_total > 1 else 1
    test_ids = set(pids_shuffled[:n_test])
    train_ids = [pid for pid in pids_shuffled if pid not in test_ids]
    return train_ids, list(test_ids)

def safe_prepare_dir(d: Path, force: bool):
    if d.exists():
        # If directory exists and not empty, guard unless force
        if any(d.iterdir()):
            if not force:
                raise SystemExit(f"Refusing to write into non-empty dir: {d} (use --force 1 to clear)")
            shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

def transfer(paths: List[Path], dst_dir: Path, mode: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in paths:
        dst = dst_dir / src.name
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "move":
            shutil.move(str(src), str(dst))
        elif mode == "symlink":
            # relative symlink for portability
            rel = Path(shutil.os.path.relpath(src, dst_dir))
            if dst.exists():
                dst.unlink()
            dst.symlink_to(rel)
        else:
            raise ValueError(f"Unknown mode: {mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Marco dataset root (with class subfolders)")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of patients per class for test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", type=str, default="copy", choices=["copy","move","symlink"],
                    help="How to write files into train/test")
    ap.add_argument("--force", type=int, default=0, help="If 1, clear existing train/test before writing")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    class_dirs = find_class_dirs(root)
    if not class_dirs:
        raise SystemExit(f"No class subdirectories found under {root}")

    train_root = root / "train"
    test_root  = root / "test"

    # Prepare output dirs (guarded by --force)
    safe_prepare_dir(train_root, force=bool(args.force))
    safe_prepare_dir(test_root,  force=bool(args.force))

    random.seed(args.seed)

    summary = []
    total_train_imgs = total_test_imgs = 0

    for cdir in sorted(class_dirs, key=lambda p: p.name):
        buckets = collect_by_patient(cdir)
        if not buckets:
            print(f"[WARN] No images found in {cdir.name}; skipping.")
            continue

        pids = sorted(buckets.keys(), key=lambda x: int(x))
        train_ids, test_ids = split_patients(pids, test_ratio=args.test_ratio, seed=args.seed)

        # Flatten files for each split
        train_files = [p for pid in train_ids for p in buckets[pid]]
        test_files  = [p for pid in test_ids  for p in buckets[pid]]

        # Transfer
        transfer(train_files, train_root / cdir.name, mode=args.mode)
        transfer(test_files,  test_root  / cdir.name, mode=args.mode)

        total_train_imgs += len(train_files)
        total_test_imgs  += len(test_files)

        summary.append({
            "class": cdir.name,
            "n_patients": len(pids),
            "train_patients": len(train_ids),
            "test_patients": len(test_ids),
            "train_imgs": len(train_files),
            "test_imgs": len(test_files),
        })

        print(f"[{cdir.name}] patients: {len(pids)}  ->  train: {len(train_ids)} ({len(train_files)} imgs), "
              f"test: {len(test_ids)} ({len(test_files)} imgs)")

    print("\n=== Split complete ===")
    print(f"Train images: {total_train_imgs}")
    print(f"Test  images: {total_test_imgs}")
    print(f"Train dir: {train_root}")
    print(f"Test  dir: {test_root}")

    # Optional: write CSV summary
    try:
        import pandas as pd
        import json
        df = None
        if summary:
            df = pd.DataFrame(summary)
            df.to_csv(root / "split_summary.csv", index=False)
            (root / "split_summary.json").write_text(json.dumps(summary, indent=2))
            print(f"Summary saved to: {root/'split_summary.csv'}")
    except Exception:
        pass

if __name__ == "__main__":
    main()

