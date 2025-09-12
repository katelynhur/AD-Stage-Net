#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export best single-model checkpoints (best.pt only) per architecture.

- Reads <proj_root>/<sweep_dir>/arch_sweep_results.csv
- Picks the top row per arch by test_acc (fallback: best_val_acc)
- Copies <run_dir>/best.pt -> <out_dir>/<ARCH>_best.pt  (optionally add cfg_hash)
- Writes a small manifest CSV/JSON for traceability.

Usage:
  python 3-6_export_best_ckpts.py \
    --proj_root ~/Alzheimers \
    --sweep_dir Results/ArchSweep \
    --out_dir Results/BestSingles \
    --include_hash 0 \
    --overwrite 1
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

def pick_best_row(df_arch: pd.DataFrame) -> pd.Series:
    """Pick best by test_acc (desc), then best_val_acc (desc)."""
    df = df_arch.copy()
    for col in ("test_acc", "best_val_acc"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df["_rank_key"] = list(zip(
        df["test_acc"].fillna(-np.inf),
        df["best_val_acc"].fillna(-np.inf),
    ))
    idx = df["_rank_key"].idxmax()
    return df.loc[idx].drop(labels=["_rank_key"])

def safe_arch_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, required=True)
    ap.add_argument("--sweep_dir", type=str, default="Results/ArchSweep")
    ap.add_argument("--out_dir", type=str, default="Results/BestSingles")
    ap.add_argument("--include_hash", type=int, default=0, help="Append _<cfg_hash> to filename")
    ap.add_argument("--overwrite", type=int, default=0, help="Overwrite existing exported files")
    args = ap.parse_args()

    proj_root = Path(args.proj_root).expanduser().resolve()
    sweep_dir = (proj_root / args.sweep_dir).resolve()
    out_dir = (proj_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = sweep_dir / "arch_sweep_results.csv"
    if not csv_path.exists():
        print(f"[ERR] Missing CSV: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    needed = {"arch", "run_dir", "cfg_hash"}
    if not needed.issubset(df.columns):
        print(f"[ERR] CSV missing required columns: {needed - set(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Normalize run_dir to absolute paths
    def norm_run_dir(p):
        if pd.isna(p): return None
        pth = Path(str(p))
        return pth if pth.is_absolute() else (proj_root / pth).resolve()
    df["run_dir"] = df["run_dir"].apply(norm_run_dir)

    # Select best per arch
    best_rows = []
    for arch, g in df.groupby("arch", sort=False):
        try:
            row = pick_best_row(g)
        except Exception:
            row = g.iloc[0]
        best_rows.append(row)
    sel = pd.DataFrame(best_rows).reset_index(drop=True)

    # Export
    manifest_rows = []
    exported = 0
    skipped = 0
    for _, r in sel.iterrows():
        arch = str(r["arch"])
        run_dir: Path = r["run_dir"]
        cfg_hash = str(r.get("cfg_hash") or "")
        if run_dir is None or not run_dir.exists():
            print(f"[WARN] Missing run_dir for {arch}: {r.get('run_dir')}")
            skipped += 1
            continue
        src = run_dir / "best.pt"
        if not src.exists():
            print(f"[WARN] best.pt not found for {arch}: {src}")
            skipped += 1
            continue

        base = f"{safe_arch_name(arch)}_best"
        if args.include_hash and cfg_hash:
            base = f"{base}_{cfg_hash[:10]}"
        dst = out_dir / f"{base}.pt"

        if dst.exists() and not args.overwrite:
            print(f"[SKIP] Exists: {dst}")
        else:
            if dst.exists():
                dst.unlink()
            shutil.copy2(src, dst)
            print(f"[COPIED] {arch}: {src} -> {dst}")
            exported += 1

        manifest_rows.append({
            "arch": arch,
            "cfg_hash": cfg_hash,
            "source_run_dir": str(run_dir),
            "src_best": str(src),
            "dst_file": str(dst),
            "test_acc": float(r["test_acc"]) if "test_acc" in r and pd.notna(r["test_acc"]) else None,
            "best_val_acc": float(r["best_val_acc"]) if "best_val_acc" in r and pd.notna(r["best_val_acc"]) else None,
            "img_size": int(r["img_size"]) if "img_size" in r and pd.notna(r["img_size"]) else None,
        })

    # Save manifest
    manifest_df = pd.DataFrame(manifest_rows).sort_values("arch").reset_index(drop=True)
    (out_dir / "export_manifest.csv").write_text(manifest_df.to_csv(index=False))
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest_rows, indent=2))
    (out_dir / "README.txt").write_text(
        f"BestSingles Checkpoints\n"
        f"=======================\n"
        f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Source CSV: {csv_path}\n"
        f"Files: one <ARCH>_best(.<hash>).pt per architecture.\n"
    )

    print(f"\nDone. Exported: {exported} | Skipped: {skipped}")
    print(f"- {out_dir/'export_manifest.csv'}")
    print(f"- {out_dir/'export_manifest.json'}")

if __name__ == "__main__":
    main()

