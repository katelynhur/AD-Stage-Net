#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Delete all param-keyed checkpoint files except the top-K by (best_val_acc, test_acc).

Usage:
  python cleanup_keep_topk.py \
    --results_dir ~/Alzheimers/Results/HP_Search_Luke_DDP \
    --csv resnet50_luke_small_search_ddp.csv \
    --topk 3

Optional:
  --file_globs "*.pt,*.pth"          Comma-separated patterns to target
  --exclude "final_best*.pt,final_best*.pth"  Patterns never to delete
  --recursive                         Search subdirectories too
  --dry_run                           Show what would be deleted without deleting
  --verbose                           Print extra details
"""

import argparse
from pathlib import Path
import pandas as pd

def parse_patterns(s: str):
    return [p.strip() for p in s.split(",") if p.strip()]

def collect_candidates(root: Path, patterns, recursive=False):
    files = set()
    if recursive:
        for pat in patterns:
            files.update(root.rglob(pat))
    else:
        for pat in patterns:
            files.update(root.glob(pat))
    return {p for p in files if p.is_file()}

def matches_any(path: Path, patterns):
    return any(path.match(pat) for pat in patterns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True, help="Directory containing CSV and checkpoints")
    ap.add_argument("--csv", type=str, required=True, help="CSV filename with metrics and ckpt_path")
    ap.add_argument("--topk", type=int, default=3, help="How many checkpoints to keep")
    ap.add_argument("--file_globs", type=str, default="*.pt,*.pth", help="Comma-separated file patterns to consider")
    ap.add_argument("--exclude", type=str, default="final_best*.pt,final_best*.pth", help="Comma-separated patterns to never delete")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--dry_run", action="store_true", help="Preview deletions only")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    args = ap.parse_args()

    out_dir = Path(args.results_dir).expanduser()
    csv_path = out_dir / args.csv
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV has no rows; nothing to do.")
        return

    required_cols = ["best_val_acc", "test_acc", "ckpt_path"]
    for col in required_cols:
        if col not in df.columns:
            print(f"CSV missing required column: {col}")
            return

    # Ensure numeric sort
    for col in ["best_val_acc", "test_acc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_sorted = df.sort_values(["best_val_acc", "test_acc"], ascending=False).reset_index(drop=True)

    keep_names = {Path(r["ckpt_path"]).name for _, r in df_sorted.head(args.topk).iterrows()}

    if args.verbose:
        print("Top-K to keep (by filename):")
        for n in keep_names:
            print("  -", n)

    file_globs = parse_patterns(args.file_globs)
    exclude_patterns = parse_patterns(args.exclude)

    candidates = collect_candidates(out_dir, file_globs, recursive=args.recursive)

    # Apply excludes
    candidates = {p for p in candidates if not matches_any(p, exclude_patterns)}

    # Delete everything not in keep set (comparison by basename)
    to_delete = [p for p in candidates if p.name not in keep_names]

    if args.dry_run:
        print(f"[DRY RUN] Would delete {len(to_delete)} files:")
        for p in sorted(to_delete):
            print("  ", p)
        print(f"[DRY RUN] Would keep {len(candidates) - len(to_delete)} matching files.")
        return

    cnt_del = 0
    for p in to_delete:
        try:
            p.unlink()
            cnt_del += 1
            if args.verbose:
                print("Deleted:", p)
        except Exception as e:
            print(f"Failed to delete {p}: {e}")

    print(f"Kept top-{args.topk}; deleted {cnt_del} other checkpoints.")

if __name__ == "__main__":
    main()
