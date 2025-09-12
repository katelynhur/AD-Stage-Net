#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combine multiple ensemble_results.csv (from different test datasets) into one table.

- Reads N inputs specified as:  --inputs "path1::Luke,path2::Marco,path3::Falah"
  (You may also use '|' instead of '::' if that’s what your wrapper has.)
- For each file, takes the BEST row per unique ensemble combo (by acc).
- Builds a canonical combo_key from sorted members_run_dirs (fallback to members_names).
- Produces per-dataset accuracy columns: acc_<LABEL>
- Adds aggregate columns: avg_acc, min_acc, std_acc
- Keeps metadata (members_names, members_families, members_run_dirs, n_models) from
  the first time a combo appears.

Output:
  <out_dir>/combined_ensemble_results.csv
  <out_dir>/combined_ensemble_results.md   (top 50 pretty table)
"""

import argparse
from pathlib import Path
import os
import math
import pandas as pd
import numpy as np


def parse_inputs(spec: str):
    """
    Accepts comma-separated items: 'path::Label' or 'path|Label'.
    If no label provided, infer from parent folder name or file stem.
    Returns list of (Path, label).
    """
    items = []
    if not spec.strip():
        return items
    for raw in [p.strip() for p in spec.split(",") if p.strip()]:
        if "::" in raw:
            pth, lab = raw.split("::", 1)
        elif "|" in raw:
            pth, lab = raw.split("|", 1)
        else:
            pth = raw
            lab = Path(pth).parent.name or Path(pth).stem
        items.append((Path(os.path.expanduser(pth)).resolve(), lab.strip()))
    return items


def make_combo_key(row: pd.Series) -> str:
    """
    Canonical key built from sorted members_run_dirs (preferred).
    Fallback to sorted members_names if needed.
    """
    if "members_run_dirs" in row and pd.notna(row["members_run_dirs"]):
        parts = [p.strip() for p in str(row["members_run_dirs"]).split("|")]
        parts = [p for p in parts if p]
        parts = sorted(parts)
        return "||".join(parts)
    # fallback
    names = [p.strip() for p in str(row.get("members_names", "")).split("+")]
    names = [p for p in names if p]
    names = sorted(names)
    return " + ".join(names)


def best_per_combo(df: pd.DataFrame) -> pd.DataFrame:
    """
    From an ensemble_results.csv, keep only the best row per combo_key by 'acc'.
    """
    if "acc" not in df.columns:
        raise SystemExit("Input CSV missing required column 'acc'.")

    df = df.copy()
    df["combo_key"] = df.apply(make_combo_key, axis=1)
    # sort by acc desc and drop dup combo_key
    df = df.sort_values("acc", ascending=False).drop_duplicates(subset=["combo_key"], keep="first")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, required=True,
                    help='Comma list like "path1::Luke,path2::Marco,path3::Falah" (also supports "|" instead of "::").')
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--top_preview", type=int, default=50, help="Rows to show in markdown preview")
    args = ap.parse_args()

    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = parse_inputs(args.inputs)
    if not inputs:
        raise SystemExit("No inputs provided to --inputs")

    # Accumulate rows keyed by combo_key
    store = {}  # combo_key -> dict of merged fields

    for csv_path, label in inputs:
        if not csv_path.exists():
            raise SystemExit(f"Missing input file: {csv_path}")
        df = pd.read_csv(csv_path)

        # Normalize and keep best per combo
        df_best = best_per_combo(df)

        # Walk rows and merge
        for _, r in df_best.iterrows():
            key = r["combo_key"]
            if key not in store:
                # capture metadata from the first dataset that has this combo
                store[key] = {
                    "combo_key": key,
                    "members_names": r.get("members_names"),
                    "members_families": r.get("members_families"),
                    "members_run_dirs": r.get("members_run_dirs"),
                    "n_models": int(r.get("n_models")) if pd.notna(r.get("n_models")) else None,
                }
            # per-dataset accuracy column
            store[key][f"acc_{label}"] = float(r["acc"]) if pd.notna(r["acc"]) else np.nan

    # Convert to DataFrame
    rows = list(store.values())
    combined = pd.DataFrame(rows)

    # Compute aggregates over all acc_* columns present
    acc_cols = [c for c in combined.columns if c.startswith("acc_")]
    if acc_cols:
        combined["avg_acc"] = combined[acc_cols].mean(axis=1, skipna=True)
        combined["min_acc"] = combined[acc_cols].min(axis=1, skipna=True)
        combined["std_acc"] = combined[acc_cols].std(axis=1, ddof=0, skipna=True)
    else:
        combined["avg_acc"] = np.nan
        combined["min_acc"] = np.nan
        combined["std_acc"] = np.nan

    # Sort by robust criteria
    combined = combined.sort_values(["min_acc", "avg_acc", "std_acc"], ascending=[False, False, True]).reset_index(drop=True)

    # Save
    out_csv = out_dir / "combined_ensemble_results.csv"
    combined.to_csv(out_csv, index=False)

    # Markdown preview
    out_md = out_dir / "combined_ensemble_results.md"
    head = combined.head(args.top_preview).copy()
    try:
        out_md.write_text(head.to_markdown(index=False))
    except Exception:
        out_md.write_text(head.to_string(index=False))

    print("Wrote:")
    print("-", out_csv)
    print("-", out_md)
    print(f"Rows: {len(combined)} | Datasets merged: {len(acc_cols)} [{', '.join(acc_cols)}]")


if __name__ == "__main__":
    main()

