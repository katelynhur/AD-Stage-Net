#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combine multiple Singles leaderboards (each from 3-2_summarize_models_v2.py)
into one comparison table.

Inputs: one or more "LABEL=path/to/leaderboard.csv"

Outputs under <proj_root>/<out_dir>/:
- combined_singles.csv       # wide table: test_acc__LABEL columns
- combined_singles.md        # markdown preview
- combined_singles_long.csv  # tidy format: (arch, cfg_hash, dataset, test_acc)
- Figures/combined_top_models.png  # grouped bars of top-N by mean across datasets
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def short_label(s, max_len=28):
    s = str(s)
    return s if len(s) <= max_len else s[:max_len-1] + "…"

def read_lb(p):
    df = pd.read_csv(p)
    # normalize types we care about
    for c in ["test_acc","best_val_acc","img_size","batch_size","global_batch_size"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["arch","cfg_hash","name","family","source","run_dir"]:
        if c not in df.columns:
            df[c] = ""
    return df

def dataframe_to_markdown(df: pd.DataFrame, max_rows=30) -> str:
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, required=True)
    ap.add_argument(
        "--inputs",
        type=str,
        required=True,
        help='Comma-separated: "Luke=Results/Model_Leaderboard_Singles_Luke/leaderboard.csv,'
             'Marco=Results/Model_Leaderboard_Singles_Marco/leaderboard.csv,'
             'Falah=Results/Model_Leaderboard_Singles_Falah/leaderboard.csv"'
    )
    ap.add_argument("--out_dir", type=str, default="Results/Combined_Singles_Leaderboard")
    ap.add_argument("--join_on", type=str, default="arch", choices=["arch", "arch_cfg"],
                    help='Join key: "arch" (default) or "arch_cfg" (use both arch and cfg_hash).')
    ap.add_argument("--top_n", type=int, default=20, help="For the grouped bar chart.")
    args = ap.parse_args()

    proj = Path(os.path.expanduser(args.proj_root)).resolve()
    out_dir = (proj / args.out_dir).resolve()
    fig_dir = out_dir / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Parse inputs
    items = [s.strip() for s in args.inputs.split(",") if s.strip()]
    pairs = []
    for it in items:
        if "=" not in it:
            raise ValueError(f'Bad --inputs item (missing "="): {it}')
        label, rel = it.split("=", 1)
        label = label.strip()
        p = (proj / rel.strip()).resolve()
        if not p.exists():
            raise FileNotFoundError(f"{label}: file not found: {p}")
        pairs.append((label, p))

    # Load and prepare each leaderboard
    key_cols = ["arch"] if args.join_on == "arch" else ["arch", "cfg_hash"]
    wide = None
    meta_cols = ["arch", "cfg_hash", "family", "name"]  # carry from the first table, when available
    all_longs = []

    for idx, (label, path) in enumerate(pairs):
        df = read_lb(path)

        # If multiple rows per arch (unlikely for single_best_only), keep the top test_acc per key
        df = df.sort_values("test_acc", ascending=False)
        df = df.drop_duplicates(subset=key_cols, keep="first")

        # Make a slim view for merging
        slim_cols = list(dict.fromkeys(key_cols + ["test_acc", "name", "family"]))
        slim = df[slim_cols].copy()
        slim = slim.rename(columns={"test_acc": f"test_acc__{label}"})

        # Long/tidy for plotting across datasets later
        long = df[key_cols + ["test_acc"]].copy()
        long["dataset"] = label
        all_longs.append(long)

        if wide is None:
            # Start with the first table and keep some meta columns
            base_cols = list(dict.fromkeys(key_cols + meta_cols))
            base = df.copy()
            for c in base_cols:
                if c not in base.columns:
                    base[c] = ""
            wide = base[base_cols].drop_duplicates(subset=key_cols, keep="first").copy()
            wide = wide.merge(slim[key_cols + [f"test_acc__{label}"]], on=key_cols, how="left")
        else:
            wide = wide.merge(slim[key_cols + [f"test_acc__{label}"]], on=key_cols, how="outer")

    # Compute mean across datasets (for sorting/plotting)
    dataset_cols = [c for c in wide.columns if c.startswith("test_acc__")]
    wide["mean_test_acc"] = wide[dataset_cols].mean(axis=1, skipna=True)

    # Sort by mean desc
    wide = wide.sort_values("mean_test_acc", ascending=False).reset_index(drop=True)

    # If name is missing, synthesize it from arch (+ short cfg)
    if "name" in wide.columns:
        needs_name = wide["name"].isna() | (wide["name"].astype(str).str.len() == 0)
        if "cfg_hash" in wide.columns:
            wide.loc[needs_name, "name"] = wide.loc[needs_name].apply(
                lambda r: f"{r.get('arch','?')} ({str(r.get('cfg_hash',''))[:6]})", axis=1
            )
        else:
            wide.loc[needs_name, "name"] = wide.loc[needs_name, "arch"]

    # Save wide + md
    (out_dir / "combined_singles.csv").write_text(wide.to_csv(index=False))
    (out_dir / "combined_singles.md").write_text(dataframe_to_markdown(wide, max_rows=60))

    # Save long/tidy
    long_df = pd.concat(all_longs, ignore_index=True)
    long_df.to_csv(out_dir / "combined_singles_long.csv", index=False)

    # -------- Plot: grouped bars for top-N by mean_test_acc --------
    top = wide.head(args.top_n).copy()
    if not top.empty and len(dataset_cols) >= 2:
        x = np.arange(len(top))
        width = max(0.8 / len(dataset_cols), 0.15)

        plt.figure(figsize=(max(10, 1.2 * len(top)), 6))
        for i, col in enumerate(dataset_cols):
            vals = top[col].values
            plt.bar(x + i * width, vals, width=width, label=col.replace("test_acc__", ""))

            # annotate bars
            for j, v in enumerate(vals):
                if pd.notna(v):
                    plt.text(x[j] + i * width, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        xt = top["name"] if "name" in top.columns else top["arch"]
        plt.xticks(x + (len(dataset_cols) - 1) * width / 2, [short_label(t) for t in xt], rotation=45, ha="right")
        plt.ylabel("Test Accuracy")
        plt.title(f"Top {args.top_n} Models — cross-dataset comparison")
        plt.legend(title="Dataset")
        plt.tight_layout()
        plt.savefig(fig_dir / "combined_top_models.png", dpi=200)
        plt.close()

    print("Wrote:")
    print(f"- {out_dir / 'combined_singles.csv'}")
    print(f"- {out_dir / 'combined_singles.md'}")
    print(f"- {out_dir / 'combined_singles_long.csv'}")
    print(f"- {fig_dir / 'combined_top_models.png'}")

if __name__ == "__main__":
    main()

