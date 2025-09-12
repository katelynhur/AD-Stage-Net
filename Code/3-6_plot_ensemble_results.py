#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make presentation-ready plots and tables from combined_ensemble_results.csv.

Inputs  (from combine_ensemble_results.py):
  - combined_ensemble_results.csv  (must include columns like acc_Luke, acc_Marco, ...)

Outputs (written under --out_dir):
  - summary_top_robust.csv             # Top-N by robustness (min_acc desc, avg_acc desc, std_acc asc)
  - summary_top_avg.csv                # Top-N by avg_acc
  - summary_per_dataset_<NAME>.csv     # Top-N per dataset (e.g., acc_Luke)
  - summary_family_pairs.csv           # Family-pair aggregate (mean avg_acc, counts)
  - summary.md                         # All top tables in Markdown
  - fig_topN_min_acc.png               # Bar chart (Top-N by min_acc)
  - fig_avg_vs_std.png                 # Scatter of avg_acc vs std_acc
  - fig_topN_<NAME>.png                # Bar chart (Top-N for each acc_<NAME>)
  - fig_family_pair_heatmap.png        # Heatmap of mean avg_acc by family pairs

Usage:
  python plot_ensemble_results.py \
    --combined_csv ~/Alzheimers/Results/EnsembleEval/Combined_Pairs_From_Luke/combined_ensemble_results.csv \
    --out_dir       ~/Alzheimers/Results/EnsembleEval/Combined_Pairs_From_Luke/plots \
    --top_n 10
"""

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_markdown_tables(md_path: Path, blocks: list[tuple[str, pd.DataFrame]]):
    lines = []
    for title, df in blocks:
        lines.append(f"## {title}")
        try:
            lines.append(df.to_markdown(index=False))
        except Exception:
            lines.append(df.to_string(index=False))
        lines.append("")  # blank line
    md_path.write_text("\n".join(lines))


def bar_topn(df: pd.DataFrame, value_col: str, label_col: str, top_n: int, out_path: Path, title: str):
    sub = df[[label_col, value_col]].copy()
    sub = sub.head(top_n)
    # guard for empty
    if sub.empty:
        return
    # One chart per plot, default matplotlib styles/colors
    plt.figure()
    # safer labels (shorten)
    labels = sub[label_col].astype(str).str.replace("|", "│").str.slice(0, 80)
    y = sub[value_col].astype(float).values
    x = np.arange(len(sub))
    plt.bar(x, y)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def scatter_avg_std(df: pd.DataFrame, avg_col: str, std_col: str, out_path: Path, title: str):
    if df.empty:
        return
    x = df[avg_col].astype(float).values
    y = df[std_col].fillna(0.0).astype(float).values
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(avg_col)
    plt.ylabel(std_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_family_pair(s: str) -> tuple[str, str]:
    # "ResNet + DenseNet" -> ("DenseNet","ResNet") sorted for canonical key
    if not isinstance(s, str) or "+" not in s:
        # fallback
        return (str(s or "Unknown").strip(), "Unknown")
    parts = [p.strip() for p in s.split("+")]
    parts = [p for p in parts if p]
    if len(parts) == 1:
        parts = [parts[0], parts[0]]
    parts = tuple(sorted(parts))
    return parts[0], parts[1]


def build_family_pair_heatmap(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Returns: (matrix, family_labels)
      - matrix[i,j] = mean avg_acc for (fam_i, fam_j) combinations (using sorted pair key)
                      diagonal allowed but might be NaN if no same-family pairs exist.
    """
    if df.empty or "members_families" not in df.columns or "avg_acc" not in df.columns:
        return np.zeros((0, 0)), []

    # Collect stats per family pair
    stats = {}
    fams_set = set()
    for _, r in df.iterrows():
        f1, f2 = parse_family_pair(r["members_families"])
        fams_set.add(f1); fams_set.add(f2)
        key = (min(f1, f2), max(f1, f2))
        val = float(r["avg_acc"]) if not pd.isna(r["avg_acc"]) else None
        if val is None: 
            continue
        if key not in stats:
            stats[key] = {"sum": 0.0, "count": 0}
        stats[key]["sum"] += val
        stats[key]["count"] += 1

    fams = sorted(fams_set)
    if not fams:
        return np.zeros((0, 0)), []
    m = np.full((len(fams), len(fams)), np.nan, dtype=float)
    for (a, b), sc in stats.items():
        i = fams.index(a)
        j = fams.index(b)
        mean_val = sc["sum"] / max(1, sc["count"])
        m[i, j] = mean_val
        m[j, i] = mean_val
    return m, fams


def heatmap_family_pairs(df: pd.DataFrame, out_path: Path, title: str):
    M, fams = build_family_pair_heatmap(df)
    if M.size == 0:
        return
    plt.figure()
    im = plt.imshow(M, aspect="auto")
    plt.xticks(range(len(fams)), fams, rotation=45, ha="right")
    plt.yticks(range(len(fams)), fams)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    # annotate with values where not NaN
    for i in range(len(fams)):
        for j in range(len(fams)):
            v = M[i, j]
            if not (isinstance(v, float) and math.isnan(v)):
                plt.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    combined_csv = Path(args.combined_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(combined_csv)

    # Identify dataset-specific accuracy columns and key columns
    acc_cols = [c for c in df.columns if c.startswith("acc_")]
    required_cols = ["members_names", "members_families", "members_run_dirs"]
    for c in required_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing required column '{c}' in {combined_csv}")

    # Compute or validate aggregate columns
    if "avg_acc" not in df.columns:
        df["avg_acc"] = df[acc_cols].mean(axis=1, skipna=True) if acc_cols else np.nan
    if "min_acc" not in df.columns:
        df["min_acc"] = df[acc_cols].min(axis=1, skipna=True) if acc_cols else np.nan
    if "std_acc" not in df.columns:
        df["std_acc"] = df[acc_cols].std(axis=1, ddof=0, skipna=True) if acc_cols else 0.0

    # Sorts
    robust_sorted = df.sort_values(["min_acc", "avg_acc", "std_acc"], ascending=[False, False, True]).reset_index(drop=True)
    avg_sorted    = df.sort_values(["avg_acc", "min_acc", "std_acc"], ascending=[False, False, True]).reset_index(drop=True)

    # Save top tables
    topN = args.top_n
    robust_top = robust_sorted.head(topN).copy()
    avg_top = avg_sorted.head(topN).copy()

    robust_csv = out_dir / "summary_top_robust.csv"
    avg_csv    = out_dir / "summary_top_avg.csv"
    robust_top.to_csv(robust_csv, index=False)
    avg_top.to_csv(avg_csv, index=False)

    # Per-dataset top-N
    per_ds_blocks = []
    for col in acc_cols:
        per_ds = df.sort_values([col, "avg_acc"], ascending=[False, False]).head(topN).copy()
        per_ds.to_csv(out_dir / f"summary_per_dataset_{col.replace('acc_', '')}.csv", index=False)
        per_ds_blocks.append((f"Top {topN} by {col}", per_ds[["members_names","members_families","members_run_dirs", col, "avg_acc", "min_acc", "std_acc"]]))

    # Family-pair aggregate (mean avg_acc and counts)
    fam_pairs = []
    for _, r in df.iterrows():
        f1, f2 = parse_family_pair(r["members_families"])
        fam_pairs.append({"fam_a": f1, "fam_b": f2, "avg_acc": r["avg_acc"]})
    fam_df = pd.DataFrame(fam_pairs)
    fam_summary = fam_df.groupby(["fam_a","fam_b"], as_index=False).agg(mean_avg=("avg_acc","mean"),
                                                                        count=("avg_acc","count"))
    fam_summary.to_csv(out_dir / "summary_family_pairs.csv", index=False)

    # Markdown summary
    blocks = [
        (f"Top {topN} robust (by min_acc)", robust_top[["members_names","members_families","members_run_dirs","min_acc","avg_acc","std_acc"]]),
        (f"Top {topN} by avg_acc",         avg_top[   ["members_names","members_families","members_run_dirs","avg_acc","min_acc","std_acc"]]),
    ]
    blocks.extend(per_ds_blocks)
    save_markdown_tables(out_dir / "summary.md", blocks)

    # --------- Plots ----------
    # 1) Top-N by min_acc
    bar_topn(
        robust_sorted,
        value_col="min_acc",
        label_col="members_names",
        top_n=topN,
        out_path=out_dir / "fig_topN_min_acc.png",
        title=f"Top {topN} Ensembles by Min Accuracy (Robustness)"
    )

    # 2) Avg vs Std scatter (all)
    scatter_avg_std(
        df,
        avg_col="avg_acc",
        std_col="std_acc",
        out_path=out_dir / "fig_avg_vs_std.png",
        title="Ensembles: Average Accuracy vs Std Dev Across Datasets"
    )

    # 3) Per-dataset bar charts
    for col in acc_cols:
        per_ds_sorted = df.sort_values([col, "avg_acc"], ascending=[False, False]).reset_index(drop=True)
        bar_topn(
            per_ds_sorted,
            value_col=col,
            label_col="members_names",
            top_n=topN,
            out_path=out_dir / f"fig_topN_{col.replace('acc_','')}.png",
            title=f"Top {topN} Ensembles by {col}"
        )

    # 4) Family-pair heatmap (mean avg_acc)
    heatmap_family_pairs(
        df,
        out_path=out_dir / "fig_family_pair_heatmap.png",
        title="Mean Avg Accuracy by Family Pair"
    )

    print("Wrote tables & figures to:", out_dir)


if __name__ == "__main__":
    main()

