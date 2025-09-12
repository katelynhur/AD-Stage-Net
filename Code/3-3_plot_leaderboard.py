#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
  python plot_leaderboard.py --proj_root ~/Alzheimers \
    --leaderboard Results/Model_Leaderboard/leaderboard.csv \
    --out Results/Model_Leaderboard/Figures \
    --top_n 20
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, required=True)
    ap.add_argument("--leaderboard", type=str, default="Results/Model_Leaderboard/leaderboard.csv")
    ap.add_argument("--out", type=str, default="Results/Model_Leaderboard/Figures")
    ap.add_argument("--top_n", type=int, default=20)
    args = ap.parse_args()

    proj = Path(os.path.expanduser(args.proj_root))
    src_csv = proj / args.leaderboard
    out_dir = proj / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_csv)

    # numeric cleanup
    numeric_cols = [
        "img_size","batch_size","global_batch_size",
        "lr","dropout","weight_decay","label_smoothing",
        "best_val_acc","best_epoch","test_acc","precision","recall","f1",
        "epochs","val_split","num_workers"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["name","source","family","arch","run_dir","cfg_hash"]:
        if c not in df.columns:
            df[c] = ""

    df_acc = df.dropna(subset=["test_acc"]).copy()

    # ---------- Figure 1: Top-N bar chart ----------
    top_df = df_acc.sort_values("test_acc", ascending=False).head(args.top_n).copy()
    top_df["xlab"] = top_df.apply(lambda r: short_label(f"{r.get('name', r.get('arch','?'))} [{r.get('source','?')}]"), axis=1)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_df)), top_df["test_acc"])
    plt.xticks(range(len(top_df)), top_df["xlab"], rotation=45, ha="right")
    plt.ylabel("Test Accuracy")
    plt.title(f"Top {args.top_n} Models by Test Accuracy")
    for i, v in enumerate(top_df["test_acc"]):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "top_models_test_acc.png", dpi=200)
    plt.close()

    # ---------- Figure 2: Test vs Best-Val scatter ----------
    singles = df_acc[df_acc.get("source","") == "single_arch"].copy()
    hybrids = df_acc[df_acc.get("source","") != "single_arch"].copy()

    plt.figure(figsize=(7, 6))
    if not singles.empty:
        plt.scatter(singles["best_val_acc"], singles["test_acc"], label="Single", marker="o", alpha=0.8)
    if not hybrids.empty:
        plt.scatter(hybrids["best_val_acc"], hybrids["test_acc"], label="Hybrid", marker="^", alpha=0.8)
    mn = np.nanmin([df_acc["best_val_acc"].min(), df_acc["test_acc"].min()])
    mx = np.nanmax([df_acc["best_val_acc"].max(), df_acc["test_acc"].max()])
    if not np.isnan(mn) and not np.isnan(mx):
        plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Best Val Accuracy")
    plt.ylabel("Test Accuracy")
    plt.title("Generalization: Test vs. Best-Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "test_vs_val_scatter.png", dpi=200)
    plt.close()

    # ---------- Figure 3: Family boxplot ----------
    fam_groups, fam_labels = [], []
    if "family" in df_acc.columns:
        fam_stats = df_acc.groupby("family")["test_acc"].median().sort_values(ascending=False)
        for fam in fam_stats.index.tolist():
            vals = df_acc[df_acc["family"] == fam]["test_acc"].dropna().values
            if len(vals) > 0:
                fam_groups.append(vals)
                fam_labels.append(fam)
    if fam_groups:
        plt.figure(figsize=(10, 6))
        try:
            # Matplotlib >= 3.9
            plt.boxplot(fam_groups, tick_labels=fam_labels, showmeans=True)
        except TypeError:
            # Matplotlib < 3.9
            plt.boxplot(fam_groups, labels=fam_labels, showmeans=True)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Test Accuracy")
        plt.title("Test Accuracy by Architecture Family")
        plt.tight_layout()
        plt.savefig(out_dir / "family_boxplot.png", dpi=200)
        plt.close()

    # ---------- Figure 4: Family means with error bars ----------
    if "family" in df_acc.columns:
        fam_agg = (df_acc.groupby("family")["test_acc"]
                   .agg(["count","mean","std"])
                   .sort_values("mean", ascending=False)
                   .reset_index())
        if not fam_agg.empty:
            plt.figure(figsize=(10, 6))
            x = np.arange(len(fam_agg))
            y = fam_agg["mean"].values
            yerr = fam_agg["std"].values
            plt.bar(x, y, yerr=yerr, capsize=4)
            plt.xticks(x, fam_agg["family"].tolist(), rotation=45, ha="right")
            plt.ylabel("Mean Test Accuracy")
            plt.title("Family Averages (error bars = stdev)")
            plt.tight_layout()
            plt.savefig(out_dir / "family_means_errbars.png", dpi=200)
            plt.close()

    # ---------- Figure 5: Test Acc vs Image Size ----------
    if "img_size" in df_acc.columns:
        plt.figure(figsize=(8, 6))
        if not singles.empty and "img_size" in singles.columns:
            plt.scatter(singles["img_size"], singles["test_acc"], label="Single", marker="o", alpha=0.8)
        if not hybrids.empty and "img_size" in hybrids.columns:
            plt.scatter(hybrids["img_size"], hybrids["test_acc"], label="Hybrid", marker="^", alpha=0.8)
        plt.xlabel("Image Size")
        plt.ylabel("Test Accuracy")
        plt.title("Test Accuracy vs. Input Image Size")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "test_acc_vs_img_size.png", dpi=200)
        plt.close()

    print("Saved figures in:", out_dir)

if __name__ == "__main__":
    main()
