#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retrain top-K configs (from HPO CSV) with multiple seeds and summarize results.

NEW IN THIS VERSION
- Creates best_params.json immediately from the HPO CSV (pre-retrain best),
  so other scripts can reuse the best hyperparameters even if you skip retraining.

Pipeline
- Reads HPO CSV (from small_search_resnet50_luke_ddp.py)
- Writes best_params.json (top row by best_val_acc then test_acc)
- Selects top-K unique param sets
- Retrains each config for multiple seeds (default 3 via --seeds 202 303 404)
- Writes:
  - retrain_per_seed_results.csv
  - retrain_summary_mean_std.csv
  - final_best_resnet50.pt (best single-seed checkpoint among retrains)
  - best_retrained.json (best-by-mean-across-seeds hyperparams & stats)

GPU behavior
- Auto-detect A800 GPUs (via nvidia-smi) and re-exec with CUDA_VISIBLE_DEVICES
  restricted to those. If none found or nvidia-smi unavailable, leave GPUs as-is.
- DataParallel uses all visible GPUs (no T1000 imbalance warning if hidden).

Paths assume LukeChugh dataset:
  ~/Alzheimers/Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/{train,test}

Usage:
  python retrain_topk_with_seeds.py \
     --results_dir ~/Alzheimers/Results/HP_Search_Luke_DDP \
     --csv resnet50_luke_small_search_ddp.csv \
     --topk 3 \
     --seeds 202 303 404
"""

import os
import sys
import json
import time
import random
import argparse
import subprocess
import csv
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# ----------------------------
# Early arg parse (for GPU setup)
# ----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--results_dir", type=str, required=True, help="Directory with HPO CSV & where to save outputs")
ap.add_argument("--csv", type=str, required=True, help="HPO CSV filename")
ap.add_argument("--topk", type=int, default=3, help="Top-K configs to retrain")
ap.add_argument("--seeds", type=int, nargs="+", default=[202, 303, 404], help="Seeds to use for retraining")
ap.add_argument("--gpus", type=str, default="auto",
                help='GPU selection: "auto" (pick A800s), "" (no override), or e.g. "0,2,3"')
args, _ = ap.parse_known_args()

# ----------------------------
# Auto GPU selection (A800) + re-exec once
# ----------------------------
def detect_gpu_indices_by_name(match_any: List[str]) -> List[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            stderr=subprocess.STDOUT, text=True
        )
    except Exception:
        return []
    picks = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            idx, name = parts[0], ",".join(parts[1:]).strip()
            if any(tok.lower() in name.lower() for tok in match_any):
                picks.append(idx)
    return picks

def ensure_cuda_visibility():
    if os.environ.get("_CUDA_VIS_SET") == "1":
        return
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    # Respect explicit --gpus (including empty string meaning "no override")
    if args.gpus != "auto":
        if args.gpus.strip():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()
        os.environ["_CUDA_VIS_SET"] = "1"
        os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)

    # Auto: prefer A800s if found
    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
        picks = detect_gpu_indices_by_name(["A800"])
        if picks:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(picks)
            os.environ["_CUDA_VIS_SET"] = "1"
            os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)

ensure_cuda_visibility()

# Only now is it safe to import torch CUDA bits
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as F, InterpolationMode

# ----------------------------
# Stable ordering + device
# ----------------------------
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# CSV robustness
# ----------------------------
def safe_read_results_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return pd.DataFrame()
    header = rows[0]; n = len(header)
    fixed = []; bad = 0
    for r in rows[1:]:
        if len(r) == n:
            fixed.append(r)
        elif len(r) > n:
            fixed.append(r[:n-1] + [",".join(r[n-1:])]); bad += 1
        else:
            fixed.append(r + [""]*(n-len(r))); bad += 1
    if bad:
        print(f"[safe_read_results_csv] Repaired {bad} malformed rows in {csv_path.name}")
    return pd.DataFrame(fixed, columns=header)


# ----------------------------
# Transforms (Pad -> Resize 224)
# ----------------------------
class PadToSquare:
    def __call__(self, img):
        w, h = F.get_image_size(img)
        s = max(w, h)
        pad_l = (s - w) // 2
        pad_r = s - w - pad_l
        pad_t = (s - h) // 2
        pad_b = s - h - pad_t
        return F.pad(img, [pad_l, pad_t, pad_r, pad_b], fill=0)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

tf_train = transforms.Compose([
    PadToSquare(),
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

tf_eval = transforms.Compose([
    PadToSquare(),
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_resnet50(num_classes: int, dropout: float):
    m = models.resnet50(weights="IMAGENET1K_V1")
    in_dim = m.fc.in_features
    if dropout and dropout > 0:
        m.fc = nn.Sequential(nn.Dropout(p=float(dropout)), nn.Linear(in_dim, num_classes))
    else:
        m.fc = nn.Linear(in_dim, num_classes)
    return m

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total if total else 0.0

def fmt_float(x: float) -> str:
    return f"{float(x):.8f}".rstrip("0").rstrip(".")

def param_key_from_dict(p: Dict[str, Any]) -> str:
    # Stable compact key for filenames (avoids scientific notation)
    order = ["lr", "weight_decay", "dropout", "label_smoothing", "batch_size"]
    parts = []
    for k in order:
        v = p[k]
        if isinstance(v, (float, np.floating)):
            parts.append(f"{k}-{fmt_float(float(v))}")
        else:
            parts.append(f"{k}-{int(v) if k=='batch_size' else v}")
    return "__".join(parts)


# ----------------------------
# Data split & loaders
# ----------------------------
def make_split_loaders(train_dir: Path, test_dir: Path, batch_size: int, seed_for_split: int = 42):
    full_train_aug  = datasets.ImageFolder(str(train_dir), transform=tf_train)
    full_train_eval = datasets.ImageFolder(str(train_dir), transform=tf_eval)
    test_set        = datasets.ImageFolder(str(test_dir),  transform=tf_eval)

    targets = [y for _, y in full_train_aug.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed_for_split)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_ds = Subset(full_train_aug, train_idx)
    val_ds   = Subset(full_train_eval, val_idx)

    num_classes = len(full_train_aug.classes)
    class_names = full_train_aug.classes

    common = dict(batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common)
    test_loader  = DataLoader(test_set, shuffle=False, **common)

    train_labels = [full_train_aug.samples[i][1] for i in train_idx]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    total = class_counts.sum()
    weights = total / (class_counts + 1e-12)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    return train_loader, val_loader, test_loader, num_classes, class_names, class_weights


# ----------------------------
# Train one config across seeds
# ----------------------------
def train_one_run(params: dict, seeds: list, proj_root: Path, results_dir: Path):
    train_dir = proj_root / "Data" / "Kaggle_LukeChugh_Best_Alzheimers_MRI" / "train"
    test_dir  = proj_root / "Data" / "Kaggle_LukeChugh_Best_Alzheimers_MRI" / "test"

    SPLIT_SEED = 42  # fixed split across seeds

    best_seed_ckpt = None
    best_seed_val  = -1.0
    per_seed_rows  = []

    for seed in seeds:
        set_seed(int(seed))

        train_loader, val_loader, test_loader, num_classes, _, class_weights = make_split_loaders(
            train_dir, test_dir, batch_size=int(params["batch_size"]), seed_for_split=SPLIT_SEED
        )

        model = build_resnet50(num_classes, float(params["dropout"])).to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)  # uses all visible GPUs

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(params["label_smoothing"]))
        optimizer = optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)

        max_epochs = 30
        patience   = 10
        wait = 0
        best_val, best_epoch = -1.0, -1

        pkey = param_key_from_dict(params)
        ckpt_path = results_dir / f"retrain_ckpt_{pkey}__seed-{int(seed)}.pt"

        for epoch in range(1, max_epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            val_acc = evaluate(model, val_loader, device)
            scheduler.step(val_acc)

            if val_acc > best_val + 1e-6:
                best_val, best_epoch = val_acc, epoch
                wait = 0
                core = model.module if isinstance(model, nn.DataParallel) else model
                torch.save({
                    "state_dict": core.state_dict(),
                    "params": dict(params),
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "val_acc": float(val_acc)
                }, ckpt_path)
            else:
                wait += 1
                if wait >= patience:
                    break

        # Test best weights
        payload = torch.load(ckpt_path, map_location=device)
        core = model.module if isinstance(model, nn.DataParallel) else model
        core.load_state_dict(payload["state_dict"])
        test_acc = evaluate(core, test_loader, device)

        per_seed_rows.append({
            **params,
            "seed": int(seed),
            "best_val_acc": round(float(payload["val_acc"]), 6),
            "best_epoch": int(payload["epoch"]),
            "test_acc": round(float(test_acc), 6),
            "ckpt_path": str(ckpt_path)
        })

        if payload["val_acc"] > best_seed_val:
            best_seed_val = float(payload["val_acc"])
            best_seed_ckpt = ckpt_path

    return per_seed_rows, best_seed_ckpt


# ----------------------------
# Main
# ----------------------------
def main():
    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / args.csv
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '') or '<all visible>'}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"Using {n} visible GPU(s):")
        for i in range(n):
            print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available; using CPU.")

    # Read & sort HPO CSV
    df = safe_read_results_csv(csv_path)
    if df.empty:
        print("HPO CSV is empty; nothing to retrain.")
        sys.exit(0)

    # Coerce numerics
    for c in ["lr","weight_decay","dropout","label_smoothing","batch_size","best_val_acc","test_acc","best_epoch"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df_sorted = df.sort_values(["best_val_acc", "test_acc"], ascending=False, na_position="last").reset_index(drop=True)

    # ---- Write best_params.json from the top row of HPO CSV ----
    best_row = df_sorted.iloc[0].to_dict()
    best_params = {
        "arch": "ResNet50",
        "lr": float(best_row["lr"]),
        "weight_decay": float(best_row["weight_decay"]),
        "dropout": float(best_row["dropout"]),
        "label_smoothing": float(best_row["label_smoothing"]),
        "batch_size": int(best_row["batch_size"]),
        "best_val_acc": float(best_row["best_val_acc"]),
        "best_epoch": int(best_row["best_epoch"]) if not pd.isna(best_row.get("best_epoch", np.nan)) else None,
        "test_acc": float(best_row["test_acc"]),
        "checkpoint": str(best_row["ckpt_path"]) if "ckpt_path" in best_row and pd.notna(best_row["ckpt_path"]) else "",
        "source": "HPO_CSV"
    }
    (results_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))
    print(f"Wrote HPO best params to: {results_dir / 'best_params.json'}")

    # Select top-K UNIQUE param sets
    param_cols = ["lr", "weight_decay", "dropout", "label_smoothing", "batch_size"]
    topk_df = df_sorted[param_cols].drop_duplicates().head(args.topk).reset_index(drop=True)
    print(f"Selected top-{len(topk_df)} unique configs for retraining:\n{topk_df}")

    proj_root = Path("~").expanduser() / "Alzheimers"
    per_seed_out = results_dir / "retrain_per_seed_results.csv"
    summary_out  = results_dir / "retrain_summary_mean_std.csv"
    overall_best = results_dir / "final_best_resnet50.pt"

    all_rows = []
    best_overall_val = -1.0
    best_overall_ckpt = None

    for i, row in topk_df.iterrows():
        params = {
            "lr": float(row["lr"]),
            "weight_decay": float(row["weight_decay"]),
            "dropout": float(row["dropout"]),
            "label_smoothing": float(row["label_smoothing"]),
            "batch_size": int(row["batch_size"]),
        }
        print(f"\n=== Retraining config {i+1}/{len(topk_df)}: {params} ===")
        per_seed_rows, best_ckpt_for_config = train_one_run(params, args.seeds, proj_root, results_dir)
        all_rows.extend(per_seed_rows)

        # Track best across configs by highest validation among best seeds
        best_cfg_val = max(r["best_val_acc"] for r in per_seed_rows)
        if best_cfg_val > best_overall_val:
            best_overall_val = best_cfg_val
            best_overall_ckpt = best_ckpt_for_config

    # Save per-seed results
    df_seeds = pd.DataFrame(all_rows)
    df_seeds.to_csv(per_seed_out, index=False,
                    quoting=csv.QUOTE_MINIMAL, escapechar="\\", line_terminator="\n")
    print(f"\nSaved per-seed results: {per_seed_out}")

    # Summarize mean ± std per config
    group_cols = param_cols
    agg = df_seeds.groupby(group_cols, dropna=False).agg(
        val_mean=("best_val_acc", "mean"),
        val_std =("best_val_acc", "std"),
        test_mean=("test_acc", "mean"),
        test_std =("test_acc", "std"),
        n=("seed", "count")
    ).reset_index()

    for c in ["val_mean","val_std","test_mean","test_std"]:
        agg[c] = agg[c].astype(float).round(6)

    agg.to_csv(summary_out, index=False,
               quoting=csv.QUOTE_MINIMAL, escapechar="\\", line_terminator="\n")
    print(f"Saved summary (mean ± std): {summary_out}")

    # Copy overall best single-seed checkpoint
    if best_overall_ckpt and Path(best_overall_ckpt).exists():
        Path(best_overall_ckpt).replace(overall_best)
        print(f"Saved overall best checkpoint (single-seed) to: {overall_best}")
    else:
        print("No overall best checkpoint found to copy.")

    # Also write best_retrained.json by mean across seeds
    agg_sorted = agg.sort_values(["val_mean","test_mean"], ascending=False).reset_index(drop=True)
    best_mean = agg_sorted.iloc[0].to_dict()
    best_retrained = {
        "arch": "ResNet50",
        "lr": float(best_mean["lr"]),
        "weight_decay": float(best_mean["weight_decay"]),
        "dropout": float(best_mean["dropout"]),
        "label_smoothing": float(best_mean["label_smoothing"]),
        "batch_size": int(best_mean["batch_size"]),
        "val_mean": float(best_mean["val_mean"]),
        "val_std": float(best_mean["val_std"]) if not np.isnan(best_mean["val_std"]) else None,
        "test_mean": float(best_mean["test_mean"]),
        "test_std": float(best_mean["test_std"]) if not np.isnan(best_mean["test_std"]) else None,
        "n_seeds": int(best_mean["n"]),
        "checkpoint_single_seed_best": str(overall_best),
        "source": "RETRAIN_MEAN_ACROSS_SEEDS"
    }
    (results_dir / "best_retrained.json").write_text(json.dumps(best_retrained, indent=2))
    print(f"Wrote retrain best (mean across seeds) to: {results_dir / 'best_retrained.json'}")

    print("\n=== Top configs (mean ± std) ===")
    print(agg_sorted.head(10))


if __name__ == "__main__":
    main()
