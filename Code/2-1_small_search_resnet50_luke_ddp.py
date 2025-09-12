#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel hyperparameter search for ResNet50 on Luke Chugh dataset.

Updates in this version:
- Robust CSV IO:
    * safe_read_results_csv() repairs malformed rows (extra/fewer commas).
    * Rank 0 reads prior CSV and broadcasts 'seen' param keys to other ranks.
    * Final write uses explicit quoting/escaping.
- DDP hygiene:
    * Set CUDA device BEFORE init_process_group to avoid warnings.
    * Always destroy_process_group() via try/finally.

Other features (as before):
- Self-launch with torchrun on A800s (physical 0,2,3) for 3 procs.
- Trial-level parallelism (no DDP wrapper for the model).
- Stratified 80/20 validation split; AdamW + ReduceLROnPlateau + early stopping.
- Label smoothing + class weights; Pad->Resize(224) transforms.
- Checkpoints saved as ckpt_<paramkey>.pt; skip trials already done (CSV or ckpt).
- Rank 0 aggregates CSV and copies best checkpoint to best_resnet50.pt.

Run:
  python small_search_resnet50_luke_ddp.py
"""

import os
import sys
import time
import json
import random
import shutil
import csv
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as F, InterpolationMode


# -----------------------------------------------------------------------------
# Self-launch with torchrun on selected physical GPUs (0,2,3)
# -----------------------------------------------------------------------------
def self_launch_if_needed():
    if "LOCAL_RANK" in os.environ:
        return  # already under torchrun

    gpu_list = "0,2,3"  # A800s only (skip T1000 at index 1)
    nproc = len(gpu_list.split(","))

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = gpu_list

    torchrun = shutil.which("torchrun")
    if torchrun:
        cmd = [torchrun, f"--nproc_per_node={nproc}", sys.argv[0], "--_launched=1"]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}", sys.argv[0], "--_launched=1"]

    for a in sys.argv[1:]:
        if a != "--_launched=1":
            cmd.append(a)

    print("Launching:", " ".join(cmd))
    os.execvpe(cmd[0], cmd, env)


self_launch_if_needed()


# -----------------------------------------------------------------------------
# Utilities: robust CSV and broadcast helpers
# -----------------------------------------------------------------------------
def safe_read_results_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read results CSV even if some rows are malformed (too many/few commas).
    Repairs by merging extra columns into the last column or padding missing ones.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return pd.DataFrame()

    header = rows[0]
    n = len(header)
    fixed_rows = []
    bad = 0

    for r in rows[1:]:
        if len(r) == n:
            fixed_rows.append(r)
        elif len(r) > n:
            merged = r[:n-1] + [",".join(r[n-1:])]
            fixed_rows.append(merged)
            bad += 1
        else:  # len(r) < n
            fixed_rows.append(r + [""] * (n - len(r)))
            bad += 1

    if bad:
        print(f"[safe_read_results_csv] Repaired {bad} malformed rows in {csv_path.name}")

    return pd.DataFrame(fixed_rows, columns=header)


def broadcast_object(obj, src=0):
    """Broadcast a picklable Python object from src to all ranks."""
    rank = int(os.environ.get("RANK", "0"))
    container = [obj if rank == src else None]
    dist.broadcast_object_list(container, src=src)
    return container[0]


# -----------------------------------------------------------------------------
# Distributed setup
# -----------------------------------------------------------------------------
def ddp_setup():
    # Set device FIRST to silence the device-id warning
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return rank, world_size, local_rank


# -----------------------------------------------------------------------------
# Transforms (Pad -> Resize 224)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Param key helpers (for skipping & checkpoint naming)
# -----------------------------------------------------------------------------
PARAM_KEYS = ["lr", "weight_decay", "dropout", "label_smoothing", "batch_size"]

def param_key(p: dict) -> str:
    # stable compact key for filenames (avoid scientific notation)
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.8f}".rstrip("0").rstrip(".")
        return str(x)
    return "__".join([f"{k}-{fmt(p[k])}" for k in PARAM_KEYS])

def ckpt_path_for(p: dict, out_dir: Path) -> Path:
    return out_dir / f"ckpt_{param_key(p)}.pt"


# -----------------------------------------------------------------------------
# Model builder (ResNet50 + optional dropout)
# -----------------------------------------------------------------------------
def build_resnet50(num_classes: int, dropout: float):
    m = models.resnet50(weights="IMAGENET1K_V1")
    in_dim = m.fc.in_features
    if dropout and float(dropout) > 0:
        m.fc = nn.Sequential(nn.Dropout(p=float(dropout)), nn.Linear(in_dim, num_classes))
    else:
        m.fc = nn.Linear(in_dim, num_classes)
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total if total else 0.0


def run_trial(params, trial_id: int, loaders, num_classes, class_weights, device, out_dir: Path):
    lr  = float(params["lr"])
    wd  = float(params["weight_decay"])
    dp  = float(params["dropout"])
    ls  = float(params["label_smoothing"])
    bs  = int(params["batch_size"])

    train_loader, val_loader, test_loader = loaders

    model = build_resnet50(num_classes, dp).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    best_val, best_epoch = -1.0, -1
    patience, wait = 7, 0
    max_epochs = 12

    ckpt_tmp = ckpt_path_for(params, out_dir)

    for epoch in range(max_epochs):
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
            best_val, best_epoch = val_acc, epoch + 1
            wait = 0
            torch.save(model.state_dict(), ckpt_tmp)
        else:
            wait += 1
            if wait >= patience:
                break

    # Test best
    model.load_state_dict(torch.load(ckpt_tmp, map_location=device))
    test_acc = evaluate(model, test_loader, device)

    return {
        "rank": int(os.environ.get("RANK", "0")),
        "trial": trial_id,
        "lr": lr,
        "weight_decay": wd,
        "dropout": dp,
        "label_smoothing": ls,
        "batch_size": bs,
        "best_val_acc": round(best_val, 6),
        "best_epoch": best_epoch,
        "test_acc": round(test_acc, 6),
        "ckpt_path": str(ckpt_tmp),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    rank, world_size, local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    print(f"[rank {rank}] LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
    print(f"[rank {rank}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if torch.cuda.is_available():
        print(f"[rank {rank}] torch sees {torch.cuda.device_count()} device(s):")
        for i in range(torch.cuda.device_count()):
            print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")

    # Reproducibility
    SEED = 42 + rank
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Paths
    PROJ_ROOT = Path("~").expanduser() / "Alzheimers"
    DATA_ROOT = PROJ_ROOT / "Data" / "Kaggle_LukeChugh_Best_Alzheimers_MRI"
    TRAIN_DIR = DATA_ROOT / "train"
    TEST_DIR  = DATA_ROOT / "test"

    OUT_DIR   = PROJ_ROOT / "Results" / "HP_Search_Luke_DDP"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH  = OUT_DIR / "resnet50_luke_small_search_ddp.csv"
    if is_main:
        print(f"[rank 0] Results dir: {OUT_DIR}")

    # Datasets & stratified 80/20 val split
    full_train_aug  = datasets.ImageFolder(str(TRAIN_DIR), transform=tf_train)
    full_train_eval = datasets.ImageFolder(str(TRAIN_DIR), transform=tf_eval)
    test_set        = datasets.ImageFolder(str(TEST_DIR),  transform=tf_eval)

    num_classes = len(full_train_aug.classes)
    class_names = full_train_aug.classes
    if is_main:
        print("Classes:", class_names)

    targets = [y for _, y in full_train_aug.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    train_ds = Subset(full_train_aug, train_idx)
    val_ds   = Subset(full_train_eval, val_idx)

    # Class weights from TRAIN subset
    train_labels = [full_train_aug.samples[i][1] for i in train_idx]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    total = class_counts.sum()
    weights = total / (class_counts + 1e-12)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    if is_main:
        print("Class counts (train):", class_counts.tolist())

    def make_loaders(bs: int):
        common = dict(batch_size=bs, num_workers=4, pin_memory=True, persistent_workers=True)
        return (
            DataLoader(train_ds, shuffle=True,  **common),
            DataLoader(val_ds,   shuffle=False, **common),
            DataLoader(test_set, shuffle=False, **common),
        )

    # Search grid
    LRS = [5e-5, 1e-4, 2e-4]
    WDS = [5e-5, 1e-4, 2e-4]
    DPS = [0.2, 0.3, 0.5]
    LSS = [0.03, 0.05, 0.07]
    BSS = [16, 32]

    grid = [
        {"lr": lr, "weight_decay": wd, "dropout": dp, "label_smoothing": ls, "batch_size": bs}
        for lr in LRS
        for wd in WDS
        for dp in DPS
        for ls in LSS
        for bs in BSS
    ]

    # Build 'seen' from existing CSV on rank 0, then broadcast
    seen_from_csv = set()
    if is_main and CSV_PATH.exists():
        try:
            df_prev = safe_read_results_csv(CSV_PATH)
            if not df_prev.empty:
                # Ensure expected columns exist
                missing = [c for c in ["lr","weight_decay","dropout","label_smoothing","batch_size"] if c not in df_prev.columns]
                if missing:
                    print(f"[rank 0] Warning: CSV missing columns {missing}; proceeding with whatever is present.")
                for _, r in df_prev.iterrows():
                    try:
                        p = {
                            "lr": float(r["lr"]),
                            "weight_decay": float(r["weight_decay"]),
                            "dropout": float(r["dropout"]),
                            "label_smoothing": float(r["label_smoothing"]),
                            "batch_size": int(float(r["batch_size"]))  # handle e.g., "32.0"
                        }
                        seen_from_csv.add(param_key(p))
                    except Exception:
                        # Skip rows that don't parse cleanly
                        continue
        except Exception as e:
            print(f"[rank 0] Warning: could not parse existing CSV for skipping: {e}")

    seen_from_csv = broadcast_object(seen_from_csv, src=0)

    # Also mark completed if a param-keyed checkpoint exists
    seen = set(seen_from_csv)
    for p in grid:
        if ckpt_path_for(p, OUT_DIR).exists():
            seen.add(param_key(p))

    # Filter grid: only run trials not seen
    grid_to_run = [p for p in grid if param_key(p) not in seen]

    # Shard across ranks
    my_grid = grid_to_run[rank::world_size]
    if is_main:
        print(f"Total trials: {len(grid)} | already done/seen: {len(seen)} | remaining: {len(grid_to_run)} | per-rank: ~{len(my_grid)}")

    # Run local trials
    TRIALS_LOCAL = []
    start = time.time()
    for t_id, params in enumerate(my_grid, 1):
        bs = int(params["batch_size"])
        loaders = make_loaders(bs)
        print(f"[rank {rank}] Trial {t_id}/{len(my_grid)} :: {params}")
        res = run_trial(params, trial_id=t_id, loaders=loaders,
                        num_classes=num_classes, class_weights=class_weights,
                        device=device, out_dir=OUT_DIR)
        print(f"[rank {rank}] -> val={res['best_val_acc']:.4f} @ epoch {res['best_epoch']} | test={res['test_acc']:.4f}")
        TRIALS_LOCAL.append(res)
    elapsed = (time.time() - start) / 60.0
    print(f"[rank {rank}] Done in {elapsed:.1f} min")

    # Gather to rank 0 and finalize CSV / best ckpt
    obj_list = [None for _ in range(world_size)]
    dist.all_gather_object(obj_list, TRIALS_LOCAL)

    if is_main:
        new_rows = [row for sub in obj_list for row in sub]
        df_new = pd.DataFrame(new_rows)

        if CSV_PATH.exists():
            df_old = safe_read_results_csv(CSV_PATH)
            df = pd.concat([df_old, df_new], ignore_index=True)
            # Drop duplicate param rows (keep last occurrence)
            df.drop_duplicates(subset=PARAM_KEYS, keep="last", inplace=True)
        else:
            df = df_new

        # Ensure numeric types where expected
        for c in ["lr","weight_decay","dropout","label_smoothing","batch_size","best_val_acc","test_acc"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "best_epoch" in df.columns:
            df["best_epoch"] = pd.to_numeric(df["best_epoch"], errors="coerce", downcast="integer")

        # Save with explicit quoting/escaping
        df.to_csv(CSV_PATH, index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\", line_terminator="\n")

        # Pick best by val_acc then test_acc
        best = df.sort_values(["best_val_acc", "test_acc"], ascending=False).iloc[0].to_dict()
        best_ckpt_src = Path(str(best["ckpt_path"]))
        best_ckpt_dst = OUT_DIR / "best_resnet50.pt"
        if best_ckpt_src.exists():
            # replace() moves across filesystems too
            best_ckpt_src.replace(best_ckpt_dst)

        print("\n=== SUMMARY (rank 0) ===")
        print(f"CSV: {CSV_PATH}")
        print(f"Best trial: {json.dumps(best, indent=2)}")
        print(f"Best checkpoint: {best_ckpt_dst}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure clean shutdown even on exceptions during setup/run
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

