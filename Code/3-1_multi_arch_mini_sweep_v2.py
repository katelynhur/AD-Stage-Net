#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train multiple architectures with a tiny per-arch sweep, log every epoch, and save best/final checkpoints.

New:
- --best_params_json accepts either a GLOBAL dict:
    {
      "lr": 5e-5,
      "label_smoothing": 0.05,
      "dropout": 0.3,
      "weight_decay": 1e-4,
      "batch_size": 32
    }
  or a PER-ARCH dict:
    {
      "ResNet50": {"lr": 5e-5, "dropout": 0.2, "label_smoothing": 0.05, "weight_decay": 1e-4, "batch_size": 32},
      "EffNetB0": {...},
      ...
    }
- If provided, these values override CLI defaults when present for that arch.
- LR grid is centered at the chosen lr × [0.5, 1.0, 2.0].
- Dropout candidates come from a small arch-aware function seeded by the chosen base dropout.
- **Auto GPU selection** (default): prefers A800s (via nvidia-smi) and re-execs with CUDA_VISIBLE_DEVICES.
  Override with --gpus "0,2,3" or disable override with --gpus "".
  Backward-compat flag --visible_devices is still accepted.
- **DDP implementation** with self-launch via torchrun; one process per visible GPU.

Architectures covered (224x unless noted):
- CNN_Small (placeholder; enable if desired)
- ResNet50 / 101 / 152
- DenseNet121 / 161 / 169 / 201
- EfficientNet-B0
- MobileNetV2 / MobileNetV3-Large
- ResNeXt50_32x4d / ResNeXt101_32x8d
- VGG16-BN
- InceptionV3 (299x, aux logits handled)

Data assumption (Luke dataset):
- Train/Val: taken from Luke's 'train' split by an internal stratified split (default 80/20)
- Test: Luke's 'test' split

Outputs (under <proj_root>/<results_subdir>/):
- Per-arch, per-config folder:
  - params.json (what was used)
  - epoch_log.csv (epoch, train_loss, val_loss, train_acc, val_acc, lr)
  - best.pt (best val checkpoint, state_dict)   [rank 0 only]
  - final.pt (last epoch checkpoint, state_dict) [rank 0 only]
  - complete.txt (marker)                        [rank 0 only]
- Global CSV: arch_sweep_results.csv (row per config with best val acc, test acc, etc.) [rank 0 only]
"""

import os
import sys
import json
import time
import hashlib
import random
import argparse
import warnings
import subprocess
import shutil
import math
import gc
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# ----------------------------
# Early arg parse (for GPU setup)
# ----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--proj_root", type=str, required=True)
ap.add_argument("--train_dir", type=str, required=True)  # Luke train
ap.add_argument("--test_dir", type=str, required=True)   # Luke test
ap.add_argument("--results_subdir", type=str, default="Results/ArchSweep")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--num_workers", type=int, default=4)
ap.add_argument("--best_lr", type=float, default=5e-5)
ap.add_argument("--best_label_smoothing", type=float, default=0.05)
ap.add_argument("--best_dropout", type=float, default=0.3)
ap.add_argument("--weight_decay", type=float, default=1e-4)
ap.add_argument("--val_split", type=float, default=0.2)
ap.add_argument("--best_params_json", type=str, default="",
                help="Path to best_retrained.json or best_params.json; "
                     "if provided, values inside override defaults (globally or per-arch) when present")
ap.add_argument("--single_best_only", type=int, default=0,
                help="If 1, disable LR×dropout sweep and train exactly once per arch with best params.")
ap.add_argument("--extra_tests", type=str, default="",
                help='Semicolon list like "Marco:Data/Marco/test;Falah:Data/Falah/test"')
# NEW: combine-tests mode (minimal addition)
ap.add_argument("--combine_tests", type=int, default=0,
                help="If 1, combine --test_dir and all --extra_tests into a temporary test set and evaluate once. "
                     "Per-extra evaluation is skipped (no acc_* columns).")

# GPU control (new): prefer A800s automatically; override with --gpus; legacy --visible_devices honored if set.
ap.add_argument("--gpus", type=str, default="auto",
                help='GPU selection: "auto" (prefer A800s), "" (no override), or e.g. "0,2,3"')
ap.add_argument("--visible_devices", type=str, default="",
                help="(deprecated) e.g., '0,2,3'; if provided and --gpus is 'auto', this will be used.")
ap.add_argument("--skip_completed", type=int, default=1)
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
    # Prevent infinite recursion
    if os.environ.get("_CUDA_VIS_SET") == "1":
        return
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    # If explicit --gpus is given (including empty string), respect it.
    if args.gpus != "auto":
        if args.gpus.strip():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()
        # else: empty string means no override
        os.environ["_CUDA_VIS_SET"] = "1"
        os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)

    # Backward compat: if legacy --visible_devices is set, use it in auto mode.
    if args.visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
        os.environ["_CUDA_VIS_SET"] = "1"
        os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)

    # Auto mode: prefer A800s if present; otherwise leave as-is.
    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
        picks = detect_gpu_indices_by_name(["A800"])
        if picks:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(picks)
            os.environ["_CUDA_VIS_SET"] = "1"
            os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)

ensure_cuda_visibility()

def self_launch_with_torchrun_if_needed():
    if os.environ.get("LOCAL_RANK") is not None:
        return
    if os.environ.get("_SELF_LAUNCHED") == "1":
        return
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis.strip():
        nproc = len([x for x in vis.split(",") if x.strip() != ""])
    else:
        # try to count GPUs
        try:
            out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
            nproc = max(1, len([ln for ln in out.strip().splitlines() if ln.strip()]))
        except Exception:
            nproc = 1
    if nproc <= 1:
        return  # single process path (no DDP)
    torchrun = shutil.which("torchrun")
    if torchrun:
        cmd = [torchrun, f"--nproc_per_node={nproc}", sys.argv[0]]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}", sys.argv[0]]
    cmd.extend(sys.argv[1:])
    env = os.environ.copy()
    env["_SELF_LAUNCHED"] = "1"
    os.execvpe(cmd[0], cmd, env)

self_launch_with_torchrun_if_needed()

# ----------------------------
# Now it's safe to import torch/torchvision and set up DDP
# ----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torchvision import models, transforms, datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

mp.set_sharing_strategy("file_system")
warnings.filterwarnings("ignore", category=UserWarning)

def ddp_setup():
    if os.environ.get("LOCAL_RANK") is None:
        # not under torchrun: fallback to single-process / single-GPU or CPU
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return rank, world_size, local_rank, device, False
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device, True

rank, world_size, local_rank, device, using_ddp = ddp_setup()
is_main = (rank == 0)

# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int = 42):
    seed = seed + rank  # different initial seeds across ranks
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Transforms (Pad -> Resize)
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
IMAGENET_STD = [0.229, 0.224, 0.225]

def make_transforms(arch_name: str):
    size = 299 if arch_name.lower().startswith("inception") else 224
    tf_train = transforms.Compose([
        PadToSquare(),
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tf_eval = transforms.Compose([
        PadToSquare(),
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf_train, tf_eval, size

# ----------------------------
# Model builders
# ----------------------------
def _resnet(ctor, num):
    m = ctor(weights="IMAGENET1K_V1")
    in_dim = m.fc.in_features
    m.fc = nn.Linear(in_dim, num)
    return m

def _densenet(ctor, num):
    m = ctor(weights="IMAGENET1K_V1")
    in_dim = m.classifier.in_features
    m.classifier = nn.Linear(in_dim, num)
    return m

def _effnet(ctor, num):
    m = ctor(weights="IMAGENET1K_V1")
    in_dim = m.classifier[1].in_features
    # keep built-in dropout in m.classifier[0]; replace final linear
    m.classifier[1] = nn.Linear(in_dim, num)
    return m

def _mobilenet(ctor, num):
    m = ctor(weights="IMAGENET1K_V1")
    in_dim = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_dim, num)
    return m

def _vgg(ctor, num):
    m = ctor(weights="IMAGENET1K_V1")
    in_dim = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_dim, num)
    return m

def build_inception(num_classes):
    m = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
    in_dim = m.fc.in_features
    m.fc = nn.Linear(in_dim, num_classes)
    return m

# Placeholder small CNN (swap with your own if desired)
class YourSmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        z = self.features(x)
        z = z.view(z.size(0), -1)
        return self.classifier(z)

MODEL_REGISTRY_224 = {
    "CNN_Small": lambda num: YourSmallCNN(num),
    "ResNet50": lambda num: _resnet(models.resnet50, num),
    "ResNet101": lambda num: _resnet(models.resnet101, num),
    "ResNet152": lambda num: _resnet(models.resnet152, num),
    "DenseNet121": lambda num: _densenet(models.densenet121, num),
    "DenseNet161": lambda num: _densenet(models.densenet161, num),
    "DenseNet169": lambda num: _densenet(models.densenet169, num),
    "DenseNet201": lambda num: _densenet(models.densenet201, num),
    "EffNetB0": lambda num: _effnet(models.efficientnet_b0, num),
    "MobileNetV2": lambda num: _mobilenet(models.mobilenet_v2, num),
    "MobileNetV3_L": lambda num: _mobilenet(models.mobilenet_v3_large, num),
    "ResNeXt50_32x4d": lambda num: _resnet(models.resnext50_32x4d, num),
    "ResNeXt101_32x8d": lambda num: _resnet(models.resnext101_32x8d, num),
    "VGG16": lambda num: _vgg(models.vgg16_bn, num),
    # Inception handled separately for 299
}

# Insert head dropout if requested (where applicable)
def add_head_dropout(model, arch, dp, num_classes):
    if dp is None or dp <= 0:
        return model
    # Common cases:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_dim = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dp), nn.Linear(in_dim, num_classes))
        return model
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            # replace last Linear and insert dropout before it
            for i in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[i], nn.Linear):
                    in_dim = model.classifier[i].in_features
                    new_seq = list(model.classifier)
                    new_seq[i] = nn.Linear(in_dim, num_classes)
                    if i == 0 or not isinstance(new_seq[i - 1], nn.Dropout):
                        new_seq.insert(i, nn.Dropout(dp))
                    model.classifier = nn.Sequential(*new_seq)
                    return model
        elif isinstance(model.classifier, nn.Linear):
            in_dim = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(dp), nn.Linear(in_dim, num_classes))
            return model
    return model  # fallback: unchanged

# ----------------------------
# DDP-aware train / eval (global aggregation)
# ----------------------------
def _all_reduce_sum(x: torch.Tensor):
    if using_ddp and world_size > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

def train_one_epoch(model, loader, ce, optimizer, device, arch_name):
    model.train()
    loss_sum = torch.tensor(0.0, device=device)
    correct_sum = torch.tensor(0.0, device=device)
    total_sum = torch.tensor(0.0, device=device)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        if arch_name.lower().startswith("inception") and isinstance(out, tuple):
            main_out, aux_out = out
            loss = ce(main_out, yb) + 0.4 * ce(aux_out, yb)
            logits = main_out
        else:
            loss = ce(out, yb)
            logits = out
        loss.backward()
        optimizer.step()

        bsz = yb.size(0)
        loss_sum += loss.detach() * bsz
        correct_sum += (logits.argmax(1) == yb).sum()
        total_sum += bsz

    # aggregate across ranks
    loss_sum = _all_reduce_sum(loss_sum)
    correct_sum = _all_reduce_sum(correct_sum)
    total_sum = _all_reduce_sum(total_sum)

    total = max(total_sum.item(), 1.0)
    return float(loss_sum.item() / total), float(correct_sum.item() / total)

@torch.no_grad()
def evaluate(model, loader, ce, device, arch_name):
    model.eval()
    loss_sum = torch.tensor(0.0, device=device)
    correct_sum = torch.tensor(0.0, device=device)
    total_sum = torch.tensor(0.0, device=device)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb)
        if arch_name.lower().startswith("inception") and isinstance(out, tuple):
            out = out[0]
        loss = ce(out, yb)
        bsz = yb.size(0)
        loss_sum += loss * bsz
        correct_sum += (out.argmax(1) == yb).sum()
        total_sum += bsz

    # aggregate across ranks
    loss_sum = _all_reduce_sum(loss_sum)
    correct_sum = _all_reduce_sum(correct_sum)
    total_sum = _all_reduce_sum(total_sum)

    total = max(total_sum.item(), 1.0)
    return float(loss_sum.item() / total), float(correct_sum.item() / total)

# ----------------------------
# Utilities
# ----------------------------
def hash_config(d: dict):
    s = json.dumps(d, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:10]

def write_epoch_csv(path: Path, row: dict):
    if not is_main:
        return
    header = not path.exists()
    df = pd.DataFrame([row])
    df.to_csv(path, index=False, header=header, mode="a")

def evaluate_on_test(model, test_loader, device, arch_name):
    ce = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, ce, device, arch_name)
    return test_loss, test_acc

# Loader shutdown + skip helpers
def shutdown_loader(loader):
    if loader is None:
        return
    try:
        it = getattr(loader, "_iterator", None)
        if it is not None:
            it._shutdown_workers()  # best-effort; supported in recent PyTorch
    except Exception:
        pass

def is_run_complete(run_dir: Path) -> bool:
    if (run_dir / "complete.txt").exists():
        return True
    # robust fallback
    needed = ["best.pt", "final.pt", "params.json", "epoch_log.csv"]
    return all((run_dir / n).exists() for n in needed)

# --- helpers to load best params JSON ---
def load_best_params_json(path_str: str, arch_name: str):
    """Return dict of best params for this arch, or global, or {} if unavailable."""
    if not path_str:
        return {}
    p = Path(os.path.expanduser(path_str))
    if not p.exists():
        if is_main:
            print(f"[WARN] best_params_json not found: {p}")
        return {}
    try:
        with open(p, "r") as f:
            data = json.load(f)
    except Exception as e:
        if is_main:
            print(f"[WARN] failed to parse {p}: {e}")
        return {}
    # If top-level looks like global (single set), return it
    if any(k in data for k in ["lr", "label_smoothing", "dropout", "weight_decay", "batch_size"]):
        return data
    # Else assume per-arch mapping
    return data.get(arch_name, {})

def pick_or_default(d, key, fallback):
    v = d.get(key, None)
    try:
        return type(fallback)(v) if v is not None else fallback
    except Exception:
        return fallback

# --- NEW (minimal): helpers to build a temporary combined test folder ---
def _safe_link_or_copy(src: Path, dst: Path):
    try:
        os.symlink(src, dst)
    except Exception:
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass

def build_combined_test_dir(dest_dir: Path, main_test: Path, extras: list):
    """
    Create an ImageFolder-compatible directory at dest_dir that contains
    images from main_test and each extra test root. Class subfolders are preserved.
    Filename conflicts are resolved by prefixing the dataset label.
      extras: list of (label, abs_path)
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    datasets_to_merge = [("_main", Path(main_test))] + [(lab, Path(p)) for lab, p in extras]
    for label, src_root in datasets_to_merge:
        if not src_root.exists():
            continue
        for class_name in sorted(os.listdir(src_root)):
            class_src = src_root / class_name
            if not class_src.is_dir():
                continue
            class_dst = dest_dir / class_name
            class_dst.mkdir(parents=True, exist_ok=True)
            for fname in os.listdir(class_src):
                src_file = class_src / fname
                if not src_file.is_file():
                    continue
                target = class_dst / fname
                if target.exists():
                    stem, ext = os.path.splitext(fname)
                    alt = f"{label}_{stem}{ext}"
                    target = class_dst / alt
                    k = 1
                    while target.exists():
                        alt = f"{label}_{stem}_{k}{ext}"
                        target = class_dst / alt
                        k += 1
                _safe_link_or_copy(src_file, target)

# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(42)

    proj_root = Path(os.path.expanduser(args.proj_root))
    train_dir = proj_root / args.train_dir
    test_dir = proj_root / args.test_dir
    out_root = proj_root / args.results_subdir
    if is_main:
        out_root.mkdir(parents=True, exist_ok=True)

    # Parse and de-dup extra tests vs main test_dir ---
    main_test_abs = (proj_root / args.test_dir).resolve()
    extra_tests = []  # list[(name, abs_path)]
    if getattr(args, "extra_tests", ""):
        parts = [p.strip() for p in args.extra_tests.split(";") if p.strip()]
        for it in parts:
            if ":" in it:
                name, rel = it.split(":", 1)
            else:
                rel = it
                name = Path(rel).name
            p_abs = (proj_root / rel).resolve()
            if p_abs == main_test_abs:
                if is_main:
                    print(f"[extra_tests] Skipping '{name}' because it matches --test_dir")
                continue
            extra_tests.append((name, p_abs))

    # Dataset (ImageFolder) and stratified split of Luke 'train'
    base_tf = transforms.Compose([transforms.ToTensor()])  # minimal just to read
    base_ds = datasets.ImageFolder(str(train_dir), transform=base_tf)
    class_names = base_ds.classes
    y_all = np.array([y for _, y in base_ds.samples])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(y_all)), y_all))

    # Architecture list
    ARCH_LIST = [
        "CNN_Small",
        "ResNet50", "ResNet101", "ResNet152",
        "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201",
        "EffNetB0",
        "MobileNetV2", "MobileNetV3_L",
        "ResNeXt50_32x4d", "ResNeXt101_32x8d",
        "VGG16",
        "InceptionV3",
    ]

    # Tiny per-arch sweep (or single-best)
    if args.single_best_only:
        LR_SCALE = [1.0]
        def dp_candidates(arch, base_dp):  # single value
            return [base_dp]
    else:
        LR_SCALE = [0.5, 1.0, 2.0]
        def dp_candidates(arch, base_dp):
            if "EffNet" in arch: return [0.0, min(0.2, base_dp)]
            if arch == "VGG16":  return [0.5, max(0.3, base_dp)]
            if arch == "InceptionV3": return [base_dp, 0.2]
            return [base_dp, 0.0]

    # Results aggregator
    global_csv = out_root / "arch_sweep_results.csv"
    if is_main and not global_csv.exists():
        pd.DataFrame(columns=[
            "arch", "cfg_hash", "lr", "dropout", "weight_decay", "label_smoothing", "batch_size",
            "best_val_acc", "best_epoch", "test_acc", "img_size", "run_dir"
        ]).to_csv(global_csv, index=False)

    if is_main:
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "") or "<all visible>")
        print("CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
        print(f"[rank {rank}] WORLD_SIZE={world_size} LOCAL_RANK={local_rank}")

    num_classes = len(class_names)
    persistent_workers = False  # safer for long sweeps

    for arch in ARCH_LIST:
        # Pull best defaults for this arch (or global) if provided
        best = load_best_params_json(args.best_params_json, arch)
        lr_center = pick_or_default(best, "lr", args.best_lr)
        label_smoothing = pick_or_default(best, "label_smoothing", args.best_label_smoothing)
        base_dropout = pick_or_default(best, "dropout", args.best_dropout)
        weight_decay = pick_or_default(best, "weight_decay", args.weight_decay)
        base_batch_size = pick_or_default(best, "batch_size", args.batch_size)

        for lr in [lr_center * s for s in LR_SCALE]:
            for dp in dp_candidates(arch, base_dropout):
                # Build config + hash (global→per-rank BS)
                global_bs   = int(base_batch_size)
                per_rank_bs = max(1, math.ceil(global_bs / max(1, world_size)))
                bs_for_arch = per_rank_bs if arch != "InceptionV3" else max(2, per_rank_bs // 2)

                if is_main:
                    print(f"[BS] arch={arch} world_size={world_size} global={global_bs} "
                          f"per_rank={per_rank_bs} -> used={bs_for_arch}{' (Inception/2)' if arch=='InceptionV3' else ''}")

                cfg = {
                    "arch": arch,
                    "lr": float(lr),
                    "dropout": float(dp),
                    "weight_decay": float(weight_decay),
                    "label_smoothing": float(label_smoothing),
                    "batch_size": int(bs_for_arch),       # per-rank (kept for back-compat)
                    "global_batch_size": int(global_bs),  # record intended global
                    "epochs": int(args.epochs),
                    "val_split": float(args.val_split),
                }
                cfg_hash = hash_config(cfg)
                run_dir = out_root / arch / cfg_hash
                if is_main:
                    run_dir.mkdir(parents=True, exist_ok=True)

                # Skip if completed
                if args.skip_completed and is_run_complete(run_dir):
                    if is_main:
                        print(f"[SKIP] {arch} cfg={cfg_hash} already complete.")
                    if using_ddp:
                        dist.barrier()
                    continue

                # Build per-arch transforms
                tf_train, tf_eval, size = make_transforms(arch)

                # Real datasets
                full_train_tf = datasets.ImageFolder(str(train_dir), transform=tf_train)
                ds_train = Subset(full_train_tf, train_idx.tolist())
                ds_val = Subset(datasets.ImageFolder(str(train_dir), transform=tf_eval), val_idx.tolist())
                ds_test = datasets.ImageFolder(str(test_dir), transform=tf_eval)

                # Samplers for DDP
                if using_ddp and world_size > 1:
                    #train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
                    train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
                    val_sampler   = DistributedSampler(ds_val,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
                    test_sampler  = DistributedSampler(ds_test,  num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
                else:
                    train_sampler = None
                    val_sampler = None
                    test_sampler = None

                # Loaders
                train_loader = DataLoader(
                    ds_train, batch_size=bs_for_arch,
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers,
                    drop_last=True
                )
                val_loader = DataLoader(
                    ds_val, batch_size=bs_for_arch,
                    shuffle=False, sampler=val_sampler,
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers
                )
                test_loader = DataLoader(
                    ds_test, batch_size=bs_for_arch,
                    shuffle=False, sampler=test_sampler,
                    num_workers=args.num_workers, pin_memory=False,  # safer for cleanup
                    persistent_workers=False
                )

                # Combined test mode ---
                combined_tmp_dir = None
                if args.combine_tests and extra_tests:
                    combined_tmp_dir = run_dir / "_tmp_combined_test"
                    if is_main:
                        if combined_tmp_dir.exists():
                            shutil.rmtree(combined_tmp_dir, ignore_errors=True)
                        build_combined_test_dir(combined_tmp_dir, Path(test_dir), extra_tests)
                    if using_ddp:
                        dist.barrier()
                    ds_test = datasets.ImageFolder(str(combined_tmp_dir), transform=tf_eval)
                    if using_ddp and world_size > 1:
                        test_sampler = DistributedSampler(ds_test, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
                    else:
                        test_sampler = None
                    test_loader = DataLoader(
                        ds_test, batch_size=bs_for_arch,
                        shuffle=False, sampler=test_sampler,
                        num_workers=args.num_workers, pin_memory=False, persistent_workers=False
                    )

                # Build model
                if arch == "InceptionV3":
                    model = build_inception(num_classes)
                else:
                    builder = MODEL_REGISTRY_224[arch]
                    model = builder(num_classes)
                model = add_head_dropout(model, arch, dp, num_classes)
                model = model.to(device)

                # DDP wrap if applicable
                if using_ddp and world_size > 1:
                    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

                # Loss / Opt / Sched
                ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
                )

                # Logging (rank 0)
                if is_main:
                    params_path = run_dir / "params.json"
                    with open(params_path, "w") as f:
                        json.dump({**cfg, "img_size": int(size), "run_dir": str(run_dir)}, f, indent=2)

                epoch_csv = run_dir / "epoch_log.csv"
                best_path = run_dir / "best.pt"
                final_path = run_dir / "final.pt"

                best_val_acc = -1.0
                best_epoch = -1
                patience, wait = 8, 0

                for epoch in range(1, args.epochs + 1):
                    if using_ddp and world_size > 1 and isinstance(train_sampler, DistributedSampler):
                        train_sampler.set_epoch(epoch)
                    t0 = time.time()
                    tr_loss, tr_acc = train_one_epoch(model, train_loader, ce, optimizer, device, arch)
                    if using_ddp and world_size > 1 and isinstance(val_sampler, DistributedSampler):
                        val_sampler.set_epoch(epoch)
                    val_loss, val_acc = evaluate(model, val_loader, ce, device, arch)

                    scheduler.step(val_acc)
                    curr_lr = optimizer.param_groups[0]["lr"]

                    # Epoch log row (rank 0)
                    if is_main:
                        row = {
                            "epoch": epoch,
                            "train_loss": tr_loss,
                            "val_loss": val_loss,
                            "train_acc": tr_acc,
                            "val_acc": val_acc,
                            "lr": curr_lr,
                            "secs": time.time() - t0,
                        }
                        write_epoch_csv(epoch_csv, row)
                        print(f"[{arch} | {cfg_hash} | rank0] Epoch {epoch:02d}/{args.epochs} "
                              f"train_acc={tr_acc:.4f} val_acc={val_acc:.4f} lr={curr_lr:.2e}")

                    # Early stopping & best save (rank 0)
                    improved = val_acc > best_val_acc
                    if improved:
                        best_val_acc = val_acc
                        best_epoch = epoch
                        if is_main:
                            to_save = model.module if isinstance(model, DDP) else model
                            torch.save(to_save.state_dict(), best_path)
                            print(f"  ↑ New best {best_val_acc:.4f} @ epoch {epoch}; saved {best_path}")
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            if is_main:
                                print(f"  Early stop at epoch {epoch} (best {best_val_acc:.4f} @ {best_epoch})")
                            break

                    if using_ddp:
                        dist.barrier()

                # Save final (rank 0)
                if is_main:
                    to_save = model.module if isinstance(model, DDP) else model
                    torch.save(to_save.state_dict(), final_path)

                if using_ddp:
                    dist.barrier()

                # Evaluate best on test (reload best weights into the object we call)
                to_eval = model.module if isinstance(model, DDP) else model

                if using_ddp and world_size > 1:
                    # Load on rank0, then broadcast the Python object safely to all ranks
                    if is_main:
                        state_dict = torch.load(best_path, map_location="cpu")  # CPU is fine for broadcast
                    else:
                        state_dict = None
                    obj_list = [state_dict]
                    dist.broadcast_object_list(obj_list, src=0)
                    state_dict = obj_list[0]
                    to_eval.load_state_dict(state_dict, strict=True)
                else:
                    # Single process (or no DDP)
                    state_dict = torch.load(best_path, map_location=device)
                    to_eval.load_state_dict(state_dict, strict=True)

                if using_ddp and world_size > 1 and isinstance(test_sampler, DistributedSampler):
                    test_sampler.set_epoch(0)
                test_loss, test_acc = evaluate_on_test(to_eval, test_loader, device, arch)

                # Extra test sets (optional) — skipped in combined mode
                extra_accs = {}
                if (not args.combine_tests) and extra_tests:
                    for et_name, et_path in extra_tests:
                        et_ds   = datasets.ImageFolder(str(et_path), transform=tf_eval)
                        if using_ddp and world_size > 1:
                            et_sampler = DistributedSampler(et_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
                        else:
                            et_sampler = None
                        et_loader = DataLoader(et_ds, batch_size=bs_for_arch, shuffle=False, sampler=et_sampler,
                                               num_workers=args.num_workers, pin_memory=False, persistent_workers=False)
                        if using_ddp and world_size > 1 and isinstance(et_sampler, DistributedSampler):
                            et_sampler.set_epoch(0)
                        et_loss, et_acc = evaluate(to_eval, et_loader, ce, device, arch)
                        extra_accs[f"acc_{et_name}Test"] = float(et_acc)

                        shutdown_loader(et_loader)
                        del et_loader, et_ds

                # Update global CSV (rank 0)
                if is_main:
                    row_summary = {
                        "arch": arch,
                        "cfg_hash": cfg_hash,
                        "lr": float(lr),
                        "dropout": float(dp),
                        "weight_decay": float(weight_decay),
                        "label_smoothing": float(label_smoothing),
                        "batch_size": int(bs_for_arch),
                        "best_val_acc": float(best_val_acc),
                        "best_epoch": int(best_epoch),
                        "test_acc": float(test_acc),
                        "img_size": int(size),
                        "run_dir": str(run_dir),
                    }
                    if extra_accs:
                        row_summary.update(extra_accs)

                    df = pd.read_csv(global_csv)
                    df = pd.concat([df, pd.DataFrame([row_summary])], ignore_index=True)
                    df.to_csv(global_csv, index=False)
                    (run_dir / "complete.txt").write_text("done\n")

                if using_ddp:
                    dist.barrier()

                # ---- Hard cleanup between configs to avoid FD/pin-memory leaks ----
                shutdown_loader(train_loader)
                shutdown_loader(val_loader)
                shutdown_loader(test_loader)
                del train_loader, val_loader, test_loader
                del ds_train, ds_val, ds_test, full_train_tf
                # remove temp combined dir (rank 0)
                if combined_tmp_dir is not None and is_main:
                    shutil.rmtree(combined_tmp_dir, ignore_errors=True)
                gc.collect()
                torch.cuda.empty_cache()
                # ------------------------------------------------------------------

    if is_main:
        print("\nAll runs finished.")
        print(f"Global results at: {global_csv}")

    if using_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
