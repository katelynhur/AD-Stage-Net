#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download Luke Chugh's dataset into ~/Alzheimers/Data/Kaggle_LukeChugh_Best_Alzheimers_MRI,
unzip, flatten, and rename class folders (spaces -> underscores).
"""

import os, sys, shutil, subprocess
from pathlib import Path

REPO = "lukechugh/best-alzheimer-mri-dataset-99-accuracy"
PROJ_ROOT = Path("~").expanduser() / "Alzheimers"
TARGET_DIRNAME = "Kaggle_LukeChugh_Best_Alzheimers_MRI"

def ensure_kaggle_cli():
    try:
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])

def ensure_kaggle_creds():
    kj = Path.home() / ".kaggle" / "kaggle.json"
    if not kj.exists() and not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        raise RuntimeError("Kaggle credentials not found.")
    if kj.exists():
        os.chmod(kj, 0o600)

def unpack_all_archives(root: Path):
    for pat in ("*.zip", "*.tar", "*.tar.gz", "*.tgz"):
        for arc in root.rglob(pat):
            shutil.unpack_archive(str(arc), str(arc.parent))
            arc.unlink(missing_ok=True)

def merge_move(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            merge_move(item, target)
        else:
            if target.exists():
                target.unlink()
            shutil.move(str(item), str(target))

def flatten_combined_dataset(out_dir: Path):
    combined = out_dir / "Combined Dataset"
    if combined.exists():
        for sub in ("train", "test"):
            src, dst = combined / sub, out_dir / sub
            if src.exists():
                merge_move(src, dst)
        shutil.rmtree(combined, ignore_errors=True)

def rename_class_dirs(out_dir: Path):
    for split in ("train", "test"):
        split_dir = out_dir / split
        if split_dir.exists():
            for d in split_dir.iterdir():
                if d.is_dir() and " " in d.name:
                    new_name = d.name.replace(" ", "_")
                    d.rename(d.parent / new_name)
                    print(f"Renamed {d.name} -> {new_name}")

def print_tiny_tree(root: Path):
    for p in sorted(root.glob("*")):
        print(" -", p.name)
        if p.is_dir():
            for q in sorted(p.glob("*")):
                print("   └─", q.name)

def main():
    ensure_kaggle_cli(); ensure_kaggle_creds()
    data_root = PROJ_ROOT / "Data"
    out_dir = data_root / TARGET_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.check_call([
        "kaggle", "datasets", "download",
        "-d", REPO, "-p", str(out_dir), "--unzip"
    ])

    unpack_all_archives(out_dir)
    flatten_combined_dataset(out_dir)
    rename_class_dirs(out_dir)
    print("\nFinal layout:"); print_tiny_tree(out_dir)

if __name__ == "__main__":
    main()

