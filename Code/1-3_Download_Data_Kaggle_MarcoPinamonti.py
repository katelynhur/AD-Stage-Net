#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download Marco Pinamonti's dataset into ~/Alzheimers/Data/Kaggle_MarcoPinamonti_Alzheimers_MRI,
unzip, flatten "Alzheimer_MRI_4_classes_dataset", and rename class folders.
"""

import os, sys, shutil, subprocess
from pathlib import Path

REPO = "marcopinamonti/alzheimer-mri-4-classes-dataset"
PROJ_ROOT = Path("~").expanduser() / "Alzheimers"
TARGET_DIRNAME = "Kaggle_MarcoPinamonti_Alzheimers_MRI"

CLASS_MAP = {
    "MildDemented": "Mild_Impaired",
    "ModerateDemented": "Moderate_Impaired",
    "VeryMildDemented": "Very_Mild_Impaired",
    "NonDemented": "No_Impairment",
}

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

def flatten_and_rename(out_dir: Path):
    inner = out_dir / "Alzheimer_MRI_4_classes_dataset"
    if inner.exists() and inner.is_dir():
        for child in inner.iterdir():
            dst = out_dir / child.name
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.move(str(child), str(dst))
        shutil.rmtree(inner, ignore_errors=True)

    # Rename class dirs according to CLASS_MAP
    for d in out_dir.iterdir():
        if d.is_dir() and d.name in CLASS_MAP:
            new_name = CLASS_MAP[d.name]
            d.rename(d.parent / new_name)
            print(f"Renamed {d.name} -> {new_name}")

def print_tiny_tree(root: Path):
    for p in sorted(root.glob("*")):
        print(" -", p.name)

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
    flatten_and_rename(out_dir)
    print("\nFinal layout:"); print_tiny_tree(out_dir)

if __name__ == "__main__":
    main()

