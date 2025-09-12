#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download Falah/Alzheimer_MRI from Hugging Face and save as images
into ~/Alzheimers/Data/HuggingFace_Falah_Alzheimer_MRI.
Renames class folders so *_Demented → *_Impaired for consistency.
"""

from datasets import load_dataset, Image as HFImage
from PIL import Image as PILImage
from pathlib import Path
import os

# ------------------------------------------------------------------
# Project root (home-relative)
# ------------------------------------------------------------------
PROJ_ROOT = Path("~").expanduser() / "Alzheimers"
DATA_ROOT = PROJ_ROOT / "Data" / "HuggingFace_Falah_Alzheimer_MRI"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------
ds_dict = load_dataset("Falah/Alzheimer_MRI")

# ------------------------------------------------------------------
# Column detection helper
# ------------------------------------------------------------------
def detect_columns(ds):
    img_col = None
    lbl_col = None
    for col, feat in ds.features.items():
        if isinstance(feat, HFImage):
            img_col = col
        if hasattr(feat, "names"):   # ClassLabel
            lbl_col = col
    if img_col is None:
        for c in ["image", "img", "path", "file", "filepath"]:
            if c in ds.column_names:
                img_col = c; break
    if lbl_col is None:
        for c in ["label", "class", "target", "y"]:
            if c in ds.column_names:
                lbl_col = c; break
    return img_col, lbl_col

# ------------------------------------------------------------------
# Class folder rename map
# ------------------------------------------------------------------
CLASS_RENAME = {
    "Mild_Demented": "Mild_Impaired",
    "Moderate_Demented": "Moderate_Impaired",
    "Very_Mild_Demented": "Very_Mild_Impaired",
    "Non_Demented": "No_Impairment",
}

# ------------------------------------------------------------------
# Iterate splits and export
# ------------------------------------------------------------------
for split, ds in ds_dict.items():
    img_col, lbl_col = detect_columns(ds)
    if img_col is None or lbl_col is None:
        raise RuntimeError(f"Could not infer columns for split '{split}': {ds.column_names}")

    # Ensure decoding to PIL
    if not isinstance(ds.features[img_col], HFImage):
        ds = ds.cast_column(img_col, HFImage())

    # Extract class names if available
    class_names = ds.features[lbl_col].names if hasattr(ds.features[lbl_col], "names") else None

    split_dir = DATA_ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(ds):
        im = row[img_col]
        y  = row[lbl_col]
        cls = class_names[y] if class_names else str(y)

        # Apply rename map
        cls = CLASS_RENAME.get(cls, cls)

        out_dir = split_dir / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{i:07d}.jpg"
        out_path = out_dir / fname

        # Defensive: ensure PIL.Image
        if not isinstance(im, PILImage.Image):
            im = PILImage.fromarray(im)

        im.save(out_path, format="JPEG", quality=95)

    print(f"Saved split '{split}' to: {split_dir}")

