#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, re, zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from PIL import Image

import pandas as pd
import gradio as gr

# Hugging Face Hub
from huggingface_hub import hf_hub_download, list_repo_files

# Optional: faster downloads on Spaces (aria2)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# ==============================
# Project text (with links)
# ==============================
PROJECT_MD = r"""
# 🧠 AD-Stage-Net: Four-Stage Alzheimer’s MRI Classification

**AD-Stage-Net** is an AI-driven project that utilizes deep learning to classify brain MRI scans into four stages of Alzheimer's disease severity: No Impairment, Very Mild Impairment, Mild Impairment, and Moderate Impairment. The model was trained on a combined and curated dataset sourced from Kaggle and Hugging Face, which helped to improve the diversity and generalizability of the training data. This project explores a range of convolutional neural networks (CNNs), including strong single backbones like ResNet and EfficientNet, and also employs hybrid models and ensembles to achieve high classification accuracy. The ultimate goal is to provide a robust and accessible AI tool for clinical decision support, allowing users to upload their own MRI scans and receive a real-time classification.

This demo runs strong single CNN backbones (and an optional 2-model ensemble) to classify brain MRI slices into four stages:
- **No Impairment**
- **Very Mild Impairment**
- **Mild Impairment**
- **Moderate Impairment**

**Code and more details:** <https://github.com/katelynhur/AD-Stage-Net>  
**Data sources (download to test yourself!):**  
- Kaggle ("Best Alzheimer's MRI Dataset 99% Accuracy" by Luke Chugh): <https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy>  
- Kaggle ("Alzheimer MRI 4 classes dataset" by Marco Pinamonti): <https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset>  
- HuggingFace ("Alzheimer_MRI" by Falah): <https://huggingface.co/datasets/Falah/Alzheimer_MRI>

> ⚠️ For research only; not a medical device.
"""

# ==============================
# Config
# ==============================
CLASS_NAMES = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
NUM_CLASSES = 4

MODEL_REPO = os.environ.get("MODEL_REPO", "katelynhur/AD-MRI-Classifier-Models")

LEADER_CSV    = Path("Results/Model_Leaderboard/leaderboard.csv")
PAIR_COMBINED = Path("Results/EnsembleEval/combined_ensemble_results.csv")
#PAIR_FALLBACK = Path("Results/EnsembleEval/Pairs_LukeTest_From_Luke/ensemble_results.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Warm cache (Hub)
# ==============================
def warm_cache_all_pt_files():
    try:
        files = list_repo_files(MODEL_REPO)
        pt_files = [f for f in files if f.lower().endswith(".pt")]
        for f in pt_files:
            try:
                hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=f,
                    local_dir="models_cache",
                    local_dir_use_symlinks=False,
                    local_files_only=False,
                )
                print(f"[warm] cached {f}")
            except Exception as e:
                print(f"[warm] skip {f}: {e}")
    except Exception as e:
        print(f"[warm] listing failed for {MODEL_REPO}: {e}")

warm_cache_all_pt_files()

# ==============================
# Transforms
# ==============================
def pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    pad_l = (s - w) // 2
    pad_r = s - w - pad_l
    pad_t = (s - h) // 2
    pad_b = s - h - pad_t
    return TF.pad(img, [pad_l, pad_t, pad_r, pad_b], fill=0)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def make_eval_tf(arch: str):
    size = 299 if arch.lower().startswith("inception") else 224
    return transforms.Compose([
        transforms.Lambda(lambda im: im.convert("L").convert("RGB")),  # grayscale -> 3ch
        transforms.Lambda(pad_to_square),
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# ==============================
# Model builders (match training)
# ==============================
def build_resnet(ctor, num):
    m = ctor(weights=None)
    in_dim = m.fc.in_features
    m.fc = nn.Linear(in_dim, num)
    return m

def build_densenet(ctor, num):
    m = ctor(weights=None)
    in_dim = m.classifier.in_features
    m.classifier = nn.Linear(in_dim, num)
    return m

def build_effnet(ctor, num):
    m = ctor(weights=None)
    in_dim = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_dim, num)
    return m

def build_mobilenet(ctor, num):
    m = ctor(weights=None)
    in_dim = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_dim, num)
    return m

def build_vgg(ctor, num):
    m = ctor(weights=None)
    in_dim = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_dim, num)
    return m

def build_inception(num_classes):
    m = models.inception_v3(weights=None, aux_logits=True)
    in_dim = m.fc.in_features
    m.fc = nn.Linear(in_dim, num_classes)
    return m

class _SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        z = self.features(x).view(x.size(0), -1)
        return self.classifier(z)

def _small_cnn(num_classes=NUM_CLASSES):
    return _SmallCNN(num_classes)

MODEL_BUILDERS: Dict[str, callable] = {
    "CNN_Small":            lambda num: _small_cnn(num_classes=num),
    "ResNet50":             lambda num: build_resnet(models.resnet50, num),
    "ResNet101":            lambda num: build_resnet(models.resnet101, num),
    "ResNet152":            lambda num: build_resnet(models.resnet152, num),
    "DenseNet121":          lambda num: build_densenet(models.densenet121, num),
    "DenseNet161":          lambda num: build_densenet(models.densenet161, num),
    "DenseNet169":          lambda num: build_densenet(models.densenet169, num),
    "DenseNet201":          lambda num: build_densenet(models.densenet201, num),
    "EffNetB0":             lambda num: build_effnet(models.efficientnet_b0, num),
    "MobileNetV2":          lambda num: build_mobilenet(models.mobilenet_v2, num),
    "MobileNetV3_L":        lambda num: build_mobilenet(models.mobilenet_v3_large, num),
    "ResNeXt50_32x4d":      lambda num: build_resnet(models.resnext50_32x4d, num),
    "ResNeXt101_32x8d":     lambda num: build_resnet(models.resnext101_32x8d, num),
    "VGG16":                lambda num: build_vgg(models.vgg16_bn, num),
    "InceptionV3":          lambda num: build_inception(num),
}

# ==============================
# Robust head adapter
# ==============================
def adapt_head_for_state_dict(model: nn.Module, arch: str, state: Dict, num_classes:int):
    # ResNet/ResNeXt
    has_fc1 = any(k.startswith("fc.1.weight") for k in state.keys())
    if hasattr(model, "fc"):
        if has_fc1 and not isinstance(model.fc, nn.Sequential):
            in_dim = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.0), nn.Linear(in_dim, num_classes))

    # DenseNet/MobileNet/VGG/EffNet
    has_clf1 = any(k.startswith("classifier.1.weight") for k in state.keys())
    if hasattr(model, "classifier"):
        if has_clf1 and not isinstance(model.classifier, nn.Sequential):
            if isinstance(model.classifier, nn.Linear):
                in_dim = model.classifier.in_features
            elif isinstance(model.classifier, nn.Sequential):
                in_dim = None
            else:
                in_dim = None
            if in_dim is not None:
                model.classifier = nn.Sequential(nn.Dropout(0.0), nn.Linear(in_dim, num_classes))
    return model

# ==============================
# Hub checkpoint discovery & loading
# ==============================
def arch_from_filename(fname: str) -> str:
    base = Path(fname).stem
    base = re.sub(r"_best$", "", base)
    return base

def list_available_checkpoints_from_hub(repo_id: str) -> Dict[str, str]:
    files = list_repo_files(repo_id)
    ckpts: Dict[str, str] = {}
    for f in files:
        if not f.lower().endswith(".pt"):
            continue
        name = Path(f).stem
        disp = re.sub(r"_best$", "", name)
        if disp in ckpts:
            if name.endswith("_best"):
                ckpts[disp] = f
        else:
            ckpts[disp] = f
    return ckpts

_LOADED: Dict[str, nn.Module] = {}
def load_model_from_hub(display_name: str, repo_filename: str) -> nn.Module:
    key = f"{MODEL_REPO}::{repo_filename}"
    if key in _LOADED:
        return _LOADED[key]

    local_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=repo_filename,
        local_dir="models_cache",
        local_dir_use_symlinks=False
    )

    arch = arch_from_filename(repo_filename)
    if arch not in MODEL_BUILDERS:
        raise RuntimeError(f"Unknown architecture inferred from filename: {arch}")

    model = MODEL_BUILDERS[arch](NUM_CLASSES).to(DEVICE)
    state = torch.load(local_path, map_location="cpu")
    model = adapt_head_for_state_dict(model, arch, state, NUM_CLASSES)
    model.load_state_dict(state, strict=True)
    model.eval()
    _LOADED[key] = model
    return model

# ==============================
# Inference helpers
# ==============================
@torch.no_grad()
def logits_for_model(model: nn.Module, arch: str, img: Image.Image) -> torch.Tensor:
    tfm = make_eval_tf(arch)
    xb = tfm(img).unsqueeze(0).to(DEVICE)
    out = model(xb)
    if arch.lower().startswith("inception") and isinstance(out, tuple):
        out = out[0]
    return out  # [1, C]

def ensemble_logits(logits_list: List[torch.Tensor]) -> torch.Tensor:
    return torch.mean(torch.stack(logits_list, dim=0), dim=0)

def _open_image(path_or_obj) -> Tuple[Image.Image, str]:
    """
    Accept a server path or an upload object; return (PIL Image, basename).
    Files are stored as paths in files_state, so we mainly handle strings.
    """
    if isinstance(path_or_obj, str):
        return Image.open(path_or_obj), Path(path_or_obj).name
    # Fallbacks
    if isinstance(path_or_obj, dict) and "name" in path_or_obj:
        return Image.open(path_or_obj["name"]), Path(path_or_obj["name"]).name
    if hasattr(path_or_obj, "name"):
        return Image.open(path_or_obj.name), Path(path_or_obj.name).name
    raise ValueError("Unsupported file object.")

def _round_df3(df: pd.DataFrame) -> pd.DataFrame:
    # round all float columns to 3 decimals
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(3)
    return df

def predict_files(
    files: List[str],
    model1_name: str,
    model2_name: str,
    ckpt_map_json: str,
    compact: bool
) -> pd.DataFrame:
    if not model1_name:
        raise gr.Error("Please select Model 1.")

    ckpt_map = json.loads(ckpt_map_json)

    repo_file1 = ckpt_map[model1_name]
    m1 = load_model_from_hub(model1_name, repo_file1)
    arch1 = arch_from_filename(repo_file1)

    m2, arch2 = None, None
    if model2_name:
        repo_file2 = ckpt_map[model2_name]
        m2 = load_model_from_hub(model2_name, repo_file2)
        arch2 = arch_from_filename(repo_file2)

    rows = []
    for f in files:
        try:
            img, base = _open_image(f)
        except Exception:
            continue

        L1 = logits_for_model(m1, arch1, img)
        L2 = logits_for_model(m2, arch2, img) if m2 else None

        P1 = F.softmax(L1, dim=1)[0].cpu().numpy()
        top1_idx = int(P1.argmax())
        top1 = CLASS_NAMES[top1_idx]
        
        # ✅ CHANGE 1: Format confidence for compact view
        conf1 = f"{P1[top1_idx]*100:.2f}%"

        row = {"filename": base, "Model 1": model1_name, "M1:top": top1, "M1:conf": conf1}

        if L2 is not None:
            P2 = F.softmax(L2, dim=1)[0].cpu().numpy()
            top2_idx = int(P2.argmax())
            top2 = CLASS_NAMES[top2_idx]
            
            # ✅ CHANGE 2: Format confidence for compact view
            conf2 = f"{P2[top2_idx]*100:.2f}%"

            L_ens = ensemble_logits([L1, L2])
            P_ens = F.softmax(L_ens, dim=1)[0].cpu().numpy()
            topE_idx = int(P_ens.argmax())
            topE = CLASS_NAMES[topE_idx]
            
            # ✅ CHANGE 3: Format confidence for compact view
            confE = f"{P_ens[topE_idx]*100:.2f}%"

            row.update({"Model 2": model2_name, "M2:top": top2, "M2:conf": conf2, "ENS:top": topE, "ENS:conf": confE})

            if not compact:
                for i, cls in enumerate(CLASS_NAMES):
                    # ✅ CHANGE 4: Format all probabilities as percentages for detailed view
                    row[f"M1:{cls}"]  = f"{float(P1[i])*100:.2f}%"
                    row[f"M2:{cls}"]  = f"{float(P2[i])*100:.2f}%"
                    row[f"ENS:{cls}"] = f"{float(P_ens[i])*100:.2f}%"
        else:
            if not compact:
                for i, cls in enumerate(CLASS_NAMES):
                    # ✅ CHANGE 5: Format all probabilities as percentages for detailed view
                    row[f"M1:{cls}"] = f"{float(P1[i])*100:.2f}%"

        rows.append(row)
    
    # NOTE: The _round_df3 function will no longer affect these columns since they are now strings.
    return _round_df3(pd.DataFrame(rows))
    
# ==============================
# Metrics tables (right panel)
# ==============================
def _round_selected_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(3)
    return df

# def load_best_singles_table(leader_csv: Path) -> pd.DataFrame:
#     if not leader_csv.exists():
#         return pd.DataFrame([{"info": "leaderboard.csv not found"}])
#     df = pd.read_csv(leader_csv)
#     if "source" in df.columns:
#         df = df[df["source"] == "single_arch"].copy()
#     sort_cols = [c for c in ["test_acc", "best_val_acc"] if c in df.columns]
#     if sort_cols:
#         df = df.sort_values(sort_cols, ascending=False)
#     if "arch" in df.columns:
#         df = df.drop_duplicates(subset=["arch"], keep="first")
#     # Exclude run_dir from display
#     keep = [c for c in ["arch","test_acc","best_val_acc"] if c in df.columns]
#     df = df[keep].reset_index(drop=True) if keep else df
#     return _round_selected_cols(df, ["test_acc","best_val_acc"])

def load_best_singles_table(leader_csv: Path) -> pd.DataFrame:
    if not leader_csv.exists():
        return pd.DataFrame([{"info": "leaderboard.csv not found"}])
    df = pd.read_csv(leader_csv)
    if "source" in df.columns:
        df = df[df["source"] == "single_arch"].copy()
    sort_cols = [c for c in ["test_acc", "best_val_acc"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)
    if "arch" in df.columns:
        df = df.drop_duplicates(subset=["arch"], keep="first")
    # Exclude run_dir from display
    keep = [c for c in ["arch","test_acc","best_val_acc"] if c in df.columns]
    df = df[keep].reset_index(drop=True) if keep else df

    # ✅ CHANGE: Convert accuracy columns to percentages
    if "test_acc" in df.columns:
        df["test_acc"] = (df["test_acc"] * 100).round(2).astype(str) + "%"
    if "best_val_acc" in df.columns:
        df["best_val_acc"] = (df["best_val_acc"] * 100).round(2).astype(str) + "%"

    return df

# def load_best_pairs_table() -> Optional[pd.DataFrame]:
#     if PAIR_COMBINED.exists():
#         df = pd.read_csv(PAIR_COMBINED)
#         show = [c for c in ["members_names","avg_acc","min_acc","acc_LukeTest","acc_MarcoTest","acc_FalahTest"] if c in df.columns]
#         df = df[show].head(20) if show else df.head(20)
#         return _round_selected_cols(df, ["avg_acc","min_acc","acc_LukeTest","acc_MarcoTest","acc_FalahTest"])
#     #if PAIR_FALLBACK.exists():
#     #    df = pd.read_csv(PAIR_FALLBACK)
#     #    show = [c for c in ["members_names","acc"] if c in df.columns]
#     #    df = df[show].head(20) if show else df.head(20)
#     #    return _round_selected_cols(df, ["acc"])
#     return None  # signal "not found"

def load_best_pairs_table() -> Optional[pd.DataFrame]:
    if PAIR_COMBINED.exists():
        df = pd.read_csv(PAIR_COMBINED)
        show = [c for c in ["members_names","avg_acc","min_acc","acc_LukeTest","acc_MarcoTest","acc_FalahTest"] if c in df.columns]
        df = df[show].head(20) if show else df.head(20)
        
        # ✅ CHANGE: Convert accuracy columns to percentages
        for col in ["avg_acc", "min_acc", "acc_LukeTest", "acc_MarcoTest", "acc_FalahTest"]:
            if col in df.columns:
                df[col] = (pd.to_numeric(df[col], errors="coerce") * 100).round(2).astype(str) + "%"

        return df
    return None

    
# ==============================
# Upload helpers (gallery + state)
# ==============================
def _normalize_upload_to_path(upload_obj) -> str:
    """
    Convert an UploadButton/Files item to a server-side path string.
    - For image files: use its temp file path (upload_obj.name).
    - For already-a-string: return it if it exists.
    """
    if hasattr(upload_obj, "name"):
        return upload_obj.name
    if isinstance(upload_obj, str):
        return upload_obj
    return None

def _gallery_from_paths(paths: List[str]):
    # Gallery expects [(image_path, caption), ...]
    return [(p, Path(p).name) for p in paths]

def _append_any(files_iterable, current_state):
    """
    Core appender used by both click-upload and drag-and-drop.
    Accept images and/or ZIPs; append to state; return:
      - updated state (list of server paths)
      - label text
      - gallery items
    """
    current = list(current_state or [])
    to_add_paths: List[str] = []

    for f in files_iterable or []:
        name = getattr(f, "name", None)
        # ZIP: extract supported images
        if name and name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(f) as zf:
                    for zi in zf.infolist():
                        if zi.is_dir():
                            continue
                        if not re.search(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", zi.filename, re.I):
                            continue
                        dest_root = Path("uploaded_cache")
                        dest_root.mkdir(parents=True, exist_ok=True)
                        zf.extract(zi, dest_root)
                        to_add_paths.append(str(dest_root / zi.filename))
            except Exception:
                pass
        else:
            p = _normalize_upload_to_path(f)
            if p:
                to_add_paths.append(p)

    # de-duplicate while preserving order
    new_state = list(dict.fromkeys(current + to_add_paths))
    label = f"{len(new_state)} file(s) selected"
    gallery_items = _gallery_from_paths(new_state)
    return new_state, label, gallery_items

def _handle_upload(files_list, current_state):
    # UploadButton handler
    return _append_any(files_list, current_state)

def _handle_drop(files_list, current_state):
    # Drag-and-drop (gr.Files) handler; also clear the Files input after append
    new_state, label, gallery_items = _append_any(files_list, current_state)
    clear_files_input = gr.update(value=None)  # clear the drop zone selection
    return new_state, label, gallery_items, clear_files_input

def clear_files():
    return [], "0 file(s) selected", []

# ==============================
# UI wiring
# ==============================
def update_model2_choices(model1, all_choices):
    if not model1:
        return gr.update(choices=all_choices, value=None, interactive=False)
    choices = [c for c in all_choices if c != model1]
    return gr.update(choices=choices, value=None, interactive=True)

def run_infer(files_list, model1, model2, ckpt_map_json, compact):
    if not files_list:
        raise gr.Error("Please upload at least one image.")
    return predict_files(files_list, model1, model2, ckpt_map_json, compact)

def _hero_css():
    # Light hero card + readable markdown + subtle border for drop zone
    return """
    .hero-card {
        background: #ffffff;
        color: #111 !important;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 8px;
    }
    .hero-card a { color: #2563eb !important; text-decoration: none; }
    .hero-card a:hover { text-decoration: underline; }
    .drop-zone {
        border: 2px dashed #cbd5e1;
        border-radius: 10px;
        padding: 10px;
        background: #f8fafc;
    }
    
    /* === Scrollable tables ===
       Hit multiple possible wrappers Gradio uses across versions */
    #singles-df .overflow-auto,
    #singles-df .wrap,
    #singles-df .table-wrap {
      max-height: 220px !important;
      overflow: hidden !important;
    }
    #singles-df [data-testid="dataframe"] {
      max-height: 220px !important;
      overflow-y: auto !important;
    }

    #pairs-df .overflow-auto,
    #pairs-df .wrap,
    #pairs-df .table-wrap {
      max-height: 220px !important;
      overflow: hidden !important;
    }
    #pairs-df [data-testid="dataframe"] {
      max-height: 220px !important;
      overflow-y: auto !important;
    }
    """

def build_app():
    # Discover checkpoints
    ckpt_map = list_available_checkpoints_from_hub(MODEL_REPO)  # display_name -> repo_filename
    model_choices = sorted(ckpt_map.keys())
    ckpt_map_json = json.dumps(ckpt_map)

    # Preload metrics tables
    singles_tbl = load_best_singles_table(LEADER_CSV)
    pairs_tbl   = load_best_pairs_table()

    css = _hero_css()

    with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
        # HERO (light card)
        with gr.Row():
            with gr.Column():
                gr.Markdown(value=PROJECT_MD, elem_classes=["hero-card"])

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    with gr.Column():
                        m1 = gr.Dropdown(
                            choices=model_choices,
                            label="Model 1 (required)",
                            value=None,
                            allow_custom_value=False,
                        )
                        m2 = gr.Dropdown(
                            choices=model_choices,
                            label="Model 2 (optional, ensemble)",
                            value=None,
                            allow_custom_value=False,
                            interactive=False,
                        )
                        compact = gr.Checkbox(
                            value=True,
                            label="Compact results (show only top label + confidence)"
                        )
                    with gr.Column():
                        # Multi-upload UX: button + drag-and-drop zone (both append)
                        files_state = gr.State([])

                        upload = gr.UploadButton(
                            "📁 Add images (PNG/JPG) or ZIP",
                            file_count="multiple",
                            file_types=["image","zip"]
                        )

                        files_label = gr.Markdown("0 file(s) selected")

                        # Drag & drop zone
                        files_dnd = gr.Files(
                            label="Or drag & drop images/ZIP here",
                            file_types=["image","zip"],
                            file_count="multiple",
                            elem_classes=["drop-zone"]
                        )

                        # Thumbnail preview
                        gallery = gr.Gallery(
                            label="Staged images",
                            show_label=True,
                            columns=6,
                            height="auto",
                            allow_preview=True
                        )

                        clear_btn = gr.Button("Clear files")

                # Pre-define empty dataframe with column headers
                empty_df = pd.DataFrame(columns=["Model 1", "Model 2", "Ensemble"])

                # Predictions
                results = gr.Dataframe(label="Predictions", value=empty_df, wrap=True, interactive=False, type="pandas")

                # Action
                run_btn = gr.Button("🔍 Run inference", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Best Single-Model Performance (`leaderboard.csv`)")
                singles_df = gr.Dataframe(value=singles_tbl, interactive=False, wrap=True, type="pandas", elem_id="singles-df", elem_classes=["df-singles"])

                gr.Markdown("### 🤝 Top Pairwise Ensembles")
                pairs_df   = gr.Dataframe(value=pairs_tbl if pairs_tbl is not None else pd.DataFrame(), interactive=False, wrap=True, type="pandas", visible=(pairs_tbl is not None), elem_id="pairs-df", elem_classes=["df-singles"])
                pairs_msg  = gr.Markdown(
                    value=("ℹ️ No ensemble results CSV found.\n\n"),
                    visible=(pairs_tbl is None)
                )

                #reload_btn = gr.Button("↻ Reload tables")

        # Events
        m1.change(fn=update_model2_choices, inputs=[m1, gr.State(model_choices)], outputs=m2)

        # Click-to-upload appends
        upload.upload(
            fn=_handle_upload,
            inputs=[upload, files_state],
            outputs=[files_state, files_label, gallery]
        )

        # Drag-and-drop appends, then clears its own selection
        files_dnd.upload(
            fn=_handle_drop,
            inputs=[files_dnd, files_state],
            outputs=[files_state, files_label, gallery, files_dnd]
        )

        clear_btn.click(
            fn=clear_files,
            inputs=None,
            outputs=[files_state, files_label, gallery]
        )

        run_btn.click(
            fn=run_infer,
            inputs=[files_state, m1, m2, gr.State(ckpt_map_json), compact],
            outputs=results
        )

        # Reload tables without restarting the app
        def _reload_tables():
            s = load_best_singles_table(LEADER_CSV)
            p = load_best_pairs_table()
            # Return dataframes and visibilities
            return (
                s,
                p if p is not None else pd.DataFrame(),
                gr.update(visible=(p is not None)),
                gr.update(visible=(p is None)),
            )

        # reload_btn.click(
            # fn=_reload_tables,
            # inputs=None,
            # outputs=[singles_df, pairs_df, pairs_df, pairs_msg]
        # )

    return demo

# ========= Entry =========
if __name__ == "__main__":
    demo = build_app()
    demo.launch()
