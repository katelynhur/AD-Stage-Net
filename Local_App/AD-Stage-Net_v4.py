#!/usr/bin/env python3
"""
AD-Stage-Net — Streamlit App
Tabs: Home | MRI Analysis | AI Assistant (Ollama)
"""

import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download, list_repo_files
import ollama

import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle

# ──────────────────────────────────────────────
# PAGE CONFIG 
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AD-Stage-Net Explorer",
    layout="wide",
)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
# Class names must match training label order exactly
CLASS_NAMES = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment",
]

# Display order for charts/tables (low → high severity)
CLASS_DISPLAY_ORDER = [
    "No Impairment",
    "Very Mild Impairment",
    "Mild Impairment",
    "Moderate Impairment",
]

CLASS_COLORS = {
    "No Impairment":        "#22c55e",  # green
    "Very Mild Impairment": "#3b82f6",  # blue
    "Mild Impairment":      "#f59e0b",  # amber
    "Moderate Impairment":  "#ef4444",  # red
}

MODEL_REPO = Path(r"H:\My Drive\Alzheimers\HuggingFaces\AD-MRI-Classifier-Models")
LEADER_CSV    = Path("Results/Model_Leaderboard/leaderboard.csv")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

OLLAMA_MODELS = [
    "llama3.2:3b",
    "gemma3n:e4b",
    # "mistral:7b",
]

AD_SYSTEM_PROMPT = """You are an expert AI assistant specializing exclusively in Alzheimer's Disease (AD).
You help patients, caregivers, students, and researchers understand AD clearly and compassionately.

Your areas of expertise include:
- The four stages of AD: No Impairment, Very Mild Impairment, Mild Impairment, Moderate Impairment
- Symptoms, progression, and early warning signs
- Brain changes visible on MRI (atrophy, hippocampal volume loss, etc.)
- Diagnosis methods (cognitive tests, MRI, PET, biomarkers)
- Treatment options and current clinical trials
- Caregiving strategies and coping resources
- Latest Alzheimer's research and breakthroughs

RULES:
- Only answer questions related to Alzheimer's Disease and dementia.
- If asked about unrelated topics, politely redirect: "I'm specialized in Alzheimer's Disease — feel free to ask me anything on that topic!"
- Never provide a personal medical diagnosis. Always recommend consulting a qualified healthcare provider.
- Be compassionate, clear, and avoid unnecessary medical jargon.
- Cite or acknowledge uncertainty when discussing cutting-edge or contested research.
"""

STAGE_CLINICAL_INFO = {
    "No Impairment": (
        "No observable cognitive decline. Brain structure on MRI appears within normal "
        "limits for the patient's age, with no significant atrophy or ventricular enlargement typically associated with AD."
    ),
    "Very Mild Impairment": (
        "Subtle memory lapses that may not yet be clinically apparent to others. MRI may show "
        "early, subtle structural changes, sometimes in the hippocampal region, though these can "
        "overlap with normal age-related changes."
    ),
    "Mild Impairment": (
        "Noticeable memory and cognitive difficulties that may affect daily functioning. MRI "
        "often shows visible cortical atrophy and more defined hippocampal volume loss compared "
        "to earlier stages."
    ),
    "Moderate Impairment": (
        "Significant decline in daily functioning and independence. MRI typically shows "
        "pronounced hippocampal atrophy, ventricular enlargement, and more widespread cortical thinning."
    ),
}

# ──────────────────────────────────────────────
# MODEL BUILDERS 
# ──────────────────────────────────────────────
class _SmallCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, n)

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


def _build_resnet(ctor, n):
    m = ctor(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n)
    return m

def _build_densenet(ctor, n):
    m = ctor(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, n)
    return m

def _build_effnet(ctor, n):
    m = ctor(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    return m

def _build_mobilenet(ctor, n):
    m = ctor(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
    return m

def _build_vgg(ctor, n):
    m = ctor(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
    return m

def _build_inception(n):
    m = models.inception_v3(weights=None, aux_logits=True)
    m.fc = nn.Linear(m.fc.in_features, n)
    return m

MODEL_BUILDERS = {
    "CNN_Small":        lambda n: _SmallCNN(n),
    "ResNet50":         lambda n: _build_resnet(models.resnet50, n),
    "ResNet101":        lambda n: _build_resnet(models.resnet101, n),
    "ResNet152":        lambda n: _build_resnet(models.resnet152, n),
    "DenseNet121":      lambda n: _build_densenet(models.densenet121, n),
    "DenseNet161":      lambda n: _build_densenet(models.densenet161, n),
    "DenseNet169":      lambda n: _build_densenet(models.densenet169, n),
    "DenseNet201":      lambda n: _build_densenet(models.densenet201, n),
    "EffNetB0":         lambda n: _build_effnet(models.efficientnet_b0, n),
    "MobileNetV2":      lambda n: _build_mobilenet(models.mobilenet_v2, n),
    "MobileNetV3_L":    lambda n: _build_mobilenet(models.mobilenet_v3_large, n),
    "ResNeXt50_32x4d":  lambda n: _build_resnet(models.resnext50_32x4d, n),
    "ResNeXt101_32x8d": lambda n: _build_resnet(models.resnext101_32x8d, n),
    "VGG16":            lambda n: _build_vgg(models.vgg16_bn, n),
    "InceptionV3":      lambda n: _build_inception(n),
}

# ──────────────────────────────────────────────
# TRANSFORMS
# ──────────────────────────────────────────────
def _pad_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    pl = (s - w) // 2
    pr = s - w - pl
    pt = (s - h) // 2
    pb = s - h - pt
    return TF.pad(img, [pl, pt, pr, pb], fill=0)

def make_transform(arch: str):
    size = 299 if arch.lower().startswith("inception") else 224
    return transforms.Compose([
        transforms.Lambda(lambda im: im.convert("L").convert("RGB")),
        transforms.Lambda(_pad_square),
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# ──────────────────────────────────────────────
# HEAD ADAPTER
# ──────────────────────────────────────────────
def _adapt_head(model: nn.Module, state: dict, n: int) -> nn.Module:
    if hasattr(model, "fc") and any(k.startswith("fc.1.") for k in state):
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.0), nn.Linear(in_f, n))
    if hasattr(model, "classifier") and any(k.startswith("classifier.1.") for k in state):
        if isinstance(model.classifier, nn.Linear):
            in_f = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(0.0), nn.Linear(in_f, n))
    return model

# ──────────────────────────────────────────────
# HF HUB HELPERS
# ──────────────────────────────────────────────

#   USING LOCAL NOW

def _list_checkpoints(model_dir: Path) -> dict:
    ckpts = {}
    for f in model_dir.glob("*.pt"):
        disp = re.sub(r"_best$", "", f.stem)
        if disp not in ckpts or f.stem.endswith("_best"):
            ckpts[disp] = f  # store full Path, not a repo filename
    return ckpts

def _load_one(model_path: Path) -> nn.Module:
    arch = re.sub(r"_best$", "", model_path.stem)
    if arch not in MODEL_BUILDERS:
        raise RuntimeError(f"Unknown architecture: {arch!r}")
    model = MODEL_BUILDERS[arch](len(CLASS_NAMES)).to(DEVICE)
    state = torch.load(model_path, map_location="cpu")
    model = _adapt_head(model, state, len(CLASS_NAMES))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# ──────────────────────────────────────────────
# CACHED RESOURCES  (load once per session)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_models() -> tuple[dict, dict]:
    """Returns (ckpt_map, loaded_models)."""
    ckpt_map = _list_checkpoints(MODEL_REPO)
    loaded = {}
    total = len(ckpt_map)
    progress = st.progress(0, text="Initialising models…")
    for i, (name, repo_file) in enumerate(ckpt_map.items()):
        progress.progress(i / total, text=f"Loading {name}…")
        try:
            loaded[name] = _load_one(repo_file)
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
        progress.progress((i + 1) / total, text=f"✔ {name}")
    progress.empty()
    return ckpt_map, loaded

# ──────────────────────────────────────────────
# LEADERBOARD DATA
# ──────────────────────────────────────────────
SINGLE_ARCH_LEADERBOARD = [
    ("CNN_Small",         46.06, 10.92, 39.45, 32.14, 10.92),
    ("DenseNet121",       98.36, 94.92, 98.67, 97.32, 94.92),
    ("DenseNet161",       98.91, 96.50, 98.91, 98.10, 96.50),
    ("DenseNet169",       98.67, 89.50, 97.66, 95.28, 89.50),
    ("DenseNet201",       98.83, 92.92, 98.98, 96.91, 92.92),
    ("EffNetB0",          96.88, 77.08, 94.45, 89.47, 77.08),
    ("InceptionV3",       96.64, 88.08, 97.58, 94.10, 88.08),
    ("MobileNetV2",       96.25, 55.83, 92.58, 81.55, 55.83),
    ("MobileNetV3_L",     98.13, 72.75, 90.00, 86.96, 72.75),
    ("ResNet101",         98.67, 88.00, 97.03, 94.57, 88.00),
    ("ResNet152",         97.81, 90.33, 98.44, 95.53, 90.33),
    ("ResNet50",          97.97, 90.58, 98.67, 95.74, 90.58),
    ("ResNeXt101_32x8d",  98.36, 88.83, 98.36, 95.18, 88.83),
    ("ResNeXt50_32x4d",   98.44, 84.33, 99.06, 93.94, 84.33),
    ("VGG16",             98.05, 91.33, 98.59, 95.99, 91.33),
]

ENSEMBLE_LEADERBOARD = [
    ("ResNet50 + DenseNet161",             99.14, 97.92, 99.06, 98.71, 97.92),
    ("DenseNet161 + ResNeXt101_32x8d",     99.22, 97.58, 99.14, 98.65, 97.58),
    ("DenseNet161 + VGG16",                99.14, 97.58, 99.22, 98.65, 97.58),
    ("ResNet50 + VGG16",                   99.14, 96.08, 99.22, 98.15, 96.08),
    ("ResNeXt101_32x8d + VGG16",           98.98, 95.92, 99.61, 98.17, 95.92),
    ("DenseNet161 + InceptionV3",          98.91, 95.58, 99.45, 97.98, 95.58),
    ("ResNet50 + ResNeXt101_32x8d",        98.51, 95.58, 99.45, 97.85, 95.58),
    ("DenseNet161 + EffNetB0",             98.75, 95.00, 98.91, 97.55, 95.00),
    ("InceptionV3 + ResNeXt101_32x8d",     97.81, 94.92, 99.30, 97.34, 94.92),
    ("InceptionV3 + VGG16",                97.81, 92.92, 98.98, 96.57, 92.92),
    ("DenseNet161 + MobileNetV3_L",        99.37, 92.42, 98.52, 96.77, 92.42),
]

_LEADERBOARD_COLS = ["Model", "acc_Luke", "acc_Marco", "acc_Falah", "avg_acc", "min_acc"]

@st.cache_data(show_spinner=False)
def load_leaderboard() -> "pd.DataFrame":
    df = pd.DataFrame(SINGLE_ARCH_LEADERBOARD, columns=_LEADERBOARD_COLS)
    df = df.sort_values("avg_acc", ascending=False).reset_index(drop=True)
    for col in ["acc_Luke", "acc_Marco", "acc_Falah", "avg_acc", "min_acc"]:
        df[col] = df[col].map(lambda v: f"{v:.2f}%")
    return df

@st.cache_data(show_spinner=False)
def load_ensemble_leaderboard() -> "pd.DataFrame":
    df = pd.DataFrame(ENSEMBLE_LEADERBOARD, columns=_LEADERBOARD_COLS)
    df = df.sort_values("avg_acc", ascending=False).reset_index(drop=True)
    for col in ["acc_Luke", "acc_Marco", "acc_Falah", "avg_acc", "min_acc"]:
        df[col] = df[col].map(lambda v: f"{v:.2f}%")
    return df


# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────
@torch.no_grad()
def _logits(model: nn.Module, arch: str, img: Image.Image) -> torch.Tensor:
    tfm = make_transform(arch)
    xb = tfm(img).unsqueeze(0).to(DEVICE)
    out = model(xb)
    if arch.lower().startswith("inception") and isinstance(out, tuple):
        out = out[0]
    return out  # [1, C]

def run_inference(img: Image.Image, loaded: dict, names: list) -> dict:
    results = {}
    logits_list = []
    for name in names:
        arch = re.sub(r"_best$", "", name)
        L = _logits(loaded[name], arch, img)
        logits_list.append(L)
        probs = F.softmax(L, dim=1)[0].cpu().tolist()
        top_i = int(torch.argmax(L, dim=1).item())
        results[name] = {
            "probs": probs,
            "top_class": CLASS_NAMES[top_i],
            "confidence": probs[top_i],
        }
    if len(logits_list) == 2:
        L_ens = torch.mean(torch.stack(logits_list), dim=0)
        probs  = F.softmax(L_ens, dim=1)[0].cpu().tolist()
        top_i  = int(torch.argmax(L_ens, dim=1).item())
        results["Ensemble"] = {
            "probs": probs,
            "top_class": CLASS_NAMES[top_i],
            "confidence": probs[top_i],
        }
    return results


def _get_target_layer(model: nn.Module, arch: str):
    a = arch.lower()
    if a.startswith("resnet") or a.startswith("resnext"):
        return [model.layer4[-1]]
    if a.startswith("densenet"):
        return [model.features[-1]]
    if a.startswith("vgg"):
        return [model.features[-1]]
    if a.startswith("mobilenet"):
        return [model.features[-1]]
    if a.startswith("effnet") or "efficientnet" in a:
        return [model.features[-1]]
    if a.startswith("inception"):
        return [model.Mixed_7c]
    if a == "cnn_small":
        return [model.features[6]]  # last Conv2d before the final ReLU/pool
    raise ValueError(f"No Grad-CAM target layer defined for architecture: {arch}")

def generate_gradcam(model: nn.Module, arch: str, img: Image.Image):
    """Returns (overlaid PIL image, raw grayscale CAM, normalized rgb array)."""
    tfm = make_transform(arch)
    input_tensor = tfm(img).unsqueeze(0).to(DEVICE)

    target_layers = _get_target_layer(model, arch)
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]  # HxW, values 0-1

    rgb_img = input_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization), grayscale_cam, rgb_img

def generate_ensemble_gradcam(cam_grayscale_maps: list, base_input_rgb: np.ndarray) -> Image.Image:
    """Averages multiple raw Grad-CAM heatmaps into one ensemble heatmap."""
    avg_cam = np.mean(np.stack(cam_grayscale_maps), axis=0)
    visualization = show_cam_on_image(base_input_rgb, avg_cam, use_rgb=True)
    return Image.fromarray(visualization)


# ──────────────────────────────────────────────
# UI HELPERS
# ──────────────────────────────────────────────
def _stage_badge(class_name: str, confidence: float):
    color = CLASS_COLORS.get(class_name, "#6b7280")
    st.markdown(
        f"""
        <div style="background:{color}22;border:2px solid {color};border-radius:12px;
                    padding:16px 20px;text-align:center;margin-bottom:8px;">
            <div style="font-size:0.8rem;color:#6b7280;margin-bottom:4px;">Predicted Stage</div>
            <div style="font-size:1.5rem;font-weight:700;color:{color};">{class_name}</div>
            <div style="font-size:1rem;color:#374151;margin-top:4px;">{confidence*100:.1f}% confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _confidence_chart(probs: list, title: str) -> go.Figure:
    classes = CLASS_DISPLAY_ORDER
    values  = [probs[CLASS_NAMES.index(c)] * 100 for c in classes]
    colors  = [CLASS_COLORS[c] for c in classes]
    fig = go.Figure(go.Bar(
        x=values, y=classes, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Confidence (%)", range=[0, 120]),
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def _analyze_agreement(results: dict) -> str:
    non_ensemble = {k: v for k, v in results.items() if k != "Ensemble"}
    if len(non_ensemble) < 2:
        return ""
    predictions = [v["top_class"] for v in non_ensemble.values()]
    if len(set(predictions)) == 1:
        return f"Both individual models agreed on the same predicted stage ({predictions[0]}), which increases confidence in this result."
    else:
        disagreement_detail = "; ".join(f"{k}: {v['top_class']}" for k, v in non_ensemble.items())
        return (
            f"The individual models disagreed on the predicted stage ({disagreement_detail}). "
            "This kind of disagreement between architectures suggests the case may be borderline "
            "or ambiguous, and results should be interpreted with extra caution."
        )


def _analyze_confidence(r: dict) -> str:
    sorted_probs = sorted(
        [(cls, r["probs"][CLASS_NAMES.index(cls)]) for cls in CLASS_DISPLAY_ORDER],
        key=lambda x: x[1], reverse=True
    )
    top_cls, top_p = sorted_probs[0]
    second_cls, second_p = sorted_probs[1]
    gap = (top_p - second_p) * 100
    if gap < 10:
        return f"This is a borderline call — {top_cls} ({top_p*100:.1f}%) was only {gap:.1f} points ahead of {second_cls} ({second_p*100:.1f}%)."
    elif gap < 25:
        return f"Moderate confidence — {top_cls} led {second_cls} by {gap:.1f} points."
    else:
        return f"High confidence — {top_cls} was clearly favored over the next closest class, {second_cls}, by {gap:.1f} points."


def build_chat_context(image_label: str, results: dict, models_to_run: list, gradcam_available: bool = False):
    """Returns (context_text, seed_message) for a new MRI-discussion chat session."""
    summary_lines = []
    for key, r in results.items():
        prob_str = ", ".join(
            f"{cls}: {r['probs'][CLASS_NAMES.index(cls)]*100:.1f}%"
            for cls in CLASS_DISPLAY_ORDER
        )
        confidence_note = _analyze_confidence(r)
        summary_lines.append(
            f"- {key}: predicted **{r['top_class']}** ({r['confidence']*100:.1f}% confidence). "
            f"Full breakdown — {prob_str}. {confidence_note}"
        )

    agreement_note = _analyze_agreement(results)

    top_class = list(results.values())[0]["top_class"]
    clinical_note = STAGE_CLINICAL_INFO.get(top_class, "")

    model_reliability_notes = []
    for m in models_to_run:
        match = next((row for row in SINGLE_ARCH_LEADERBOARD if row[0] == m), None)
        if match:
            model_reliability_notes.append(
                f"{m}: historically {match[3]:.1f}% average accuracy, "
                f"{match[4]:.1f}% minimum accuracy across external test datasets "
                "(minimum accuracy reflects worst-case reliability across different data sources)."
            )

    gradcam_note = (
        "\nGrad-CAM heatmaps were generated for this scan, highlighting the brain regions that "
        "most influenced each model's prediction — the user may ask about these."
        if gradcam_available else ""
    )

    context_text = f"The user just ran an MRI analysis on '{image_label}' in this app.\n\n"
    context_text += "PREDICTIONS:\n" + "\n".join(summary_lines) + "\n\n"
    if agreement_note:
        context_text += f"MODEL AGREEMENT: {agreement_note}\n\n"
    context_text += (
        f"CLINICAL CONTEXT FOR PREDICTED STAGE ({top_class}):\n{clinical_note}\n\n"
        + (("MODEL RELIABILITY:\n" + "\n".join(model_reliability_notes) + "\n\n") if model_reliability_notes else "")
        + gradcam_note
        + "\n\nUse this information to help the user understand what this predicted stage generally "
          "means, how confident/ambiguous the result is, and what patterns are typically associated "
          "with it — without giving a personal medical diagnosis. If the models disagreed or confidence "
          "was low, communicate that uncertainty clearly rather than overstating certainty."
    )

    seed = (
        f"I can see your results for {image_label} — "
        + "; ".join(f"{k}: {r['top_class']} ({r['confidence']*100:.1f}%)" for k, r in results.items())
        + ". What would you like to know about this?"
    )
    return context_text, seed


def build_pdf_report(img: Image.Image, results: dict, gradcam_images: dict | None = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AD-Stage-Net Explorer — Analysis Report", styles["Title"]))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(
        "<b>Disclaimer:</b> This is a research tool only, not a medical device. "
        "It is not intended to diagnose, treat, or replace professional medical advice.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 20))

    img_buffer = io.BytesIO()
    img.convert("RGB").save(img_buffer, format="PNG")
    img_buffer.seek(0)
    elements.append(Paragraph("Uploaded MRI Slice", styles["Heading2"]))
    elements.append(RLImage(img_buffer, width=3 * inch, height=3 * inch))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Classification Results", styles["Heading2"]))
    table_data = [["Model", "Predicted Stage", "Confidence"]]
    for key, r in results.items():
        table_data.append([key, r["top_class"], f"{r['confidence']*100:.1f}%"])
    t = Table(table_data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#374151")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Full Probability Breakdown", styles["Heading2"]))
    prob_data = [["Model"] + CLASS_DISPLAY_ORDER]
    for key, r in results.items():
        row = [key] + [f"{r['probs'][CLASS_NAMES.index(c)]*100:.1f}%" for c in CLASS_DISPLAY_ORDER]
        prob_data.append(row)
    pt = Table(prob_data, hAlign="LEFT")
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#374151")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(pt)
    elements.append(Spacer(1, 20))

    if gradcam_images:
        elements.append(Paragraph("Grad-CAM Explainability", styles["Heading2"]))
        for name, cam_img in gradcam_images.items():
            cam_buffer = io.BytesIO()
            cam_img.save(cam_buffer, format="PNG")
            cam_buffer.seek(0)
            elements.append(Paragraph(name, styles["Heading3"]))
            elements.append(RLImage(cam_buffer, width=3 * inch, height=3 * inch))
            elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

def build_pdf_report(img: Image.Image, results: dict, gradcam_images: dict | None = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AD-Stage-Net Explorer — Analysis Report", styles["Title"]))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(
        "<b>Disclaimer:</b> This is a research tool only, not a medical device. "
        "It is not intended to diagnose, treat, or replace professional medical advice.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 20))

    img_buffer = io.BytesIO()
    img.convert("RGB").save(img_buffer, format="PNG")
    img_buffer.seek(0)
    elements.append(Paragraph("Uploaded MRI Slice", styles["Heading2"]))
    elements.append(RLImage(img_buffer, width=3 * inch, height=3 * inch))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Classification Results", styles["Heading2"]))
    table_data = [["Model", "Predicted Stage", "Confidence"]]
    for key, r in results.items():
        table_data.append([key, r["top_class"], f"{r['confidence']*100:.1f}%"])
    t = Table(table_data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#374151")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Full Probability Breakdown", styles["Heading2"]))
    prob_data = [["Model"] + CLASS_DISPLAY_ORDER]
    for key, r in results.items():
        row = [key] + [f"{r['probs'][CLASS_NAMES.index(c)]*100:.1f}%" for c in CLASS_DISPLAY_ORDER]
        prob_data.append(row)
    pt = Table(prob_data, hAlign="LEFT")
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#374151")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(pt)
    elements.append(Spacer(1, 20))

    if gradcam_images:
        elements.append(Paragraph("Grad-CAM Explainability", styles["Heading2"]))
        for name, cam_img in gradcam_images.items():
            cam_buffer = io.BytesIO()
            cam_img.save(cam_buffer, format="PNG")
            cam_buffer.seek(0)
            elements.append(Paragraph(name, styles["Heading3"]))
            elements.append(RLImage(cam_buffer, width=3 * inch, height=3 * inch))
            elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# ──────────────────────────────────────────────
# BOOT — load models
# ──────────────────────────────────────────────
with st.spinner("Loading models…"):
    ckpt_map, loaded_models = load_all_models()

model_choices = sorted(loaded_models.keys())

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
import uuid

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

st.session_state.setdefault("mri_gradcam", {})

def new_chat_session(name=None, context=None, seed_message=None):
    """Create a new chat session, optionally pre-seeded with MRI result context."""
    sid = str(uuid.uuid4())[:8]
    st.session_state.chat_sessions[sid] = {
        "name": name or f"Chat {len(st.session_state.chat_sessions) + 1}",
        "messages": [],
        "context": context,  # extra text folded into the system prompt
    }
    if seed_message:
        st.session_state.chat_sessions[sid]["messages"].append(
            {"role": "assistant", "content": seed_message}
        )
    st.session_state.active_session_id = sid
    return sid

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    st.divider()

    st.subheader("MRI Analysis")
    selected_model_1 = st.selectbox("Primary Model", model_choices, index=0)
    use_ensemble = st.checkbox("Add a second model (ensemble)", value=False)
    selected_model_2 = None
    if use_ensemble:
        others = [m for m in model_choices if m != selected_model_1]
        selected_model_2 = st.selectbox("Secondary Model", others)

    st.divider()

    st.subheader("AI Assistant")
    ollama_model = st.selectbox("Ollama Model", OLLAMA_MODELS, index=0)
    temperature = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1)
    if st.button("Clear Chat"):
        if st.session_state.active_session_id:
            active_sess = st.session_state.chat_sessions.get(st.session_state.active_session_id)
            if active_sess:
                active_sess["messages"] = []
        st.rerun()

    st.divider()
    st.caption(f"AD-Stage-Net v1.0 · {len(loaded_models)} models loaded · For research use only")

# ──────────────────────────────────────────────
# MAIN TITLE
# ──────────────────────────────────────────────
st.title("AD-Stage-Net Explorer")
st.caption("An Interactive Alzheimer's Disease AI Classification and Education Platform")

tab_home, tab_mri, tab_chat = st.tabs(["Home", "MRI Analysis", "AI Assistant"])

# ══════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════
with tab_home:
    st.warning(
        "**Disclaimer:** AD-Stage-Net is a research tool only and is **not** a medical device. "
        "It is not intended to diagnose, treat, or replace professional medical advice. "
        "Always consult a qualified healthcare provider.",
        icon="⚠️",
    )

    st.header("About AD-Stage-Net Explorer")
    st.markdown("""
**AD-Stage-Net** is a deep learning system that classifies brain MRI scans into four stages of
Alzheimer's Disease severity. It was built to explore how convolutional neural networks (CNNs)
can support early detection research.

The models were trained on a single dataset from Kaggle (Luke) and tested on two additional datasets (Marco, Falah) to evaluate cross-dataset generalizability.
Upload your own MRI slices in the **MRI Analysis** tab
to receive a real-time classification, or ask the **AI Assistant** anything about Alzheimer's Disease.

**Links:** &nbsp;
[GitHub](https://github.com/katelynhur/AD-Stage-Net) &nbsp;|&nbsp;
[Hugging Face Space](https://huggingface.co/spaces/katelynhur/AD-Stage-Net)
    """)

    st.divider()
    st.header("What is Alzheimer's Disease?")
    st.markdown("""

Alzheimer’s disease is the most prevalent neurodegenerative disorder in the world, affecting over 55 million people. One of the challenges with Alzheimer’s is that by the time symptoms appear, significant brain damage has already occurred. Because of this, detecting the disease as early as possible is critical for intervention and treatment.
It is a progressive neurological disorder that gradually destroys memory, thinking skills, and the
ability to carry out everyday tasks.

**Why early detection matters:** Disease-modifying interventions are most effective in the earliest
stages. MRI-based classification tools like AD-Stage-Net aim to support earlier, data-driven
clinical conversations.
    """)

    st.subheader("The Four Stages Classified by This Model")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.success("🟢 **No Impairment**")
        # st.markdown("No observable cognitive decline. Brain structure appears normal on MRI.")
    with c2:
        st.info("🔵 **Very Mild Impairment**")
        # st.markdown("Subtle memory lapses. Slight structural changes may be detectable on MRI.")
    with c3:
        st.warning("🟡 **Mild Impairment**")
        # st.markdown("Noticeable memory and cognitive difficulties. Visible cortical atrophy on MRI.")
    with c4:
        st.error("🔴 **Moderate Impairment**")
        #st.markdown("Significant decline in daily functioning. Pronounced hippocampal atrophy.")

    st.divider()
    st.header("Model Performance")

    st.subheader("Single-Architecture Models")
    st.dataframe(load_leaderboard(), use_container_width=True, hide_index=True)

    st.subheader("Best Ensemble Combinations")
    st.dataframe(load_ensemble_leaderboard(), use_container_width=True, hide_index=True)

    st.divider()
    st.header("Datasets Used")
    st.markdown("""
| Source | Description |
|--------|-------------|
| [Kaggle — Luke Chugh](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy) | Best Alzheimer's MRI Dataset (99% Accuracy) |
| [Kaggle — Marco Pinamonti](https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset) | Alzheimer MRI 4-class dataset |
| [HuggingFace — Falah](https://huggingface.co/datasets/Falah/Alzheimer_MRI) | Alzheimer_MRI dataset |
    """)


# ══════════════════════════════════════════════
# TAB 2 — MRI ANALYSIS
# ══════════════════════════════════════════════
with tab_mri:
    st.header("MRI Analysis")
    st.markdown(
        "Upload one or more brain MRI slices — the selected model(s) will classify each into one of "
        "the four Alzheimer's Disease stages."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more MRI images (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        models_to_run = [selected_model_1]
        if selected_model_2:
            models_to_run.append(selected_model_2)

        st.markdown("**Selected model(s)**")
        for m in models_to_run:
            st.markdown(f"- `{m}`")
        if selected_model_2:
            st.markdown("- `Ensemble`")

        st.markdown(f"**{len(uploaded_files)} image(s) uploaded**")
        run_btn = st.button("🔍 Run Analysis on All Images", type="primary", use_container_width=True)

        if run_btn:
            missing = [m for m in models_to_run if m not in loaded_models]
            if missing:
                st.error(f"Model(s) not loaded: {missing}. Check your HF Hub connection.")
            else:
                batch_results = {}
                progress = st.progress(0, text="Running inference…")
                for i, uf in enumerate(uploaded_files):
                    im = Image.open(uf)
                    batch_results[uf.name] = {
                        "image": im,
                        "results": run_inference(im, loaded_models, models_to_run),
                    }
                    progress.progress((i + 1) / len(uploaded_files), text=f"Analyzed {uf.name}")
                progress.empty()
                st.session_state.mri_batch = batch_results
                st.session_state.mri_gradcam = {}

        if "mri_batch" in st.session_state and st.session_state.mri_batch:
            batch = st.session_state.mri_batch
            filenames = list(batch.keys())

            st.divider()
            st.subheader("Results")

            selected_image_name = st.selectbox("Select image to view in detail", filenames)
            entry = batch[selected_image_name]
            img = entry["image"]
            results = entry["results"]

            col_img, col_res = st.columns([1, 2])
            with col_img:
                st.image(img, caption=selected_image_name, use_container_width=True)

            with col_res:
                result_keys = list(results.keys())
                cols = st.columns(len(result_keys))
                for col, key in zip(cols, result_keys):
                    with col:
                        r = results[key]
                        label = "Ensemble" if key == "Ensemble" else f"Model: {key}"
                        st.markdown(f"**{label}**")
                        _stage_badge(r["top_class"], r["confidence"])
                        st.plotly_chart(
                            _confidence_chart(r["probs"], ""),
                            use_container_width=True,
                            key=f"chart_{selected_image_name}_{key}",
                        )

            with st.expander("📋 Full probability breakdown (this image)"):
                rows = []
                for key, r in results.items():
                    row = {"Model": key}
                    for cls in CLASS_DISPLAY_ORDER:
                        row[cls] = f"{r['probs'][CLASS_NAMES.index(cls)] * 100:.2f}%"
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with st.expander("📊 Summary across all uploaded images"):
                summary_rows = []
                for fname, e in batch.items():
                    for key, r in e["results"].items():
                        summary_rows.append({
                            "Image": fname,
                            "Model": key,
                            "Predicted Stage": r["top_class"],
                            "Confidence": f"{r['confidence']*100:.1f}%",
                        })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            st.divider()
            if st.button("Generate Grad-CAM Explainability (this image)"):
                with st.spinner("Generating Grad-CAM heatmaps…"):
                    cam_images = {}
                    raw_cams = []
                    ref_rgb = None
                    ref_shape = None
                    skip_ensemble = False
                    for name in models_to_run:
                        arch = re.sub(r"_best$", "", name)
                        try:
                            overlay, raw_cam, rgb_arr = generate_gradcam(loaded_models[name], arch, img)
                            cam_images[name] = overlay
                            if ref_shape is None:
                                ref_shape = raw_cam.shape
                                ref_rgb = rgb_arr
                            elif raw_cam.shape != ref_shape:
                                skip_ensemble = True
                            raw_cams.append(raw_cam)
                        except Exception as e:
                            st.warning(f"Could not generate Grad-CAM for {name}: {e}")
                    if len(raw_cams) == 2 and not skip_ensemble:
                        cam_images["Ensemble"] = generate_ensemble_gradcam(raw_cams, ref_rgb)
                    elif len(raw_cams) == 2 and skip_ensemble:
                        st.info("Ensemble Grad-CAM skipped — selected models use different input resolutions (e.g. InceptionV3).")
                    st.session_state.mri_gradcam[selected_image_name] = cam_images

            if selected_image_name in st.session_state.get("mri_gradcam", {}):
                st.subheader("Grad-CAM Explainability")
                st.caption("Highlighted regions show where each model focused most when making its prediction.")
                cam_cols = st.columns(len(st.session_state.mri_gradcam[selected_image_name]))
                for col, (name, cam_img) in zip(cam_cols, st.session_state.mri_gradcam[selected_image_name].items()):
                    with col:
                        st.image(cam_img, caption=name, use_container_width=True)

            st.divider()
            pdf_bytes = build_pdf_report(
                img, results, st.session_state.get("mri_gradcam", {}).get(selected_image_name)
            )
            st.download_button(
                "Export This Image's Report as PDF",
                data=pdf_bytes,
                file_name=f"AD-Stage-Net_{selected_image_name}_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            st.divider()
            if st.button("Discuss this image's results with the AI Assistant", type="primary"):
                context_text, seed = build_chat_context(
                    image_label=selected_image_name,
                    results=results,
                    models_to_run=models_to_run,
                    gradcam_available=selected_image_name in st.session_state.get("mri_gradcam", {}),
                )
                new_chat_session(
                    name=f"MRI: {list(results.values())[0]['top_class']}",
                    context=context_text,
                    seed_message=seed,
                )
                st.success("New chat created — click the **AI Assistant** tab above to continue.")

    else:
        st.info("Upload one or more MRI images above to get started.")


# ══════════════════════════════════════════════
# TAB 3 — AI ASSISTANT
# ══════════════════════════════════════════════
with tab_chat:
    st.header("AI Assistant")

    if not st.session_state.chat_sessions:
        new_chat_session(name="General Chat")

    # ── Session tab bar ──
    session_ids = list(st.session_state.chat_sessions.keys())
    tab_cols = st.columns(len(session_ids) + 1)
    for i, sid in enumerate(session_ids):
        sess = st.session_state.chat_sessions[sid]
        is_active = sid == st.session_state.active_session_id
        label = ("🟢 " if is_active else "") + sess["name"]
        if tab_cols[i].button(label, key=f"chattab_{sid}", use_container_width=True):
            st.session_state.active_session_id = sid
            st.rerun()
    if tab_cols[-1].button(" + New", use_container_width=True):
        new_chat_session()
        st.rerun()

    active = st.session_state.chat_sessions[st.session_state.active_session_id]

    with st.expander("Manage this chat"):
        renamed = st.text_input("Chat name", value=active["name"], key=f"rename_{st.session_state.active_session_id}")
        c1, c2 = st.columns(2)
        if c1.button("Save name"):
            active["name"] = renamed
            st.rerun()
        if c2.button("Delete chat", disabled=len(st.session_state.chat_sessions) <= 1):
            del st.session_state.chat_sessions[st.session_state.active_session_id]
            st.session_state.active_session_id = next(iter(st.session_state.chat_sessions))
            st.rerun()

    if active.get("context"):
        st.caption("This chat has MRI result context attached.")

    with st.container(height=450):
        for message in active["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_prompt = st.chat_input("Ask about Alzheimer's Disease...")

    if user_prompt:
        active["messages"].append({"role": "user", "content": user_prompt})

        system_content = AD_SYSTEM_PROMPT
        if active.get("context"):
            system_content += "\n\nContext for this conversation:\n" + active["context"]

        messages_for_model = [{"role": "system", "content": system_content}] + active["messages"]

        with st.spinner("Thinking..."):
            assistant_reply = ""
            try:
                response = ollama.chat(
                    model=ollama_model,
                    messages=messages_for_model,
                    options={"temperature": temperature},
                    stream=True,
                )
                for chunk in response:
                    assistant_reply += chunk["message"]["content"]
            except Exception as e:
                assistant_reply = f"Could not reach the local Ollama model.\n\nError: {e}"
                st.error(assistant_reply)

        active["messages"].append({"role": "assistant", "content": assistant_reply})
        st.rerun()
