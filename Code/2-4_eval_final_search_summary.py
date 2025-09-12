#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
final_eval.py — One-stop final evaluation + search summary for ResNet50

- Loads final_best_resnet50.pt (payload from retrain) OR a raw HPO best .pt state_dict
- Evaluates on a test ImageFolder
- Auto-detects head style (linear vs dropout->linear) from checkpoint keys
- Auto-infers dropout and eval batch size from checkpoint payload if present
- Writes to <checkpoint_dir>/reports/:
  * classification_report.txt
  * confusion_matrix.png
  * roc_curves_{perclass,micro,macro}.png
  * pr_curves_{perclass,micro,macro}.png
  * calibration_reliability.png
  * ece.txt
  * metrics.json (accuracy, ROC/PR AUCs, ECE)
  * (optional) hpo_top10.csv + hpo_summary.md from --hpo_csv
  * (optional) retrain_summary_sorted.csv from --retrain_csv
  * REPORT.md (high-level summary with links)
"""

import os
import csv
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score
)

import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as F, InterpolationMode

# --- Quiet specific, known deprecations cleanly (optional but requested) ---
warnings.filterwarnings(
    "ignore",
    message="`estimator_name` is deprecated",
    category=FutureWarning,
    module="sklearn",
)

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
IMAGENET_STD  = [0.229, 0.224, 0.225]

def make_eval_tf(size: int = 224):
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# ----------------------------
# Model builder (ResNet50) with head style handling
# ----------------------------
def build_resnet50_for_style(num_classes: int, head_style: str, dropout_p: float = 0.0):
    """
    head_style: "linear" (fc: Linear) or "seq" (fc: Dropout -> Linear)
    dropout_p is used only when head_style == "seq" (eval mode disables dropout anyway).
    """
    m = models.resnet50(weights=None)  # load weights from checkpoint
    in_dim = m.fc.in_features
    if head_style == "seq":
        dp = float(dropout_p) if dropout_p is not None else 0.3
        m.fc = nn.Sequential(nn.Dropout(p=dp), nn.Linear(in_dim, num_classes))
    else:
        m.fc = nn.Linear(in_dim, num_classes)
    return m

def detect_head_style_from_state_dict(sd: dict) -> str:
    """
    Returns "seq" if the checkpoint expects fc.1.* keys (Dropout->Linear),
    otherwise "linear" if it expects fc.weight/fc.bias.
    """
    has_seq = any(k.startswith("fc.1.") for k in sd.keys())
    has_lin = "fc.weight" in sd and "fc.bias" in sd
    if has_seq and not has_lin:
        return "seq"
    if has_lin and not has_seq:
        return "linear"
    # Ambiguous: default to linear
    return "linear"

# ----------------------------
# Prediction
# ----------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs_list, preds_list, labels_list = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        if isinstance(logits, tuple):  # future-proof (e.g., Inception)
            logits = logits[0]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        probs_list.append(probs)
        preds_list.append(preds)
        labels_list.append(yb.numpy())
    probs = np.concatenate(probs_list, axis=0)
    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return probs, preds, labels

# ----------------------------
# Plot helpers (sklearn name= shim)
# ----------------------------
def _roc_display(ax, fpr, tpr, auc_val, label):
    # name= (new) with fallback to estimator_name= (old)
    try:
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_val, name=label).plot(ax=ax)
    except TypeError:
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_val, estimator_name=label).plot(ax=ax)

def _pr_display(ax, precision, recall, ap, label):
    # name= (new) with fallback to estimator_name= (old)
    try:
        from sklearn.metrics import PrecisionRecallDisplay
        PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=ap, name=label).plot(ax=ax)
    except TypeError:
        from sklearn.metrics import PrecisionRecallDisplay
        PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=ap, estimator_name=label).plot(ax=ax)

# ----------------------------
# Plots & metrics
# ----------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_roc_curves(y_true, probs, class_names, out_dir):
    n_classes = len(class_names)
    y_bin = np.eye(n_classes)[y_true]

    # Per-class
    plt.figure(figsize=(8, 6))
    aucs = []
    ax = plt.gca()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr); aucs.append(roc_auc)
        _roc_display(ax, fpr, tpr, roc_auc, class_names[i])
    plt.title("ROC Curves (per-class)")
    plt.tight_layout(); plt.savefig(out_dir / "roc_curves_perclass.png", dpi=160); plt.close()

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.figure()
    _roc_display(plt.gca(), fpr_micro, tpr_micro, auc_micro, "micro-average")
    plt.title(f"ROC (micro-average, AUC={auc_micro:.4f})")
    plt.tight_layout(); plt.savefig(out_dir / "roc_curves_micro.png", dpi=160); plt.close()

    # Macro-average
    macro_auc = float(np.mean(aucs))
    plt.figure(figsize=(6, 5))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, alpha=0.2)
    plt.plot([0,1],[0,1],"k--",lw=1)
    plt.title(f"ROC (macro-average, AUC={macro_auc:.4f})")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.tight_layout(); plt.savefig(out_dir / "roc_curves_macro.png", dpi=160); plt.close()

    return {"auc_micro": float(auc_micro), "auc_macro": float(macro_auc)}

def plot_pr_curves(y_true, probs, class_names, out_dir):
    n_classes = len(class_names)
    y_bin = np.eye(n_classes)[y_true]

    # Per-class
    plt.figure(figsize=(8, 6))
    aps = []
    ax = plt.gca()
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap = average_precision_score(y_bin[:, i], probs[:, i]); aps.append(ap)
        _pr_display(ax, precision, recall, ap, class_names[i])
    plt.title("PR Curves (per-class)")
    plt.tight_layout(); plt.savefig(out_dir / "pr_curves_perclass.png", dpi=160); plt.close()

    # Micro-average
    precision_micro, recall_micro, _ = precision_recall_curve(y_bin.ravel(), probs.ravel())
    ap_micro = average_precision_score(y_bin.ravel(), probs.ravel())
    plt.figure()
    _pr_display(plt.gca(), precision_micro, recall_micro, ap_micro, "micro-average")
    plt.title(f"PR (micro-average, AP={ap_micro:.4f})")
    plt.tight_layout(); plt.savefig(out_dir / "pr_curves_micro.png", dpi=160); plt.close()

    # Macro (mean of per-class APs)
    ap_macro = float(np.mean(aps))
    return {"ap_micro": float(ap_micro), "ap_macro": float(ap_macro)}

def expected_calibration_error(probs, labels, n_bins: int = 15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        mask = (confidences > lo) & (confidences <= hi) if b>0 else (confidences >= lo) & (confidences <= hi)
        if not np.any(mask):
            bin_stats.append((lo, hi, 0, 0.0, 0.0)); continue
        conf_m = float(confidences[mask].mean())
        acc_m  = float(accuracies[mask].mean())
        gap = abs(acc_m - conf_m)
        ece += (mask.mean()) * gap
        bin_stats.append((lo, hi, int(mask.sum()), conf_m, acc_m))
    return float(ece), bin_stats

def plot_reliability(bin_stats, out_path):
    confs = [x[3] for x in bin_stats]
    accs  = [x[4] for x in bin_stats]
    centers = np.linspace(0, 1, len(confs), endpoint=False) + 0.5/len(confs)
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1],"k--",lw=1,label="perfect")
    plt.bar(centers, accs, width=1/len(confs), alpha=0.7, edgecolor="k", label="empirical acc")
    plt.scatter(centers, confs, s=20, label="mean conf", zorder=5)
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability Diagram")
    plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

# ----------------------------
# Safe CSV reading + numeric coercion (no deprecated args)
# ----------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path or not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        pass
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return pd.DataFrame()
    header = rows[0]; n = len(header)
    fixed = []
    for r in rows[1:]:
        if len(r) == n:
            fixed.append(r)
        elif len(r) > n:
            fixed.append(r[:n-1] + [",".join(r[n-1:])])
        else:
            fixed.append(r + [""]*(n-len(r)))
    return pd.DataFrame(fixed, columns=header)

def try_to_numeric(s: pd.Series) -> pd.Series:
    # Prefer strict conversion; if it fails, fall back to coerce (produces NaN for bad rows)
    try:
        return pd.to_numeric(s, errors="raise")
    except Exception:
        return pd.to_numeric(s, errors="coerce")

# ----------------------------
# HPO & retrain summaries
# ----------------------------
def summarize_hpo(hpo_csv: Path, out_dir: Path):
    df = safe_read_csv(hpo_csv)
    if df.empty:
        return None
    cols_pref = ["lr","weight_decay","dropout","label_smoothing","batch_size",
                 "best_val_acc","test_acc","best_epoch"]
    keep = [c for c in cols_pref if c in df.columns]
    # convert numeric-ish columns safely (avoids errors='ignore' deprecation)
    for c in keep:
        if c in df.columns:
            df[c] = try_to_numeric(df[c]) if df[c].dtype == object else df[c]
    # Rank & export
    sort_cols = [c for c in ["best_val_acc","test_acc"] if c in df.columns]
    top10 = df.sort_values(sort_cols, ascending=False)[keep].head(10).copy()
    top10.to_csv(out_dir / "hpo_top10.csv", index=False)
    md = [
        "# HPO Top-10 (by val, then test)",
        "",
        top10.to_markdown(index=False),
        "",
        f"Total trials in CSV: {len(df)}"
    ]
    (out_dir / "hpo_summary.md").write_text("\n".join(md))
    return top10

def summarize_retrain(retrain_csv: Path, out_dir: Path):
    df = safe_read_csv(retrain_csv)
    if df.empty:
        return None
    for c in ["val_mean","test_mean","val_std","test_std","n","lr","weight_decay","dropout","label_smoothing","batch_size"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = try_to_numeric(df[c])
    sort_cols = [c for c in ["val_mean","test_mean"] if c in df.columns]
    df_sorted = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    df_sorted.to_csv(out_dir / "retrain_summary_sorted.csv", index=False)
    return df_sorted

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, required=True, help="Project root, e.g., ~/Alzheimers")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to final_best_resnet50.pt OR HPO best .pt (relative to proj_root)")
    ap.add_argument("--data_dir", type=str, required=True, help="Test set dir (ImageFolder, relative to proj_root)")
    ap.add_argument("--hpo_csv", type=str, default="", help="Optional HPO CSV (relative to proj_root)")
    ap.add_argument("--retrain_csv", type=str, default="", help="Optional retrain summary CSV (relative to proj_root)")
    # Optional overrides (if omitted, infer where possible)
    ap.add_argument("--dropout", type=float, default=None, help="Override dropout p; head STYLE auto-inferred from checkpoint keys")
    ap.add_argument("--img_size", type=int, default=None, help="Override image size; default 224 for ResNet50")
    ap.add_argument("--batch_size", type=int, default=None, help="Override eval batch size; default 32 or inferred from checkpoint")
    args = ap.parse_args()

    proj_root = Path(os.path.expanduser(args.proj_root))
    ckpt_path = proj_root / args.checkpoint
    test_dir  = proj_root / args.data_dir
    hpo_csv   = proj_root / args.hpo_csv if args.hpo_csv else None
    retr_csv  = proj_root / args.retrain_csv if args.retrain_csv else None

    reports_dir = ckpt_path.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint & extract state_dict/params ---
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        ckpt_params = payload.get("params", {})
    else:
        state_dict = payload  # raw state_dict
        ckpt_params = {}

    # Determine head style; choose dropout value
    head_style = detect_head_style_from_state_dict(state_dict)
    inferred_dropout = float(ckpt_params.get("dropout")) if "dropout" in ckpt_params else None
    dropout_p = args.dropout if args.dropout is not None else (inferred_dropout if inferred_dropout is not None else (0.3 if head_style == "seq" else 0.0))

    # Infer batch size if present
    inferred_bs = int(ckpt_params.get("batch_size")) if "batch_size" in ckpt_params else None
    img_size = args.img_size if args.img_size is not None else 224
    eval_bs  = args.batch_size if args.batch_size is not None else (inferred_bs if inferred_bs is not None else 32)

    print(f"[final_eval] Head style: {head_style} | dropout_p={dropout_p} | img_size={img_size} | eval_batch_size={eval_bs}")

    # Dataset / loader
    tf_eval = make_eval_tf(img_size)
    test_set = datasets.ImageFolder(str(test_dir), transform=tf_eval)
    class_names = test_set.classes
    num_classes = len(class_names)
    loader = torch.utils.data.DataLoader(
        test_set, batch_size=eval_bs, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # Model + weights (style must match keys)
    model = build_resnet50_for_style(num_classes=num_classes, head_style=head_style, dropout_p=dropout_p).to(device)
    model.load_state_dict(state_dict)

    # Predict & metrics
    probs, y_pred, y_true = predict(model, loader, device)
    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    acc = accuracy_score(y_true, y_pred)

    (reports_dir / "classification_report.txt").write_text(report_txt + f"\n\nAccuracy: {acc:.6f}\n")
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump({"accuracy": float(acc)}, f, indent=2)

    plot_confusion_matrix(y_true, y_pred, class_names, reports_dir / "confusion_matrix.png")
    roc_stats = plot_roc_curves(y_true, probs, class_names, reports_dir)
    pr_stats  = plot_pr_curves(y_true, probs, class_names, reports_dir)

    ece, bin_stats = expected_calibration_error(probs, y_true, n_bins=15)
    plot_reliability(bin_stats, reports_dir / "calibration_reliability.png")
    (reports_dir / "ece.txt").write_text(f"ECE (15 bins): {ece:.6f}\n")

    with open(reports_dir / "metrics.json", "r") as f:
        m = json.load(f)
    m.update({"roc": roc_stats, "pr": pr_stats, "ece_15bins": float(ece)})
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)

    # Optional: HPO & retrain summaries
    top10 = summarize_hpo(hpo_csv, reports_dir) if hpo_csv else None
    retr  = summarize_retrain(retr_csv, reports_dir) if retr_csv else None

    # One-page report
    lines = [
        "# Final Evaluation Report (ResNet50)",
        "",
        f"- **Checkpoint:** `{ckpt_path}`",
        f"- **Test dir:** `{test_dir}`",
        f"- **Image size:** {img_size}",
        f"- **Eval batch size:** {eval_bs}",
        f"- **Head style:** `{head_style}`",
        f"- **Head dropout p (inactive in eval):** {dropout_p}",
        "",
        "## Overall Metrics",
        f"- **Accuracy:** `{acc:.4f}`",
        f"- **ROC AUC (micro):** `{m['roc']['auc_micro']:.4f}`  |  **ROC AUC (macro):** `{m['roc']['auc_macro']:.4f}`",
        f"- **PR AP (micro):** `{m['pr']['ap_micro']:.4f}`  |  **PR AP (macro):** `{m['pr']['ap_macro']:.4f}`",
        f"- **ECE (15 bins):** `{m['ece_15bins']:.4f}`",
        "",
        "## Plots",
        "- Confusion matrix: `confusion_matrix.png`",
        "- ROC curves: `roc_curves_perclass.png`, `roc_curves_micro.png`, `roc_curves_macro.png`",
        "- PR curves: `pr_curves_perclass.png`, `pr_curves_micro.png`, `pr_curves_macro.png`",
        "- Calibration: `calibration_reliability.png`",
        "",
    ]
    if top10 is not None:
        lines += [
            "## HPO Summary",
            f"- Source CSV: `{hpo_csv}`",
            "- Top-10 table: `hpo_top10.csv`",
            "",
            top10.to_markdown(index=False),
            ""
        ]
    if retr is not None:
        lines += [
            "## Retrain (mean ± std) Summary",
            f"- Source CSV: `{retr_csv}`",
            "- Sorted summary: `retrain_summary_sorted.csv`",
            "",
            retr.head(10).to_markdown(index=False),
            ""
        ]

    (reports_dir / "REPORT.md").write_text("\n".join(lines))

    print(f"\nSaved complete report package to: {reports_dir}")
    print(f"Test accuracy: {acc:.4f}")
    if hpo_csv:  print("Included HPO top-10 summary.")
    if retr_csv: print("Included retrain mean±std summary.")

if __name__ == "__main__":
    main()
