#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a folder of .pt weights (BestSingles) on a separate ImageFolder test set.

Defaults to STRICT loading for bit-for-bit identical accuracy versus the arch sweep.
If strict load fails (usually head-shape mismatch), the script reconstructs the
dropout-wrapped head to match training, moves it to GPU, and retries strict load.
You can allow partial loads explicitly with --allow_partial_load 1.

Example:
  # Marco test set → Results/BestSingles_Marco
  python 3-4_eval_bestsingles_on_test.py \
    --proj_root ~/Alzheimers \
    --weights_root Results/BestSingles \
    --test_dir Data/Kaggle_MarcoPinamonti_Alzheimers_MRI/test \
    --out_subdir Results/BestSingles_Marco \
    --sweep_csv Results/Singles_Luke/arch_sweep_results.csv
"""

import os, json, argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
import pandas as pd

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# -------------------
# Transforms
# -------------------
def make_eval_transform(arch_name: str):
    size = 299 if arch_name.lower().startswith("inception") else 224
    class PadToSquare:
        def __call__(self, img):
            w, h = F.get_image_size(img)
            s = max(w, h)
            pad_l = (s - w) // 2
            pad_r = s - w - pad_l
            pad_t = (s - h) // 2
            pad_b = s - h - pad_t
            return F.pad(img, [pad_l, pad_t, pad_r, pad_b], fill=0)
    tf_eval = transforms.Compose([
        PadToSquare(),
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf_eval, size

# -------------------
# Model builders
# -------------------
def base_build_model(arch: str, num_classes: int):
    """
    Build backbone with a plain Linear head (unless torchvision default is Sequential).
    Returns (model, head_attr_name).
    """
    a = arch.strip()
    if a == "InceptionV3":
        m = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
        in_dim = m.fc.in_features
        m.fc = nn.Linear(in_dim, num_classes)
        return m, "fc"

    if a == "ResNet50":
        m = models.resnet50(weights="IMAGENET1K_V1"); in_dim = m.fc.in_features; m.fc = nn.Linear(in_dim, num_classes); return m, "fc"
    if a == "ResNet101":
        m = models.resnet101(weights="IMAGENET1K_V1"); in_dim = m.fc.in_features; m.fc = nn.Linear(in_dim, num_classes); return m, "fc"
    if a == "ResNet152":
        m = models.resnet152(weights="IMAGENET1K_V1"); in_dim = m.fc.in_features; m.fc = nn.Linear(in_dim, num_classes); return m, "fc"

    if a == "DenseNet121":
        m = models.densenet121(weights="IMAGENET1K_V1"); in_dim = m.classifier.in_features; m.classifier = nn.Linear(in_dim, num_classes); return m, "classifier"
    if a == "DenseNet161":
        m = models.densenet161(weights="IMAGENET1K_V1"); in_dim = m.classifier.in_features; m.classifier = nn.Linear(in_dim, num_classes); return m, "classifier"
    if a == "DenseNet169":
        m = models.densenet169(weights="IMAGENET1K_V1"); in_dim = m.classifier.in_features; m.classifier = nn.Linear(in_dim, num_classes); return m, "classifier"
    if a == "DenseNet201":
        m = models.densenet201(weights="IMAGENET1K_V1"); in_dim = m.classifier.in_features; m.classifier = nn.Linear(in_dim, num_classes); return m, "classifier"

    if a == "EffNetB0":
        # torchvision default head is Sequential([Dropout, Linear])
        m = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_dim = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_dim, num_classes)
        return m, "classifier"

    if a == "MobileNetV2":
        m = models.mobilenet_v2(weights="IMAGENET1K_V1")
        in_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_dim, num_classes)
        return m, "classifier"

    if a == "MobileNetV3_L":
        m = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        in_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_dim, num_classes)
        return m, "classifier"

    if a == "ResNeXt50_32x4d":
        m = models.resnext50_32x4d(weights="IMAGENET1K_V1"); in_dim = m.fc.in_features; m.fc = nn.Linear(in_dim, num_classes); return m, "fc"
    if a == "ResNeXt101_32x8d":
        m = models.resnext101_32x8d(weights="IMAGENET1K_V1"); in_dim = m.fc.in_features; m.fc = nn.Linear(in_dim, num_classes); return m, "fc"

    if a == "VGG16":
        m = models.vgg16_bn(weights="IMAGENET1K_V1"); in_dim = m.classifier[-1].in_features; m.classifier[-1] = nn.Linear(in_dim, num_classes); return m, "classifier"

    if a == "CNN_Small":
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
                z = self.features(x); z = z.view(z.size(0), -1); return self.classifier(z)
        return YourSmallCNN(num_classes), "classifier"

    raise ValueError(f"Unknown arch: {arch}")

def add_head_dropout(model, head_name, p, num_classes):
    """Wrap final Linear into Sequential(Dropout(p), Linear) if not already."""
    if p is None or p <= 0:
        return model
    layer = getattr(model, head_name)
    # Plain Linear → wrap
    if isinstance(layer, nn.Linear):
        in_dim = layer.in_features
        setattr(model, head_name, nn.Sequential(nn.Dropout(p), nn.Linear(in_dim, num_classes)))
        return model
    # Sequential → ensure Dropout directly before last Linear
    if isinstance(layer, nn.Sequential):
        new_seq = list(layer)
        last_lin_idx = None
        for i in reversed(range(len(new_seq))):
            if isinstance(new_seq[i], nn.Linear):
                last_lin_idx = i
                break
        if last_lin_idx is not None:
            in_dim = new_seq[last_lin_idx].in_features
            new_seq[last_lin_idx] = nn.Linear(in_dim, num_classes)
            if last_lin_idx == 0 or not isinstance(new_seq[last_lin_idx - 1], nn.Dropout):
                new_seq.insert(last_lin_idx, nn.Dropout(p))
            setattr(model, head_name, nn.Sequential(*new_seq))
            return model
    return model

def state_uses_sequential_head(state_dict, head_name):
    """
    Heuristic: if saved keys look like 'fc.1.weight' or 'classifier.1.weight', the saved model had a Sequential head.
    """
    prefix = f"{head_name}."
    for k in state_dict.keys():
        if k.startswith(prefix + "1.weight") or k.startswith(prefix + "1.bias"):
            return True
    return False

# -------------------
# Utils
# -------------------
def infer_arch_and_params(pt_path: Path):
    """
    Prefer a params.json next to the weights. Returns (arch_name, params_dict_or_empty).
    """
    pjson = pt_path.with_name("params.json")
    if pjson.exists():
        try:
            blob = json.loads(pjson.read_text())
            if "arch" in blob and blob["arch"]:
                return str(blob["arch"]), blob
        except Exception:
            pass
    # Fallback: guess from filename
    name = pt_path.name
    candidates = [
        "InceptionV3","ResNeXt101_32x8d","ResNeXt50_32x4d","ResNet152","ResNet101","ResNet50",
        "DenseNet201","DenseNet169","DenseNet161","DenseNet121",
        "EffNetB0","MobileNetV3_L","MobileNetV2","VGG16","CNN_Small"
    ]
    for c in candidates:
        if c.lower().replace("_","") in name.lower().replace("_",""):
            return c, {}
        if c.lower() in name.lower():
            return c, {}
    return None, {}

def pick_dropout_for_arch(arch: str, params_blob: dict, sweep_csv_path: Path | None):
    """
    Decide dropout to reconstruct the head:
    1) params.json next to weights
    2) best arch row in sweep CSV (by test_acc)
    3) else None (if sequential head detected we'll use 0.3 as safe default)
    """
    if params_blob and ("dropout" in params_blob) and params_blob["dropout"] is not None:
        try:
            return float(params_blob["dropout"])
        except Exception:
            pass
    if sweep_csv_path and sweep_csv_path.exists():
        try:
            df = pd.read_csv(sweep_csv_path)
            if "arch" in df.columns:
                df_arch = df[df["arch"] == arch].copy()
                if not df_arch.empty:
                    sort_cols = [c for c in ["test_acc","best_val_acc"] if c in df_arch.columns]
                    if sort_cols:
                        df_arch = df_arch.sort_values(sort_cols, ascending=False)
                    dp = df_arch.iloc[0].get("dropout", None)
                    if pd.notna(dp):
                        return float(dp)
        except Exception:
            pass
    return None

@torch.no_grad()
def evaluate(model, loader, device, arch_name):
    model.eval()
    correct, total = 0, 0
    ce = nn.CrossEntropyLoss()
    loss_sum = 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb)
        if arch_name.lower().startswith("inception") and isinstance(out, tuple):
            out = out[0]
        loss = ce(out, yb)
        loss_sum += float(loss.item()) * yb.size(0)
        pred = out.argmax(1)
        correct += int((pred == yb).sum().item())
        total += int(yb.size(0))
    acc = correct / max(total, 1)
    return loss_sum / max(total, 1), acc

def try_strict_load(model, state_dict) -> bool:
    try:
        model.load_state_dict(state_dict, strict=True)
        return True
    except Exception as e:
        print(f"[ERROR] Strict load failed: {e}")
        return False

def partial_load_state(model, state_dict):
    model_keys = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_keys and model_keys[k].shape == v.shape}
    missing = set(model_keys.keys()) - set(filtered.keys())
    if missing:
        print(f"[WARN] Skipping {len(missing)} keys due to shape mismatch (likely classifier layer).")
    model.load_state_dict(filtered, strict=False)

# -------------------
# Main
# -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, required=True)
    ap.add_argument("--weights_root", type=str, default="Results/BestSingles")
    ap.add_argument("--pattern", type=str, default="*.pt", help="Glob for weight files (recurses).")
    ap.add_argument("--test_dir", type=str, required=True)
    ap.add_argument("--out_subdir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--sweep_csv", type=str, default="", help="Optional path to arch_sweep_results.csv to pick dropout per-arch.")
    ap.add_argument("--allow_partial_load", type=int, default=0,
                    help="If 1, fall back to shape-matched partial load when strict fails. Default 0 = strict only.")
    args = ap.parse_args()

    proj = Path(os.path.expanduser(args.proj_root)).resolve()
    wroot = (proj / args.weights_root).resolve()
    out_root = (proj / args.out_subdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sweep_csv_path = (proj / args.sweep_csv).resolve() if args.sweep_csv else None

    pts = sorted(list(wroot.rglob(args.pattern)))
    if not pts:
        print(f"No weights found under {wroot} matching {args.pattern}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset & classes
    test_root = (proj / args.test_dir).resolve()
    test_base = datasets.ImageFolder(str(test_root))
    classes = test_base.classes
    num_classes = len(classes)
    print("Classes:", classes)

    results = []

    for pt in pts:
        try:
            arch, params_blob = infer_arch_and_params(pt)
            if not arch:
                print(f"[SKIP] Could not infer architecture for {pt.name}")
                continue

            tf_eval, size = make_eval_transform(arch)
            test_ds = datasets.ImageFolder(str(test_root), transform=tf_eval)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)

            # Build model
            model, head_name = base_build_model(arch, num_classes)
            model = model.to(device)

            # Load state (unwrap if nested)
            state = torch.load(pt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            # Decide dropout for potential head reconstruction
            dp = pick_dropout_for_arch(arch, params_blob, sweep_csv_path)

            # If saved state expects Sequential head and model has Linear → add dropout head
            if isinstance(getattr(model, head_name), nn.Linear) and state_uses_sequential_head(state, head_name):
                if dp is None or dp <= 0:
                    dp = 0.3  # safe default
                model = add_head_dropout(model, head_name, dp, num_classes)
                model = model.to(device)  # ensure rebuilt head is on GPU

            # Try strict load first
            strict_loaded = try_strict_load(model, state)

            if not strict_loaded:
                if args.allow_partial_load:
                    partial_load_state(model, state)
                else:
                    # one last attempt: maybe head wasn't reconstructed; try now if Linear
                    if isinstance(getattr(model, head_name), nn.Linear):
                        if dp is None or dp <= 0:
                            dp = 0.3
                        model = add_head_dropout(model, head_name, dp, num_classes)
                        model = model.to(device)
                        strict_loaded = try_strict_load(model, state)
                    if not strict_loaded:
                        raise RuntimeError("Strict load failed and --allow_partial_load=0.")

            # Safety: ensure head params are on correct device
            head_mod = getattr(model, head_name)
            for p in head_mod.parameters():
                if p.device != device:
                    head_mod.to(device)
                    break

            # Evaluate
            test_loss, test_acc = evaluate(model, test_loader, device, arch)
            print(f"[{arch}] {pt.name}  acc={test_acc:.4f}  StrictLoad={strict_loaded}")

            # Save per-model short summary (optional but handy)
            out_dir = out_root / arch
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "summary.txt").write_text(
                f"Test Acc: {test_acc:.6f}\n"
                f"Loss: {test_loss:.6f}\n"
                f"Weights: {pt}\n"
                f"ImageSize: {size}\n"
                f"Head: {head_name} | dropout_in_head={'yes' if isinstance(head_mod, nn.Sequential) else 'no'} | dropout_p={dp if dp is not None else 'NA'}\n"
                f"StrictLoad: {strict_loaded}\n"
            )

            results.append({
                "arch": arch,
                "weights_path": str(pt),
                "test_dir": str(test_root),
                "img_size": size,
                "num_classes": num_classes,
                "test_acc": float(test_acc),
                "loss": float(test_loss),
                "out_dir": str(out_dir),
                "strict_loaded": bool(strict_loaded),
                "used_head_dropout": isinstance(head_mod, nn.Sequential),
                "dropout_p": dp if dp is not None else None,
            })
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[ERROR] {pt.name}: {e}")

    # Write aggregate CSV
    if results:
        df = pd.DataFrame(results).sort_values("test_acc", ascending=False)
        df.to_csv(out_root / "eval_results.csv", index=False)
        print("Saved:", out_root / "eval_results.csv")
    else:
        print("No successful evaluations.")

if __name__ == "__main__":
    main()

