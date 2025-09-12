#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate single or ensemble (late-fusion) models on an ImageFolder dataset.
- Loads one or more run_dirs (each must contain params.json + best.pt)
- Runs inference per model with its own eval transform (299 for InceptionV3; 224 otherwise)
- Averages logits across models (optionally weights) and reports accuracy
- Can SEARCH best pairs/triples/greedy-K from a candidate pool (e.g., top_for_hybrids.json)
- Can enforce one-per-family constraint when searching (e.g., 1 ResNet, 1 DenseNet, 1 Inception)

Outputs:
- <out_dir>/ensemble_results.csv   (ranked results if searching; else a single row)
- <out_dir>/ensemble_top.md
- (optional) <out_dir>/preds.csv   (per-sample predictions)

Examples
--------
# 1) Evaluate a specific ensemble (no search)
python ensemble_infer.py \
  --proj_root ~/Alzheimers \
  --data_dir ~/Alzheimers/Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/test \
  --run_dirs "~/Alzheimers/Results/ArchSweep/ResNet50/8d028bd145,\
~/Alzheimers/Results/ArchSweep/DenseNet169/313768fef2,\
~/Alzheimers/Results/ArchSweep/InceptionV3/abcd123456" \
  --out_dir ~/Alzheimers/Results/EnsembleEval

# 2) Search best pairs from a pool (from top_for_hybrids.json) with one-per-family
python ensemble_infer.py \
  --proj_root ~/Alzheimers \
  --data_dir ~/Alzheimers/Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/test \
  --candidates_json ~/Alzheimers/Results/Model_Leaderboard/top_for_hybrids.json \
  --include_families "ResNet,DenseNet,Inception" \
  --limit_per_family 2 \
  --search pair \
  --one_per_family 1 \
  --out_dir ~/Alzheimers/Results/EnsembleEval

# 3) Search best triples from a manual pool (use run_dirs directly)
python ensemble_infer.py \
  --proj_root ~/Alzheimers \
  --data_dir ~/Alzheimers/Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/test \
  --run_dirs "/path/A,/path/B,/path/C,/path/D" \
  --search triple \
  --out_dir ~/Alzheimers/Results/EnsembleEval

# 4) Greedy K=3 from pool (no family constraint)
python ensemble_infer.py \
  --proj_root ~/Alzheimers \
  --data_dir ~/Alzheimers/Data/Kaggle_LukeChugh_Best_Alzheimers_MRI/test \
  --run_dirs "/path/A,/path/B,/path/C,/path/D" \
  --search greedy --k 3 \
  --out_dir ~/Alzheimers/Results/EnsembleEval
"""

import os, json, argparse, itertools, math
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

# ---------- Transforms ----------
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

def make_eval_tf(size: int):
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# ---------- Family mapping ----------
def family_of(arch: str) -> str:
    a = (arch or "").lower()
    if "resnext" in a: return "ResNeXt"
    if "resnet"  in a: return "ResNet"
    if "densenet" in a: return "DenseNet"
    if "inception" in a: return "Inception"
    if "efficientnet" in a or "effnet" in a: return "EffNet"
    if "mobilenetv3" in a: return "MobileNetV3"
    if "mobilenetv2" in a: return "MobileNetV2"
    if "vgg" in a: return "VGG"
    if "cnn_small" in a: return "CNN_Small"
    return "Other"

# ---------- Models (match your training builders) ----------
def _resnet(ctor, num):
    m = ctor(weights=None)  # we load our checkpoint next
    in_dim = m.fc.in_features
    m.fc = nn.Linear(in_dim, num)
    return m

def _densenet(ctor, num):
    m = ctor(weights=None)
    in_dim = m.classifier.in_features
    m.classifier = nn.Linear(in_dim, num)
    return m

def _effnet(ctor, num):
    m = ctor(weights=None)
    in_dim = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_dim, num)
    return m

def _mobilenet(ctor, num):
    m = ctor(weights=None)
    in_dim = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_dim, num)
    return m

def _vgg(ctor, num):
    m = ctor(weights=None)
    in_dim = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_dim, num)
    return m

def build_inception(num_classes):
    m = models.inception_v3(weights=None, aux_logits=True)
    in_dim = m.fc.in_features
    m.fc = nn.Linear(in_dim, num_classes)
    return m

class YourSmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x): return self.classifier(self.features(x).view(x.size(0), -1))

MODEL_REGISTRY_224 = {
    "CNN_Small":            lambda num: YourSmallCNN(num),
    "ResNet50":             lambda num: _resnet(models.resnet50, num),
    "ResNet101":            lambda num: _resnet(models.resnet101, num),
    "ResNet152":            lambda num: _resnet(models.resnet152, num),
    "DenseNet121":          lambda num: _densenet(models.densenet121, num),
    "DenseNet161":          lambda num: _densenet(models.densenet161, num),
    "DenseNet169":          lambda num: _densenet(models.densenet169, num),
    "DenseNet201":          lambda num: _densenet(models.densenet201, num),
    "EffNetB0":             lambda num: _effnet(models.efficientnet_b0, num),
    "MobileNetV2":          lambda num: _mobilenet(models.mobilenet_v2, num),
    "MobileNetV3_L":        lambda num: _mobilenet(models.mobilenet_v3_large, num),
    "ResNeXt50_32x4d":      lambda num: _resnet(models.resnext50_32x4d, num),
    "ResNeXt101_32x8d":     lambda num: _resnet(models.resnext101_32x8d, num),
    "VGG16":                lambda num: _vgg(models.vgg16_bn, num),
}

def build_model_from_params(params: dict, num_classes: int):
    arch = params["arch"]
    dp = float(params.get("dropout", 0.0) or 0.0)

    if arch == "InceptionV3":
        m = build_inception(num_classes)
    elif arch in MODEL_REGISTRY_224:
        m = MODEL_REGISTRY_224[arch](num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    # IMPORTANT: replicate training-time head structure for correct key names
    m = add_head_dropout_eval(m, arch, dp, num_classes)
    return m

def infer_logits_for_model(run_dir: Path, data_dir: Path, device: torch.device, batch_size=64, num_workers=4):
    """Return (logits: [N,C] np.float32, labels: [N] int64, info: dict)"""
    params = json.loads((run_dir / "params.json").read_text())
    arch = params["arch"]
    size = 299 if arch.lower().startswith("inception") else 224
    tf = make_eval_tf(size)

    # Base dataset to lock order
    base = datasets.ImageFolder(str(data_dir), transform=transforms.Compose([transforms.ToTensor()]))
    ds = datasets.ImageFolder(str(data_dir), transform=tf)

    # Assert identical file order & class mapping
    assert len(base.samples) == len(ds.samples), "Dataset size mismatch"
    assert base.class_to_idx == ds.class_to_idx, "Class mapping mismatch"
    for (p1, y1), (p2, y2) in zip(base.samples, ds.samples):
        if p1 != p2 or y1 != y2:
            raise RuntimeError("Sample ordering mismatch between base and tf datasets")

    labels = np.array([y for _, y in base.samples], dtype=np.int64)
    num_classes = len(base.classes)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, persistent_workers=(num_workers>0))

    model = build_model_from_params(params, num_classes).to(device)
    state = torch.load(run_dir / "best.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    all_logits = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            if arch.lower().startswith("inception") and isinstance(out, tuple):
                out = out[0]
            all_logits.append(out.detach().cpu().float())
    logits = torch.cat(all_logits, dim=0).numpy()
    info = {
        "arch": arch,
        "family": family_of(arch),
        "run_dir": str(run_dir),
        "cfg_hash": params.get("cfg_hash"),
        "img_size": size,
    }
    return logits.astype(np.float32), labels, info

def accuracy_from_logits(logits_sum: np.ndarray, labels: np.ndarray) -> float:
    preds = logits_sum.argmax(axis=1)
    return (preds == labels).mean().item()

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

# ---------- Search helpers ----------
def combos_k(items: List[int], k: int):
    return itertools.combinations(items, k)

def greedy_select(logits_bank: List[np.ndarray], labels: np.ndarray, k: int, one_per_family: bool, fams: List[str]):
    selected = []
    used_fams = set()
    current_sum = None
    for step in range(k):
        best_gain, best_idx, best_sum = -1, None, None
        for i, logit in enumerate(logits_bank):
            if i in selected: continue
            if one_per_family and fams[i] in used_fams: continue
            candidate_sum = logit if current_sum is None else (current_sum + logit)
            acc = accuracy_from_logits(candidate_sum, labels)
            gain = acc - (accuracy_from_logits(current_sum, labels) if current_sum is not None else 0.0)
            if acc > best_gain + (0 if current_sum is None else 0):  # favor higher final acc
                best_gain, best_idx, best_sum = acc, i, candidate_sum
        if best_idx is None:
            break
        selected.append(best_idx)
        current_sum = best_sum
        if one_per_family:
            used_fams.add(fams[best_idx])
    final_acc = accuracy_from_logits(current_sum, labels) if current_sum is not None else 0.0
    return selected, final_acc

def add_head_dropout_eval(model: nn.Module, arch: str, dp: float, num_classes: int):
    if dp is None or dp <= 0:
        return model
    # ResNet-style heads
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_dim = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dp), nn.Linear(in_dim, num_classes))
        return model
    # Classifier-style heads (DenseNet, MobileNet, VGG, EfficientNet)
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            # replace last Linear and ensure a Dropout right before it
            seq = list(model.classifier)
            last_lin_idx = None
            for i in reversed(range(len(seq))):
                if isinstance(seq[i], nn.Linear):
                    last_lin_idx = i
                    break
            if last_lin_idx is not None:
                in_dim = seq[last_lin_idx].in_features
                seq[last_lin_idx] = nn.Linear(in_dim, num_classes)
                if last_lin_idx == 0 or not isinstance(seq[last_lin_idx - 1], nn.Dropout):
                    seq.insert(last_lin_idx, nn.Dropout(dp))
                model.classifier = nn.Sequential(*seq)
                return model
        elif isinstance(model.classifier, nn.Linear):
            in_dim = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(dp), nn.Linear(in_dim, num_classes))
            return model
    return model

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True, help="ImageFolder root to evaluate on (e.g., Luke test)")
    ap.add_argument("--run_dirs", type=str, default="", help="Comma-separated list of run_dir paths (pool and/or fixed ensemble)")
    ap.add_argument("--candidates_json", type=str, default="", help="Optional JSON (e.g., top_for_hybrids.json)")
    ap.add_argument("--include_families", type=str, default="", help="Limit candidates to these families (comma-separated)")
    ap.add_argument("--limit_per_family", type=int, default=0, help="Cap number per family from candidates_json (0 = no cap)")
    ap.add_argument("--search", type=str, default="none", choices=["none","pair","triple","greedy"], help="Search mode")
    ap.add_argument("--k", type=int, default=3, help="K for greedy search")
    ap.add_argument("--one_per_family", type=int, default=1, help="Enforce at most one model per family (for search)")
    ap.add_argument("--weights", type=str, default="", help="Comma weights for --run_dirs in 'none' mode (default equal)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gpus", type=str, default="", help='Set CUDA_VISIBLE_DEVICES (e.g., "0" or "0,1"); empty uses default visibility')
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--save_preds", type=int, default=0, help="Write per-sample predictions CSV")
    args = ap.parse_args()

    if args.gpus != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proj_root = Path(os.path.expanduser(args.proj_root))
    data_dir = Path(os.path.expanduser(args.data_dir))
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build candidate list ----
    pool: List[Path] = []
    info_bank: List[Dict] = []
    logits_bank: List[np.ndarray] = []
    fams: List[str] = []
    labels_ref: np.ndarray = None

    # From JSON (shortlists)
    if args.candidates_json:
        j = json.loads(Path(os.path.expanduser(args.candidates_json)).read_text())
        include_fams = [f.strip() for f in args.include_families.split(",") if f.strip()] if args.include_families else None

        # Flatten family -> list[dict] (each with run_dir)
        flat = []
        for fam, lst in j.items():
            if include_fams and fam not in include_fams:
                continue
            limit = args.limit_per_family if args.limit_per_family > 0 else len(lst)
            for d in lst[:limit]:
                if "run_dir" in d and d["run_dir"]:
                    flat.append(Path(d["run_dir"]))

        pool.extend(flat)

    # From explicit --run_dirs
    if args.run_dirs.strip():
        extra = [Path(os.path.expanduser(p.strip())) for p in args.run_dirs.split(",") if p.strip()]
        pool.extend(extra)

    # De-dup while preserving order
    seen = set()
    uniq_pool = []
    for p in pool:
        if str(p) not in seen:
            uniq_pool.append(p)
            seen.add(str(p))
    pool = uniq_pool

    if not pool:
        raise SystemExit("No candidates provided (use --run_dirs and/or --candidates_json).")

    # ---- Run per-model inference to build logits bank ----
    for rd in pool:
        logits, labels, info = infer_logits_for_model(
            rd, data_dir, device, batch_size=args.batch_size, num_workers=args.num_workers
        )
        if labels_ref is None:
            labels_ref = labels
        else:
            if not np.array_equal(labels_ref, labels):
                raise RuntimeError("Label order mismatch across models; check dataset consistency.")
        logits_bank.append(logits)
        info_bank.append(info)
        fams.append(info["family"])
        print(f"[Loaded] {info['arch']} ({info['family']}): {rd}")

    # ---- If just evaluate a fixed list (no search) ----
    if args.search == "none":
        weights = None
        if args.weights.strip():
            w = [float(x) for x in args.weights.split(",")]
            if len(w) != len(logits_bank):
                raise SystemExit("--weights length must match number of --run_dirs in 'none' mode")
            weights = np.array(w, dtype=np.float32)
        if weights is None:
            logits_sum = np.sum(logits_bank, axis=0)
        else:
            # normalize to sum=1 for stability
            weights = weights / (weights.sum() + 1e-8)
            logits_sum = np.zeros_like(logits_bank[0])
            for w, L in zip(weights, logits_bank):
                logits_sum += w * L
        acc = accuracy_from_logits(logits_sum, labels_ref)

        import pandas as pd
        row = {
            "ensemble": " + ".join([info["arch"] for info in info_bank]),
            "n_models": len(info_bank),
            "acc": acc,
            "members": [info["run_dir"] for info in info_bank],
        }
        df = pd.DataFrame([row])
        (out_dir / "ensemble_results.csv").write_text(df.to_csv(index=False))
        try:
            (out_dir / "ensemble_top.md").write_text(df.to_markdown(index=False))
        except Exception:
            (out_dir / "ensemble_top.md").write_text(df.to_string(index=False))
        print(f"\nFixed ensemble accuracy: {acc:.4f}")
        if args.save_preds:
            preds = logits_sum.argmax(1)
            pd.DataFrame({"pred": preds, "label": labels_ref}).to_csv(out_dir / "preds.csv", index=False)
        return

    # ---- Search modes ----
    results = []  # list of (acc, indices_tuple)
    n = len(logits_bank)
    one_per_family = bool(args.one_per_family)

    def valid_combo(idx_tuple):
        if not one_per_family: return True
        famset = {fams[i] for i in idx_tuple}
        return len(famset) == len(idx_tuple)

    if args.search in ("pair", "triple"):
        k = 2 if args.search == "pair" else 3
        for idxs in combos_k(list(range(n)), k):
            if not valid_combo(idxs): continue
            logits_sum = np.zeros_like(logits_bank[0])
            for i in idxs:
                logits_sum += logits_bank[i]
            acc = accuracy_from_logits(logits_sum, labels_ref)
            results.append((acc, idxs))
    elif args.search == "greedy":
        sel, acc = greedy_select(logits_bank, labels_ref, k=args.k, one_per_family=one_per_family, fams=fams)
        results.append((acc, tuple(sel)))
    else:
        raise SystemExit("Unknown search mode")

    # Rank and save
    results.sort(key=lambda x: x[0], reverse=True)
    import pandas as pd
    rows = []
    for acc, idxs in results:
        members = [info_bank[i]["run_dir"] for i in idxs]
        names   = [info_bank[i]["arch"] for i in idxs]
        fam     = [info_bank[i]["family"] for i in idxs]
        rows.append({
            "acc": acc,
            "n_models": len(idxs),
            "members_names": " + ".join(names),
            "members_families": " + ".join(fam),
            "members_run_dirs": " | ".join(members),
            "indices": list(idxs),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ensemble_results.csv", index=False)
    try:
        (out_dir / "ensemble_top.md").write_text(df.head(30).to_markdown(index=False))
    except Exception:
        (out_dir / "ensemble_top.md").write_text(df.head(30).to_string(index=False))

    # Save preds for the top-1 combo if asked
    if args.save_preds and len(results) > 0:
        _, idxs = results[0]
        logits_sum = np.zeros_like(logits_bank[0])
        for i in idxs:
            logits_sum += logits_bank[i]
        preds = logits_sum.argmax(1)
        pd.DataFrame({"pred": preds, "label": labels_ref}).to_csv(out_dir / "preds.csv", index=False)

    print(f"\nSaved ranking to {out_dir/'ensemble_results.csv'}")
    if len(results) > 0:
        print("Top-1:", rows[0]["members_names"], f"acc={rows[0]['acc']:.4f}")

if __name__ == "__main__":
    main()

