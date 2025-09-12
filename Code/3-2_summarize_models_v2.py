#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize ArchSweep runs (and optional legacy hybrid runs) into a leaderboard + JSONs
for quick retraining and hybrid selection.

Outputs (configurable):
- <proj_root>/<leaderboard_dir>/leaderboard.csv         # ranked table
- <proj_root>/<leaderboard_dir>/leaderboard.md          # markdown preview
- <proj_root>/<leaderboard_dir>/top_per_arch.json       # best single config per architecture
- <proj_root>/<leaderboard_dir>/top_for_hybrids.json    # shortlists per family (for hybrid pairing)
- <proj_root>/<leaderboard_dir>/best_retrained.json     # arch -> {lr, dropout, label_smoothing, weight_decay, batch_size}
- (optional) <proj_root>/<leaderboard_dir>/leaderboard_wide.{csv,md}  # main + extra-test metrics per row

Notes:
- Use --leaderboard_dir to avoid overwriting your legacy leaderboard (default: Results/Model_Leaderboard).
- Concatenation skips empty / all-NA DataFrames to avoid pandas FutureWarnings.
- Pass --combine_extras_wide to emit a 'wide' table that merges extra-test scores into the same row.

Usage examples
--------------
# 1) Classic ArchSweep summary (legacy default output location)
python 3-2_summarize_models.py \
  --proj_root ~/Alzheimers \
  --sweep_dir Results/ArchSweep \
  --families "ResNet,DenseNet,Inception,ResNeXt,EffNet,MobileNetV2,MobileNetV3,VGG,CNN_Small,Other" \
  --per_family 3 \
  --top_overall 100

# 2) Singles-only summary to a SEPARATE leaderboard folder (skip shortlists)
python 3-2_summarize_models.py \
  --proj_root ~/Alzheimers \
  --sweep_dir Results/Singles_Luke \
  --per_family 0 \
  --top_overall 100 \
  --combine_extras_wide \
  --leaderboard_dir Results/Model_Leaderboard_Singles_Luke

# 3) More Singles sets
python 3-2_summarize_models.py \
  --proj_root ~/Alzheimers \
  --sweep_dir Results/Singles_Marco \
  --per_family 0 \
  --top_overall 100 \
  --combine_extras_wide \
  --leaderboard_dir Results/Model_Leaderboard_Singles_Marco

python 3-2_summarize_models.py \
  --proj_root ~/Alzheimers \
  --sweep_dir Results/Singles_Falah \
  --per_family 0 \
  --top_overall 100 \
  --combine_extras_wide \
  --leaderboard_dir Results/Model_Leaderboard_Singles_Falah

# 4) Mix Singles + legacy hybrid dirs (with optional HP JSONs) into a custom leaderboard dir
python 3-2_summarize_models.py \
  --proj_root ~/Alzheimers \
  --sweep_dir Results/ArchSweep \
  --hybrid_dirs "Results/Hybrid_Results_0826|hybrid_luke,Results/Hybrid_Results_0902|hybrid_luke+marco" \
  --hybrid_hp "hybrid_luke=/mnt/data/Hyperparameters_hybrid_for_Luke.json,hybrid_luke+marco=/mnt/data/Hyperparameters_hybrid_for_LukeMarco.json" \
  --families "ResNet,DenseNet,Inception,ResNeXt,EffNet,MobileNetV2,MobileNetV3,VGG,CNN_Small,Other" \
  --per_family 3 \
  --top_overall 100 \
  --leaderboard_dir Results/Model_Leaderboard_Custom
"""

import os, re, json, argparse
from pathlib import Path
import pandas as pd

# -------------------------
# Family mapping
# -------------------------
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

# -------------------------
# ArchSweep parser
# -------------------------
def parse_arch_sweep(sweep_dir: Path) -> pd.DataFrame:
    csv_path = sweep_dir / "arch_sweep_results.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)

    # ensure columns exist
    for col in ["arch","cfg_hash","lr","dropout","weight_decay","label_smoothing",
                "batch_size","best_val_acc","best_epoch","test_acc","img_size","run_dir"]:
        if col not in df.columns:
            df[col] = None

    # enrich with params.json (epochs, val_split, global_batch_size if present)
    rows = []
    for _, r in df.iterrows():
        run_dir = Path(str(r["run_dir"])) if pd.notna(r.get("run_dir")) else None
        params = {}
        if run_dir and (run_dir / "params.json").exists():
            try:
                params = json.loads((run_dir / "params.json").read_text())
            except Exception:
                params = {}
        row = dict(r)
        row["epochs"] = params.get("epochs")
        row["val_split"] = params.get("val_split")
        row["global_batch_size"] = params.get("global_batch_size", row.get("batch_size"))
        row["family"] = family_of(str(row["arch"]))
        row["source"] = "single_arch"
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["test_acc","best_val_acc"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    return out

# -------------------------
# Hybrid hyperparameter injection
# -------------------------
def _std_hparams_from_json(p: Path) -> dict:
    """Normalize keys from legacy hybrid JSONs to our leaderboard columns."""
    try:
        blob = json.loads(p.read_text())
    except Exception:
        return {}

    def pick_lr(d):
        for k in ["learning_rate", "initial_learning_rate", "lr"]:
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except Exception:
                    pass
        return None

    def pick_img_size(d):
        v = d.get("image_size")
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and len(v) > 0:
            try:
                return int(v[0])
            except Exception:
                return None
        try:
            return int(v)
        except Exception:
            return None

    out = {
        "lr": pick_lr(blob),
        "dropout": float(blob["dropout_rate"]) if "dropout_rate" in blob and blob["dropout_rate"] is not None else None,
        "batch_size": int(blob["batch_size"]) if "batch_size" in blob and blob["batch_size"] is not None else None,
        "epochs": int(blob["epochs"]) if "epochs" in blob and blob["epochs"] is not None else None,
        "img_size": pick_img_size(blob),
        "val_split": float(blob["validation_split"]) if "validation_split" in blob and blob["validation_split"] is not None else None,
        "num_workers": int(blob["num_workers"]) if "num_workers" in blob and blob["num_workers"] is not None else None,
        "weight_decay": float(blob["weight_decay"]) if "weight_decay" in blob and blob["weight_decay"] is not None else None,
        "label_smoothing": float(blob["label_smoothing"]) if "label_smoothing" in blob and blob["label_smoothing"] is not None else None,
    }
    return out

def parse_hybrid_hp_map(spec: str, proj_root: Path) -> dict:
    """spec: 'label=/abs/or/rel.json,label2=/path2.json'"""
    if not spec.strip():
        return {}
    out = {}
    items = [s.strip() for s in spec.split(",") if s.strip()]
    for it in items:
        if "=" not in it:
            continue
        label, path_str = it.split("=", 1)
        label = label.strip()
        p = Path(os.path.expanduser(path_str.strip()))
        if not p.is_absolute():
            p = (proj_root / p).resolve()
        if p.exists():
            out[label] = _std_hparams_from_json(p)
        else:
            out[label] = {}
    return out

# -------------------------
# Hybrid results parser (summary.txt)
# -------------------------
def parse_one_hybrid_dir(hybrid_dir: Path, label: str, hp_overrides_by_label: dict) -> pd.DataFrame:
    """Expect each subdir of hybrid_dir to have summary.txt with scalar metrics."""
    rows = []
    if not hybrid_dir.exists():
        return pd.DataFrame()

    hp = hp_overrides_by_label.get(label, {}) if label else {}

    for model_dir in sorted(hybrid_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        p = model_dir / "summary.txt"
        if not p.exists():
            continue
        text = p.read_text()
        def grab(name):
            m = re.search(rf"{name}\s*:\s*([0-9.]+)", text)
            return float(m.group(1)) if m else None
        row = {
            "arch": model_dir.name,
            "source": label or "hybrid_legacy",
            "test_acc": grab("Test Acc"),
            "precision": grab("Precision"),
            "recall": grab("Recall"),
            "f1": grab("F1-score"),
            "run_dir": str(model_dir),
            "family": "Hybrid",
        }
        for k, v in hp.items():
            if v is not None:
                if k == "img_size":
                    row["img_size"] = v
                else:
                    row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)

def parse_hybrid_dirs(hybrid_specs: str, proj_root: Path, hp_overrides_by_label: dict) -> pd.DataFrame:
    """
    Accepts a comma-separated list like:
      "Results/Hybrid_Results_0826|hybrid_luke,Results/Hybrid_Results_0902|hybrid_luke+marco"
    """
    if not hybrid_specs.strip():
        return pd.DataFrame()
    parts = [p.strip() for p in hybrid_specs.split(",") if p.strip()]
    frames = []
    for item in parts:
        if "|" in item:
            path_str, label = item.split("|", 1)
        else:
            path_str, label = item, "hybrid_legacy"
        p = Path(path_str)
        if not p.is_absolute():
            p = (proj_root / p).resolve()
        frames.append(parse_one_hybrid_dir(p, label, hp_overrides_by_label))
    if frames:
        cleaned = [f for f in frames if not f.empty and not f.isna().all().all()]
        return pd.concat(cleaned, ignore_index=True, sort=False) if cleaned else pd.DataFrame()
    return pd.DataFrame()

# -------------------------
# Extras (wide view) helpers
# -------------------------
def _parse_floats_from_text(text: str) -> dict:
    def grab(name):
        m = re.search(rf"{name}\s*:\s*([0-9.]+)", text)
        return float(m.group(1)) if m else None
    return {
        "test_acc": grab("Test Acc"),
        "precision": grab("Precision"),
        "recall": grab("Recall"),
        "f1": grab("F1-score"),
    }

def _read_extra_metrics_file(p: Path) -> dict:
    try:
        if p.suffix.lower() == ".json":
            blob = json.loads(p.read_text())
            return {
                "test_acc": float(blob.get("test_acc")) if blob.get("test_acc") is not None else None,
                "precision": float(blob.get("precision")) if blob.get("precision") is not None else None,
                "recall": float(blob.get("recall")) if blob.get("recall") is not None else None,
                "f1": float(blob.get("f1")) if blob.get("f1") is not None else None,
            }
        else:
            return _parse_floats_from_text(p.read_text())
    except Exception:
        return {}

def _label_from_path(p: Path) -> str:
    """
    Infer label from path (extras/Luke/summary.txt -> 'Luke',
    summary_Marco.txt -> 'Marco', extra_Falah_metrics.json -> 'Falah').
    """
    name = p.stem
    if p.parent.name and p.parent.parent and p.parent.parent.name == "extras":
        return p.parent.name
    m = re.search(r"summary[_\-]([A-Za-z0-9\-\+]+)$", name)
    if m:
        return m.group(1)
    m = re.search(r"extra[_\-]([A-Za-z0-9\-\+]+)", name)
    if m:
        return m.group(1)
    return "extra"

def collect_extra_tests(run_dir: str, patterns_csv: str) -> dict:
    """
    Returns {label: {'test_acc':..., 'precision':..., 'recall':..., 'f1':...}}
    Scans given glob patterns under run_dir.
    """
    out = {}
    if not run_dir:
        return out
    base = Path(run_dir)
    patterns = [s.strip() for s in patterns_csv.split(",") if s.strip()]
    for pat in patterns:
        for p in base.glob(pat):
            if not p.is_file():
                continue
            metrics = _read_extra_metrics_file(p)
            if not any(v is not None for v in metrics.values()):
                continue
            lab = _label_from_path(p)
            out[lab] = metrics
    return out

# -------------------------
# Helpers
# -------------------------
def pick_best_per_arch(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in ["test_acc","best_val_acc"] if c in df.columns]
    if cols:
        sdf = df.sort_values(cols, ascending=False).copy()
    else:
        sdf = df.copy()
    return sdf.drop_duplicates(subset=["arch"], keep="first").reset_index(drop=True)

def dataframe_to_markdown(df: pd.DataFrame, max_rows=30) -> str:
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, required=True)
    ap.add_argument("--sweep_dir", type=str, default="Results/ArchSweep")
    ap.add_argument("--hybrid_dirs", type=str, default="",
                    help="Comma list of PATH or PATH|LABEL, e.g., "
                         "'Results/Hybrid_Results_0826|hybrid_luke,Results/Hybrid_Results_0902|hybrid_luke+marco'")
    ap.add_argument("--hybrid_hp", type=str, default="",
                    help='Comma list of LABEL=JSONPATH providing hyperparameters for hybrid rows, '
                         'e.g., "hybrid_luke=/path/Luke.json,hybrid_luke+marco=/path/LukeMarco.json"')
    ap.add_argument("--families", type=str, default="ResNet,DenseNet,Inception",
                    help="Comma-separated families to shortlist for hybrids (e.g., add ResNeXt,EffNet,MobileNetV2,MobileNetV3,VGG)")
    ap.add_argument("--per_family", type=int, default=2, help="Top-N per family for hybrid shortlists (0 to skip shortlists)")
    ap.add_argument("--top_overall", type=int, default=50, help="Rows to keep in leaderboard display (0 = keep all)")
    ap.add_argument("--leaderboard_dir", type=str, default="Results/Model_Leaderboard",
                    help="Output directory for leaderboard files (CSV/MD and JSONs). Default: Results/Model_Leaderboard")
    ap.add_argument("--combine_extras_wide", action="store_true",
                    help="Create leaderboard_wide.{csv,md} that merges main and extra-test metrics for each model into one row.")
    ap.add_argument("--extras_patterns", type=str,
                    default="extras/*/summary.txt,summary_*.txt,extra_*_metrics.json",
                    help="Comma-separated glob patterns (relative to run_dir) to find extra-test metrics.")
    args = ap.parse_args()

    proj_root = Path(os.path.expanduser(args.proj_root))
    sweep_dir = proj_root / args.sweep_dir
    out_dir = (proj_root / args.leaderboard_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # parse hyperparam overrides for hybrid labels
    hp_overrides_by_label = parse_hybrid_hp_map(args.hybrid_hp, proj_root)

    singles = parse_arch_sweep(sweep_dir)
    hybrids = parse_hybrid_dirs(args.hybrid_dirs, proj_root, hp_overrides_by_label)

    # robust concat to avoid FutureWarning / dtype issues
    frames = [df for df in [singles, hybrids] if not df.empty and not df.isna().all().all()]
    all_df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    # Name for display
    if "cfg_hash" in all_df.columns:
        def mkname(r):
            if r.get("source") == "single_arch":
                return f"{r.get('arch','?')} ({str(r.get('cfg_hash',''))[:6]})"
            return f"{r.get('arch','?')} [{r.get('source','hybrid')}]"
        all_df["name"] = all_df.apply(mkname, axis=1)
    else:
        all_df["name"] = all_df.get("arch", "?")

    # Sort by test_acc desc then best_val_acc
    sort_cols = [c for c in ["test_acc","best_val_acc"] if c in all_df.columns]
    if sort_cols and not all_df.empty:
        all_df = all_df.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    # Leaderboard view
    leaderboard = all_df.copy()
    if args.top_overall and args.top_overall > 0 and not leaderboard.empty:
        leaderboard = leaderboard.head(args.top_overall).copy()

    # keep standard columns, + any per-dataset accuracy columns from 3-1 (acc_*)
    keep_cols = [c for c in [
        "name","source","family","arch","img_size","batch_size","global_batch_size",
        "lr","dropout","weight_decay","label_smoothing",
        "best_val_acc","best_epoch","test_acc","precision","recall","f1","run_dir","cfg_hash","epochs","val_split",
        "num_workers"
    ] if c in leaderboard.columns]
    extra_acc_cols = [c for c in leaderboard.columns if c.startswith("acc_")]
    keep_cols = keep_cols + extra_acc_cols

    if not leaderboard.empty:
        leaderboard = leaderboard[keep_cols]

    # Save leaderboard
    leaderboard_csv = out_dir / "leaderboard.csv"
    leaderboard_md  = out_dir / "leaderboard.md"
    leaderboard.to_csv(leaderboard_csv, index=False)
    leaderboard_md.write_text(dataframe_to_markdown(leaderboard))

    # Optional: "wide" leaderboard that merges extra-test metrics per model
    if args.combine_extras_wide and not leaderboard.empty:
        rows_wide = []
        for _, r in leaderboard.iterrows():
            base_row = {
                "name": r.get("name"),
                "arch": r.get("arch"),
                "family": r.get("family"),
                "source": r.get("source"),
                "cfg_hash": r.get("cfg_hash"),
                "run_dir": r.get("run_dir"),
                "img_size": r.get("img_size"),
                "batch_size": r.get("batch_size"),
                "global_batch_size": r.get("global_batch_size"),
                "lr": r.get("lr"),
                "dropout": r.get("dropout"),
                "weight_decay": r.get("weight_decay"),
                "label_smoothing": r.get("label_smoothing"),
                "test_acc__main": r.get("test_acc"),
                "precision__main": r.get("precision"),
                "recall__main": r.get("recall"),
                "f1__main": r.get("f1"),
            }
            # Try to read extras from files under run_dir
            extras = collect_extra_tests(r.get("run_dir"), args.extras_patterns)

            # Fallback: if no files found, use any acc_* columns on this row
            if not extras:
                for col in [c for c in leaderboard.columns if c.startswith("acc_")]:
                    lab = re.sub(r"^acc_", "", col)   # acc_MarcoTest -> MarcoTest
                    lab = re.sub(r"Test$", "", lab)   # MarcoTest -> Marco
                    base_row[f"test_acc__{lab}"] = r.get(col)
            else:
                for lab, m in sorted(extras.items()):
                    base_row[f"test_acc__{lab}"]  = m.get("test_acc")
                    base_row[f"precision__{lab}"] = m.get("precision")
                    base_row[f"recall__{lab}"]    = m.get("recall")
                    base_row[f"f1__{lab}"]        = m.get("f1")

            rows_wide.append(base_row)

        leaderboard_wide = pd.DataFrame(rows_wide)
        sort_cols_wide = ["test_acc__main"] + sorted(
            [c for c in leaderboard_wide.columns if c.startswith("test_acc__") and c != "test_acc__main"],
            reverse=True
        )
        leaderboard_wide = leaderboard_wide.sort_values(sort_cols_wide, ascending=False, na_position="last").reset_index(drop=True)
        (out_dir / "leaderboard_wide.csv").write_text(leaderboard_wide.to_csv(index=False))
        (out_dir / "leaderboard_wide.md").write_text(dataframe_to_markdown(leaderboard_wide, max_rows=50))

    # Best per arch (singles only)
    singles_only = all_df[all_df["source"]=="single_arch"].copy() if "source" in all_df.columns else all_df.copy()
    best_per_arch = pick_best_per_arch(singles_only)

    top_per_arch_json = {
        row["arch"]: {
            "cfg_hash": row.get("cfg_hash"),
            "run_dir": row.get("run_dir"),
            "lr": float(row.get("lr")) if pd.notna(row.get("lr")) else None,
            "dropout": float(row.get("dropout")) if pd.notna(row.get("dropout")) else None,
            "label_smoothing": float(row.get("label_smoothing")) if pd.notna(row.get("label_smoothing")) else None,
            "weight_decay": float(row.get("weight_decay")) if pd.notna(row.get("weight_decay")) else None,
            "batch_size": int(row.get("global_batch_size") if pd.notna(row.get("global_batch_size")) else row.get("batch_size")) if pd.notna(row.get("batch_size")) or pd.notna(row.get("global_batch_size")) else None,
            "best_val_acc": float(row.get("best_val_acc")) if pd.notna(row.get("best_val_acc")) else None,
            "test_acc": float(row.get("test_acc")) if pd.notna(row.get("test_acc")) else None,
        }
        for _, row in best_per_arch.iterrows()
    }
    (out_dir / "top_per_arch.json").write_text(json.dumps(top_per_arch_json, indent=2))

    # Shortlists per family to build hybrids (skip if per_family <= 0)
    wanted_families = [f.strip() for f in args.families.split(",") if f.strip()]
    shortlists = {}
    if args.per_family and args.per_family > 0 and wanted_families:
        for fam in wanted_families:
            fam_rows = singles_only[singles_only["family"] == fam].copy()
            if fam_rows.empty:
                continue
            fam_rows = fam_rows.sort_values(sort_cols, ascending=False).head(args.per_family)
            shortlists[fam] = [
                {
                    "arch": r["arch"],
                    "cfg_hash": r.get("cfg_hash"),
                    "run_dir": r.get("run_dir"),
                    "lr": float(r.get("lr")) if pd.notna(r.get("lr")) else None,
                    "dropout": float(r.get("dropout")) if pd.notna(r.get("dropout")) else None,
                    "label_smoothing": float(r.get("label_smoothing")) if pd.notna(r.get("label_smoothing")) else None,
                    "weight_decay": float(r.get("weight_decay")) if pd.notna(r.get("weight_decay")) else None,
                    "batch_size": int(r.get("global_batch_size") if pd.notna(r.get("global_batch_size")) else r.get("batch_size")) if pd.notna(r.get("batch_size")) or pd.notna(r.get("global_batch_size")) else None,
                    "best_val_acc": float(r.get("best_val_acc")) if pd.notna(r.get("best_val_acc")) else None,
                    "test_acc": float(r.get("test_acc")) if pd.notna(r.get("test_acc")) else None,
                }
                for _, r in fam_rows.iterrows()
            ]
    (out_dir / "top_for_hybrids.json").write_text(json.dumps(shortlists, indent=2))

    # Emit a best_retrained.json (schema your trainer expects)
    best_retrained = {}
    for arch, blob in top_per_arch_json.items():
        best_retrained[arch] = {
            "lr": blob.get("lr"),
            "label_smoothing": blob.get("label_smoothing"),
            "dropout": blob.get("dropout"),
            "weight_decay": blob.get("weight_decay"),
            "batch_size": blob.get("batch_size"),
        }
    (out_dir / "best_retrained.json").write_text(json.dumps(best_retrained, indent=2))

    print("Saved:")
    print(f"- {leaderboard_csv}")
    print(f"- {leaderboard_md}")
    if args.combine_extras_wide:
        print(f"- {out_dir / 'leaderboard_wide.csv'}")
        print(f"- {out_dir / 'leaderboard_wide.md'}")
    print(f"- {out_dir / 'top_per_arch.json'}")
    print(f"- {out_dir / 'top_for_hybrids.json'}")
    print(f"- {out_dir / 'best_retrained.json'}")

if __name__ == "__main__":
    main()
