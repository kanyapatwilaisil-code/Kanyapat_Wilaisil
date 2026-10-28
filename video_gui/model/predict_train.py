#!/usr/bin/env python3
"""
predict_train.py

Create train_predictions.csv for plot_train_results.py.

It scores the windows that train_augmented.py already saved:

    <run_output>/training_windows.npy   (N, 60, 22) float32 windows
    <run_output>/training_windows.csv   metadata, same row order as the .npy
    <run_output>/model/                 model.joblib (+ weights.pt, scaler.joblib, ...)

Output columns (required by plot_train_results.py):
    subject_id, source, label, prediction, score
Extra columns (ignored by the plot script):
    window_id, group_id, sample_id, threshold, split

Example:
    python predict_train.py \
      --run-dir /path/to/model/augmented_model \
      --pdml-root /path/to/parent_of_pdml \
      --output /path/to/train_predictions.csv

Then:
    python plot_train_results.py --train /path/to/train_predictions.csv --output /path/to/out
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def import_pdml(pdml_root):
    if pdml_root:
        p = str(Path(pdml_root).expanduser().resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
    # joblib needs pdml.models importable to unpickle the saved model.
    import pdml.models  # noqa: F401
    return pdml.models


def load_model(model_dir, models_module):
    """Prefer the full saved model (includes scaler + threshold).
    Fall back to weights.pt + scaler.joblib + model_info.json."""
    model_dir = Path(model_dir)

    full = model_dir / "model.joblib"
    if full.exists():
        print(f"Loading {full}")
        return joblib.load(full)

    weights = model_dir / "weights.pt"
    scaler = model_dir / "scaler.joblib"
    info = model_dir / "model_info.json"
    missing = [p.name for p in (weights, scaler, info) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"No model.joblib in {model_dir}, and fallback files are missing: {missing}.\n"
            "The scaler fitted on the training windows is required; weights.pt alone is not enough."
        )

    import torch
    from pdml.protocol import Protocol

    meta = json.loads(info.read_text())
    print(f"Rebuilding model from {weights.name} + {scaler.name}")
    model = models_module.Temporal(
        meta["model"], meta["feature_set"], meta["params"], meta["seed"], Protocol()
    )
    model.scaler = joblib.load(scaler)
    model.state = torch.load(weights, map_location="cpu")
    model.threshold = meta.get("threshold", 0.5)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="train_augmented.py --output directory")
    ap.add_argument("--model-dir", default=None,
                    help="Model folder (default: <run-dir>/model)")
    ap.add_argument("--windows-npy", default=None,
                    help="Default: <run-dir>/training_windows.npy")
    ap.add_argument("--windows-csv", default=None,
                    help="Default: <run-dir>/training_windows.csv")
    ap.add_argument("--split-json", default=None,
                    help="Default: <run-dir>/split.json (marks train vs validation)")
    ap.add_argument("--pdml-root", default=None,
                    help="Parent directory containing the pdml/ package")
    ap.add_argument("--train-only", action="store_true",
                    help="Same as --only train")
    ap.add_argument("--only", choices=["train", "validation"], default=None,
                    help="Keep only train windows or only the early-stopping validation windows")
    ap.add_argument("--output", required=True, help="Output train_predictions.csv")
    args = ap.parse_args()
    if args.train_only and not args.only:
        args.only = "train"

    run = Path(args.run_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else run / "model"
    npy = Path(args.windows_npy) if args.windows_npy else run / "training_windows.npy"
    csv = Path(args.windows_csv) if args.windows_csv else run / "training_windows.csv"
    split_json = Path(args.split_json) if args.split_json else run / "split.json"

    for p in (npy, csv):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    models_module = import_pdml(args.pdml_root)
    model = load_model(model_dir, models_module)

    x = np.load(npy)
    meta = pd.read_csv(csv)
    if len(x) != len(meta):
        raise ValueError(
            f"Row mismatch: {npy.name} has {len(x)} windows but {csv.name} has {len(meta)} rows."
        )
    print(f"Windows: {x.shape}  model: {model.name}/{model.feature_set}  threshold: {model.threshold}")

    scores = np.asarray(model.scores(x), dtype=float)
    threshold = float(model.threshold)

    out = pd.DataFrame({
        "subject_id": meta["subject_id"],
        "source": meta["source"],
        "label": meta["label"].astype(int),
        "prediction": (scores >= threshold).astype(int),
        "score": scores,
        "window_id": meta["window_id"],
        "group_id": meta["group_id"],
        "sample_id": meta["sample_id"],
        "threshold": threshold,
    })

    if split_json.exists():
        val_groups = set(json.loads(split_json.read_text()).get("validation_groups", []))
        out["split"] = np.where(out["group_id"].isin(val_groups), "validation", "train")
        print(out["split"].value_counts().to_string())
        if args.only:
            out = out[out["split"] == args.only].reset_index(drop=True)
            print(f"--only {args.only}: kept {len(out)} windows")
    elif args.only:
        raise FileNotFoundError(f"--only {args.only} needs {split_json}")

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
