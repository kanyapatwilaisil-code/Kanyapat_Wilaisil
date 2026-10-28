#!/usr/bin/env python3
"""
Evaluate the augmented TCN on a strictly held-out TEST set.

This script is deliberately conservative about splits:

- DNE split files with no top-level "test" are NOT guessed.
- CARE-PD files with `outer` folds require --outer-fold.
- Only the selected outer fold's `test` groups are evaluated.
- Side-view TEST uses only:
    side_pose/HC_v4/HC_test/**
    side_pose/PD_v3/PD_test/**
- Side-view TRAIN and UNTOUCHED are never loaded.
- Side group_id uses the same `SIDE::<subject>` namespace as training,
  so the leakage check can actually detect subject overlap.
- The exact preprocessing helpers from train_augmented.py are reused.
- The trained pdml model.joblib is loaded and scored; weights/scaler are
  therefore handled by pdml's Temporal.scores() implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_augmented as ta


# ---------------------------------------------------------------------
# Split handling
# ---------------------------------------------------------------------

def load_splits(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"splits.json not found: {p}")
    with p.open() as f:
        return json.load(f)


def get_outer_test_groups(splits: dict, outer_fold: int) -> set[str]:
    """
    Read ONLY:
        splits["outer"][outer_fold]["test"]

    We intentionally refuse to guess a top-level test split when the
    file uses the CARE-PD outer-fold structure.
    """
    outer = splits.get("outer")

    if not isinstance(outer, list) or not outer:
        raise KeyError(
            "splits.json does not contain a valid non-empty 'outer' list."
        )

    if outer_fold < 0 or outer_fold >= len(outer):
        raise IndexError(
            f"--outer-fold {outer_fold} is invalid; "
            f"available folds are 0..{len(outer)-1}."
        )

    fold = outer[outer_fold]

    if not isinstance(fold, dict):
        raise TypeError(
            f"splits.json['outer'][{outer_fold}] must be an object."
        )

    if "test" not in fold:
        raise KeyError(
            f"splits.json['outer'][{outer_fold}] has no 'test' key. "
            f"Keys: {list(fold.keys())}"
        )

    groups = fold["test"]

    if not isinstance(groups, list):
        raise TypeError(
            f"splits.json['outer'][{outer_fold}]['test'] must be a list."
        )

    groups = {str(x) for x in groups}

    if not groups:
        raise ValueError(
            f"Outer fold {outer_fold} has an empty TEST group list."
        )

    return groups


# ---------------------------------------------------------------------
# Side TEST
# ---------------------------------------------------------------------

def make_side_test_rows(side_root: str):
    """
    Load ONLY:

        <root>/HC_v3/HC_test/**/pose.npz
        <root>/PD_v3/PD_test/**/pose.npz

    Never loads TRAIN or UNTOUCHED.
    """
    root = Path(side_root).expanduser().resolve()

    test_dirs = [
        root / "HC_v3" / "HC_test",
        root / "PD_v3" / "PD_test",
    ]

    files = []

    for d in test_dirs:
        if not d.exists():
            raise FileNotFoundError(
                f"Missing side TEST directory: {d}"
            )
        files.extend(sorted(d.rglob("pose.npz")))

    if not files:
        raise RuntimeError(
            f"No side TEST pose.npz files found under {root}"
        )

    rows = []

    for path in files:
        label = ta.infer_side_label(path)
        subject = ta.infer_side_subject(path, label)

        seq, times, fps, source_video = ta.load_side_pose(path)

        # IMPORTANT:
        # Use the same group_id namespace as train_augmented.py:
        #     SIDE::<subject>
        # The cohort/source field distinguishes TEST from TRAIN.
        r = {
            "sample_id": f"SIDE_TEST::{subject}::{path.parent.name}",
            "group_id": f"SIDE::{subject}",
            "subject_id": subject,
            "walk_id": path.parent.name,
            "cohort": "SIDE_TEST",
            "label": int(label),
            "label_source": "folder",
            "raw_fps": fps,
            "effective_fps": fps,
            "frames": len(seq),
            "source_video": source_video,
            "pose_file": str(path),
        }

        rows.append((seq, times, fps, r, "SIDE_TEST"))

    return rows


# ---------------------------------------------------------------------
# DNE selection
# ---------------------------------------------------------------------

def filter_dne_test_rows(dne_rows, test_groups: set[str]):
    """
    Keep only DNE rows whose group_id is in the selected outer TEST fold.
    """
    selected = []

    for row in dne_rows:
        meta = row[3]
        group_id = str(meta.get("group_id"))

        if group_id in test_groups:
            selected.append(row)

    return selected


# ---------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------

def check_no_leakage(test_meta: pd.DataFrame, train_output: Path):
    """
    Check selected TEST groups against the groups recorded by the
    augmented training run.

    This catches both DNE and SIDE overlap, provided group_id namespaces
    are consistent with training.
    """
    split_path = train_output / "split.json"

    if not split_path.exists():
        raise FileNotFoundError(
            "Training split.json not found at "
            f"{split_path}. Refusing to continue without the "
            "recorded training-group leakage check."
        )

    with split_path.open() as f:
        train_split = json.load(f)

    train_groups = set(map(str, train_split.get("train_groups", [])))
    val_groups = set(map(str, train_split.get("validation_groups", [])))

    test_groups = set(map(str, test_meta["group_id"].unique()))

    overlap_train = sorted(test_groups & train_groups)
    overlap_val = sorted(test_groups & val_groups)

    if overlap_train:
        raise RuntimeError(
            "LEAKAGE: selected TEST groups overlap augmented TRAIN groups:\n"
            + "\n".join(overlap_train)
        )

    if overlap_val:
        raise RuntimeError(
            "LEAKAGE: selected TEST groups overlap augmented VALIDATION groups:\n"
            + "\n".join(overlap_val)
        )

    print("Leakage check: PASS")
    print(f"  TEST groups:        {len(test_groups)}")
    print(f"  overlap TRAIN:      {len(overlap_train)}")
    print(f"  overlap VALIDATION: {len(overlap_val)}")


def check_test_group_membership(meta: pd.DataFrame, dne_test_groups: set[str]):
    """
    Confirm every DNE TEST group represented in windows belongs to the
    selected outer fold and report any selected groups that produced
    zero windows.
    """
    dne_meta = meta[meta["source"] == "DNE"]

    if dne_meta.empty:
        print("WARNING: no DNE windows survived preprocessing.")
        return

    actual = set(dne_meta["group_id"].astype(str).unique())
    unexpected = sorted(actual - dne_test_groups)

    if unexpected:
        raise RuntimeError(
            "Unexpected DNE groups appeared in TEST windows:\n"
            + "\n".join(unexpected)
        )

    missing = sorted(dne_test_groups - actual)

    if missing:
        print(
            "WARNING: selected DNE TEST groups produced no accepted "
            "windows:"
        )
        for g in missing:
            print(" ", g)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n = len(y_true)

    accuracy = (tp + tn) / n if n else float("nan")

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn)
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else float("nan")
    )

    recalls = []
    if tp + fn:
        recalls.append(tp / (tp + fn))
    if tn + fp:
        recalls.append(tn / (tn + fp))

    balanced_accuracy = (
        float(np.mean(recalls))
        if len(recalls) == 2
        else float("nan")
    )

    return {
        "n": int(n),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": float(threshold),
    }


def print_metrics(title: str, meta: pd.DataFrame, scores: np.ndarray, threshold: float):
    y = meta["label"].to_numpy(dtype=int)
    m = binary_metrics(y, scores, threshold)

    print(f"\n=== {title} ===")
    print(f"Windows:            {m['n']}")
    print(f"Accuracy:           {m['accuracy']:.4f}")
    print(f"Balanced accuracy:  {m['balanced_accuracy']:.4f}")
    print(f"Sensitivity:        {m['sensitivity']:.4f}")
    print(f"Specificity:        {m['specificity']:.4f}")
    print(
        f"Confusion matrix:   "
        f"TN={m['tn']} FP={m['fp']} "
        f"FN={m['fn']} TP={m['tp']}"
    )

    return m


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate augmented TCN on a strictly held-out outer TEST fold."
    )

    ap.add_argument(
        "--dne-manifest",
        required=True,
        help="DNE skeleton_manifest.csv used for the augmented training run.",
    )

    ap.add_argument(
        "--dne-splits",
        required=True,
        help="CARE-PD splits.json containing outer folds.",
    )

    ap.add_argument(
        "--outer-fold",
        type=int,
        required=True,
        help="Outer fold index to evaluate, e.g. 0, 1, 2, 3, or 4.",
    )

    ap.add_argument(
        "--side-test-root",
        required=True,
        help="side_pose root containing HC_v4/HC_test and PD_v3/PD_test.",
    )

    ap.add_argument(
        "--model",
        required=True,
        help="Trained augmented model directory containing model.joblib.",
    )

    ap.add_argument(
        "--pdml-root",
        default=None,
        help="Parent directory containing the pdml/ package.",
    )

    ap.add_argument(
        "--feature-set",
        default="F3",
        choices=["F1", "F2", "F3"],
    )

    ap.add_argument(
        "--output",
        required=True,
        help="Directory for TEST artifacts.",
    )

    args = ap.parse_args()

    # ---------------------------------------------------------------
    # Load exact protocol and pdml helpers
    # ---------------------------------------------------------------

    Protocol, extract_windows, make_model, save_model = ta.import_pdml(
        args.pdml_root
    )

    cfg = Protocol.read()

    print("Protocol:")
    print(json.dumps(cfg.dict(), indent=2))

    # ---------------------------------------------------------------
    # Select ONE outer TEST fold
    # ---------------------------------------------------------------

    print("\n[1/7] Loading DNE outer split...")
    splits = load_splits(args.dne_splits)

    dne_test_groups = get_outer_test_groups(
        splits,
        args.outer_fold,
    )

    print(
        f"Selected outer fold: {args.outer_fold}"
    )
    print(
        f"DNE outer TEST groups: {len(dne_test_groups)}"
    )

    for g in sorted(dne_test_groups):
        print(" ", g)

    # ---------------------------------------------------------------
    # DNE TEST
    # ---------------------------------------------------------------

    print("\n[2/7] Loading DNE manifest...")
    dne_all = ta.load_dne_rows(args.dne_manifest)

    print(f"DNE manifest rows: {len(dne_all)}")

    dne_test = filter_dne_test_rows(
        dne_all,
        dne_test_groups,
    )

    if not dne_test:
        raise RuntimeError(
            "No DNE rows matched the selected outer TEST groups.\n"
            "This may mean the manifest is TRAIN-only and the selected "
            "outer fold belongs to that training pool. Do NOT continue "
            "until the split/protocol is confirmed."
        )

    print(f"DNE TEST rows: {len(dne_test)}")

    # ---------------------------------------------------------------
    # Side TEST
    # ---------------------------------------------------------------

    print("\n[3/7] Loading side-view TEST...")
    side_test = make_side_test_rows(args.side_test_root)

    print(f"Side TEST rows: {len(side_test)}")

    side_meta = pd.DataFrame([r[3] for r in side_test])

    print("Side TEST labels:")
    print(
        side_meta["label"]
        .map({0: "HC", 1: "PD"})
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Build windows using exact training pipeline
    # ---------------------------------------------------------------

    rows = dne_test + side_test

    print("\n[4/7] Building F3 TEST windows...")

    x, meta, coverage, rejected = ta.build_windows(
        rows,
        cfg,
        extract_windows,
    )

    if len(meta) == 0:
        raise RuntimeError("No accepted TEST windows were produced.")

    print(f"TEST windows: {len(meta)}")

    print("\nTEST labels:")
    print(
        meta["label"]
        .map({0: "HC", 1: "PD"})
        .value_counts()
        .to_string()
    )

    if "source" in meta.columns:
        print("\nTEST source:")
        print(meta["source"].value_counts().to_string())

    print(
        "\nTEST groups:",
        meta["group_id"].nunique(),
    )

    check_test_group_membership(
        meta,
        dne_test_groups,
    )

    # ---------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model).expanduser().resolve()
    train_output = model_dir.parent

    # ---------------------------------------------------------------
    # Leakage check
    # ---------------------------------------------------------------

    print("\n[5/7] Checking TEST leakage...")
    check_no_leakage(
        meta,
        train_output,
    )

    # ---------------------------------------------------------------
    # Save frozen TEST dataset
    # ---------------------------------------------------------------

    np.save(out / "test_windows.npy", x)
    meta.to_csv(out / "test_windows.csv", index=False)
    coverage.to_csv(out / "test_coverage.csv", index=False)
    rejected.to_csv(out / "test_window_rejections.csv", index=False)

    (out / "test_groups.json").write_text(
        json.dumps(
            {
                "outer_fold": args.outer_fold,
                "dne_test_groups": sorted(dne_test_groups),
                "side_test_groups": sorted(
                    side_meta["group_id"]
                    .astype(str)
                    .unique()
                    .tolist()
                ),
            },
            indent=2,
        )
    )

    (out / "test_protocol.json").write_text(
        json.dumps(cfg.dict(), indent=2)
    )

    # ---------------------------------------------------------------
    # Load trained model
    # ---------------------------------------------------------------

    print("\n[6/7] Loading trained model...")

    model_path = model_dir / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"model.joblib not found: {model_path}"
        )

    import joblib

    model = joblib.load(model_path)

    if not hasattr(model, "scores"):
        raise TypeError(
            f"Loaded object from {model_path} has no .scores() method."
        )

    model_feature_set = getattr(model, "feature_set", None)
    model_name = getattr(model, "name", None)
    threshold = float(getattr(model, "threshold", 0.5))

    if model_feature_set is not None and model_feature_set != args.feature_set:
        raise ValueError(
            f"Feature-set mismatch: model was trained with "
            f"{model_feature_set}, but --feature-set={args.feature_set}."
        )

    print(f"Model:        {model_name}")
    print(f"Feature set:  {model_feature_set}")
    print(f"Threshold:    {threshold}")

    # IMPORTANT:
    # pdml Temporal.scores() applies the saved StandardScaler and loads
    # the saved TCN state internally. Do not fit anything on TEST.
    scores = np.asarray(model.scores(x), dtype=float)

    if len(scores) != len(meta):
        raise RuntimeError(
            f"Prediction length mismatch: {len(scores)} scores "
            f"for {len(meta)} TEST windows."
        )

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    print("\n[7/7] Evaluating predictions...")

    overall_metrics = print_metrics(
        "OVERALL TEST",
        meta,
        scores,
        threshold,
    )

    metrics = {
        "outer_fold": args.outer_fold,
        "model": model_name,
        "feature_set": model_feature_set,
        "threshold": threshold,
        "overall": overall_metrics,
    }

    # Per-source metrics
    if "source" in meta.columns:
        metrics["by_source"] = {}

        for source, idx in meta.groupby("source").groups.items():
            idx = np.asarray(list(idx), dtype=int)
            source_meta = meta.iloc[idx]
            source_scores = scores[idx]

            metrics["by_source"][str(source)] = binary_metrics(
                source_meta["label"].to_numpy(dtype=int),
                source_scores,
                threshold,
            )

            print_metrics(
                f"TEST SOURCE: {source}",
                source_meta,
                source_scores,
                threshold,
            )

    # Per-label counts
    metrics["label_counts"] = {
        str(k): int(v)
        for k, v in meta["label"].value_counts().sort_index().items()
    }

    # Save predictions
    pred = meta.copy()
    pred["score"] = scores
    pred["prediction"] = (scores >= threshold).astype(int)
    pred["prediction_label"] = pred["prediction"].map(
        {0: "HC", 1: "PD"}
    )
    pred.to_csv(out / "test_predictions.csv", index=False)

    (out / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print("\nDONE")
    print(f"TEST artifacts: {out}")
    print(f"Predictions:    {out / 'test_predictions.csv'}")
    print(f"Metrics:        {out / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
