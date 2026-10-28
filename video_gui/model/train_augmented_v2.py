#!/usr/bin/env python3
"""
train_augmented.py

Train a NEW side-view-augmented TCN using:
    1) the existing CARE-PD/DNE TRAIN pose manifest, and
    2) side-view TRAIN pose.npz files that passed QC.

Important:
- This does NOT modify the original weights.npz/scaler.npz bundle.
- It fits a NEW scaler on the combined training split only.
- It uses the existing pdml.features + pdml.models implementation.
- It keeps side-view TEST and UNTOUCHED out when --side-train-root points
  only to the TRAIN directory.
- Validation is subject-disjoint and is created only inside the supplied
  training pool for early stopping.

Expected side directory:
    SIDE_TRAIN/
      HC/
        *.npz or subfolders containing pose.npz
      PD/
        *.npz or subfolders containing pose.npz

Recommended:
    SIDE_TRAIN/
      HC/S001/pose.npz
      ...
      PD/001_PD_01_MD/pose.npz
      ...

DNE input:
    --dne-manifest should contain the DNE rows used by the experiment.
    --dne-splits + --outer-fold define the completely held-out DNE TEST
    subjects that are removed before training/validation.

Example:
    python train_augmented.py \
      --dne-manifest /path/to/DNE_TRAIN/skeleton_manifest.csv \
      --side-train-root /path/to/side_pose/TRAIN \
      --output /path/to/model/augmented_model \
      --pdml-root /path/to/pdml

If your pdml package is already importable from the project root, omit
--pdml-root.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

def import_pdml(pdml_root=None):
    if pdml_root:
        p = str(Path(pdml_root).expanduser().resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
    from pdml.protocol import Protocol
    from pdml.features import extract_windows
    from pdml.models import make_model, save_model
    return Protocol, extract_windows, make_model, save_model

def find_pose_files(root):
    root = Path(root)
    files = sorted(root.rglob("pose.npz"))
    # Also allow directly supplied .npz files.
    files += sorted(p for p in root.rglob("*.npz") if p.name != "pose.npz")
    return sorted(set(files))

def scalar(z, default=np.nan):
    try:
        a = np.asarray(z)
        if a.size == 1:
            return float(a.reshape(-1)[0])
    except Exception:
        pass
    return default

def get_string(d, key, default=""):
    if key not in d:
        return default
    try:
        return str(np.asarray(d[key]).reshape(-1)[0])
    except Exception:
        return default

def infer_side_label(path):
    """
    Infer HC/PD from the side-view directory structure.

    Supported examples:
      side_pose/HC_v3/HC_train/S001/pose.npz -> HC (0)
      side_pose/PD_v3/PD_train/001_PD_01_MD/pose.npz -> PD (1)

    The important rule is that folder names such as HC_v3, HC_train,
    PD_v3, and PD_train are valid cohort/label markers.
    """
    parts = [p.lower() for p in Path(path).parts]

    # Check from the filename upward.  This reaches HC_train/PD_train
    # before subject folders such as S001 or PD001.
    for p in reversed(parts):
        if p in ("hc", "normal", "healthy", "control"):
            return 0
        if p in ("pd", "parkinson", "parkinson's"):
            return 1

        # Actual side_pose layout used by this project.
        if p.startswith("hc_") or p == "hc_v3":
            return 0
        if p.startswith("pd_") or p == "pd_v3":
            return 1

    name = Path(path).name.lower()
    if "_pd_" in name or name.startswith("pd"):
        return 1
    if "_hc_" in name or name.startswith("hc"):
        return 0

    raise ValueError(f"Cannot infer HC/PD label from path: {path}")

def infer_side_subject(path, label):
    """
    Conservative subject-ID inference.

    Examples supported:
      S001/...                  -> S001
      001_PD_01_MD/...         -> PD001
      PD001/...                -> PD001

    If the side TRAIN folders already contain one subject per directory,
    the immediate parent directory is used when possible.
    """
    p = Path(path)

    # If pose.npz is inside a subject folder, use that folder.
    if p.name == "pose.npz" and p.parent.name.lower() not in {
        "hc", "pd", "normal", "healthy", "control", "parkinson"
    }:
        parent = p.parent.name
        if re.search(r"[A-Za-z]", parent):
            return parent

    stem = p.stem
    m = re.match(r"(?i)^S(\d+)", stem)
    if m:
        return "S" + m.group(1)

    m = re.match(r"(?i)^(\d+)_PD_", stem)
    if m:
        return "PD" + m.group(1)

    m = re.match(r"(?i)^PD[_-]?(\d+)", stem)
    if m:
        return "PD" + m.group(1)

    # Fallback: first numeric identifier.
    m = re.search(r"(\d+)", stem)
    if m:
        prefix = "PD" if label == 1 else "HC"
        return prefix + m.group(1)

    raise ValueError(
        f"Cannot infer subject ID from {path}. "
        "Put each subject in its own directory or rename the file."
    )

def load_side_pose(path):
    d = np.load(path, allow_pickle=True)
    if "sequence" not in d:
        raise ValueError("missing sequence")
    seq = np.asarray(d["sequence"], dtype=float)

    if seq.ndim != 3 or seq.shape[1:] != (17, 3):
        raise ValueError(f"expected sequence (T,17,3), got {seq.shape}")

    fps = scalar(d["source_fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("missing/invalid source_fps")

    if "timestamps" in d:
        times = np.asarray(d["timestamps"], dtype=float)
        if len(times) != len(seq):
            times = np.arange(len(seq), dtype=float) / fps
    else:
        times = np.arange(len(seq), dtype=float) / fps

    return seq, times, fps, get_string(d, "source_video", path.name)

def load_dne_rows(manifest_path):
    """
    Read an existing TRAIN-only DNE skeleton_manifest.csv.
    The manifest rows must point to an .npz containing the sequence_key.
    """
    manifest_path = Path(manifest_path)
    meta = pd.read_csv(manifest_path)

    required_cols = {"sample_id", "group_id", "label", "npz_path", "sequence_key", "effective_fps"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"DNE manifest missing columns: {sorted(missing)}")

    rows = []
    for r in meta.to_dict("records"):
        label = int(r["label"])
        if label not in (0, 1):
            continue
        npz_path = Path(r["npz_path"])
        with np.load(npz_path, allow_pickle=False) as z:
            key = r["sequence_key"]
            if key not in z:
                raise ValueError(f"{key} not found in {npz_path}")
            seq = np.asarray(z[key], dtype=float)

        fps = float(r["effective_fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid effective_fps for {r['sample_id']}")

        times = np.arange(len(seq), dtype=float) / fps
        rows.append((seq, times, fps, r, "DNE"))

    return rows

def make_side_rows(side_root):
    """
    Load ONLY the side-view TRAIN directories:

        HC_v3/HC_train
        PD_v3/PD_train

    This intentionally does not recursively scan the whole side_root,
    because doing so could accidentally include TEST/UNTOUCHED.
    """
    root = Path(side_root)

    train_dirs = [
        root / "HC_v3" / "HC_train",
        root / "PD_v3" / "PD_train",
    ]

    files = []
    for d in train_dirs:
        if not d.exists():
            raise ValueError(f"Missing side TRAIN directory: {d}")
        files.extend(sorted(d.rglob("pose.npz")))

    if not files:
        raise ValueError(
            f"No pose.npz files found under side TRAIN dirs: {train_dirs}"
        )

    rows = []
    for path in files:
        label = infer_side_label(path)
        subject = infer_side_subject(path, label)
        seq, times, fps, source_video = load_side_pose(path)

        r = {
            "sample_id": f"SIDE::{subject}::{path.stem}",
            "group_id": f"SIDE::{subject}",
            "subject_id": subject,
            "walk_id": path.stem,
            "cohort": "SIDE",
            "label": int(label),
            "label_source": "folder",
            "raw_fps": fps,
            "effective_fps": fps,
            "frames": len(seq),
            "source_video": source_video,
            "pose_file": str(path),
        }
        rows.append((seq, times, fps, r, "SIDE"))

    return rows

def load_outer_test_groups(splits_path, outer_fold):
    """
    Read the CARE-PD splits.json and return ONLY the selected outer-fold
    TEST groups.

    Expected structure:
        {
          "outer": [
            {"train": [...], "test": [...], ...},
            ...
          ]
        }

    We refuse to guess from inner splits or from a top-level key.
    """
    p = Path(splits_path)
    if not p.exists():
        raise FileNotFoundError(f"DNE splits.json not found: {p}")

    with p.open() as f:
        splits = json.load(f)

    outer = splits.get("outer")
    if not isinstance(outer, list):
        raise KeyError(
            "splits.json does not contain an 'outer' list. "
            "Refusing to guess a TEST split."
        )

    if outer_fold < 0 or outer_fold >= len(outer):
        raise ValueError(
            f"--outer-fold {outer_fold} is out of range; "
            f"available folds: 0..{len(outer)-1}"
        )

    fold = outer[outer_fold]
    if not isinstance(fold, dict) or "test" not in fold:
        raise KeyError(
            f"splits.json outer fold {outer_fold} does not contain 'test'."
        )

    groups = fold["test"]
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"Outer fold {outer_fold} TEST group list is empty.")

    return {str(g) for g in groups}


def exclude_dne_test_groups(dne_rows, heldout_groups):
    """
    Remove every DNE row belonging to the held-out outer TEST subjects.
    """
    kept = []
    excluded = []

    for row in dne_rows:
        meta = row[3]
        group_id = str(meta.get("group_id"))

        if group_id in heldout_groups:
            excluded.append(row)
        else:
            kept.append(row)

    return kept, excluded


def check_no_heldout_dne_in_training(rows, heldout_groups):
    """
    Hard safety check: no held-out DNE TEST group may enter the training pool.
    """
    leaked = sorted({
        str(row[3].get("group_id"))
        for row in rows
        if str(row[3].get("group_id")) in heldout_groups
    })

    if leaked:
        raise RuntimeError(
            "LEAKAGE: held-out DNE TEST groups are still present in the "
            "training pool:\n" + "\n".join(leaked)
        )


def check_side_train_paths(rows):
    """
    Hard safety check: side training rows must come only from TRAIN paths.
    """
    bad = []
    for row in rows:
        path = str(row[3].get("pose_file", "")).replace("\\", "/").upper()
        if "/HC_TEST/" in path or "/PD_TEST/" in path:
            bad.append(path)
        if "/HC_UNTOUCHED/" in path or "/PD_UNTOUCHED/" in path:
            bad.append(path)

    if bad:
        raise RuntimeError(
            "LEAKAGE: side TEST/UNTOUCHED file entered the training pool:\n"
            + "\n".join(sorted(set(bad)))
        )


def build_windows(rows, cfg, extract_windows):
    all_x = []
    all_meta = []
    clip_rows = []
    reject_rows = []

    for seq, times, fps, r, source in rows:
        try:
            x, accepted, bad, _signal = extract_windows(seq, times, cfg)

            accepted_count = 0
            for w, a in zip(x, accepted):
                accepted_count += 1
                m = dict(r)
                m.update(a)
                # Keep the provenance source at WINDOW level as well as
                # clip/rejection level.  The training audit prints
                # meta["source"], so every accepted window must carry it.
                m["source"] = source
                m["window_id"] = f"{r['sample_id']}::{a['start_frame']}"
                all_x.append(w)
                all_meta.append(m)

            for b in bad:
                reject_rows.append(
                    dict(
                        sample_id=r["sample_id"],
                        group_id=r["group_id"],
                        label=r["label"],
                        source=source,
                        **b,
                    )
                )

            clip_rows.append(
                dict(
                    sample_id=r["sample_id"],
                    group_id=r["group_id"],
                    label=r["label"],
                    source=source,
                    windows=accepted_count,
                    rejected_windows=len(bad),
                )
            )
        except Exception as e:
            reject_rows.append(
                dict(
                    sample_id=r["sample_id"],
                    group_id=r["group_id"],
                    label=r["label"],
                    source=source,
                    reason=f"feature_error: {e}",
                )
            )
            clip_rows.append(
                dict(
                    sample_id=r["sample_id"],
                    group_id=r["group_id"],
                    label=r["label"],
                    source=source,
                    windows=0,
                    rejected_windows=1,
                )
            )

    if not all_x:
        raise RuntimeError("No valid windows were produced.")

    return (
        np.stack(all_x).astype(np.float32),
        pd.DataFrame(all_meta),
        pd.DataFrame(clip_rows),
        pd.DataFrame(reject_rows),
    )

def group_split(meta, validation_fraction, seed):
    """
    Subject-disjoint validation split.

    The split is stratified approximately by class at subject level.
    It never splits windows from the same group across train/validation.
    """
    rng = np.random.default_rng(seed)

    subjects = (
        meta[["group_id", "label"]]
        .drop_duplicates("group_id")
        .sort_values("group_id")
        .reset_index(drop=True)
    )

    if subjects["label"].nunique() < 2:
        raise ValueError("Need both HC and PD subjects for validation.")

    val_groups = []
    for label in sorted(subjects["label"].unique()):
        g = subjects.loc[subjects.label == label, "group_id"].to_numpy()
        rng.shuffle(g)
        n = max(1, int(round(len(g) * validation_fraction)))
        # Keep at least one subject of each class in train when possible.
        if len(g) > 1:
            n = min(n, len(g) - 1)
        val_groups.extend(g[:n].tolist())

    val_groups = set(val_groups)
    val_mask = meta.group_id.isin(val_groups).to_numpy()
    tr_mask = ~val_mask

    if meta.iloc[tr_mask].label.nunique() < 2 or meta.iloc[val_mask].label.nunique() < 2:
        raise ValueError(
            "Validation split lost a class. Increase training subjects or "
            "change --validation-fraction."
        )

    return np.flatnonzero(tr_mask), np.flatnonzero(val_mask), sorted(val_groups)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dne-manifest", required=True,
                    help="DNE skeleton_manifest.csv containing the DNE rows.")
    ap.add_argument("--dne-splits", required=True,
                    help="CARE-PD splits.json containing outer TEST groups.")
    ap.add_argument("--outer-fold", required=True, type=int,
                    help="Outer fold whose DNE TEST groups are held out.")
    ap.add_argument("--side-train-root", required=True,
                    help="Side-view root containing HC_v3/HC_train and PD_v3/PD_train")
    ap.add_argument("--output", required=True,
                    help="New model output directory")
    ap.add_argument("--pdml-root", default=None,
                    help="Parent directory containing the pdml/ package")
    ap.add_argument("--feature-set", default="F3", choices=["F1", "F2", "F3"])
    ap.add_argument("--model", default="tcn", choices=["tcn", "cnn_lstm"])
    ap.add_argument("--width", type=int, default=64, choices=[32, 64])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--validation-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Protocol, extract_windows, make_model, save_model = import_pdml(args.pdml_root)

    cfg = Protocol.read()
    # Keep the frozen project protocol except for explicitly requested seed/epochs.
    cfg = Protocol(**{
        **cfg.dict(),
        "seed": args.seed,
        **({"epochs": args.epochs} if args.epochs is not None else {}),
    })

    print("Protocol:")
    print(json.dumps(cfg.dict(), indent=2))

    print("\n[1/6] Loading DNE outer split...")
    heldout_dne_groups = load_outer_test_groups(
        args.dne_splits,
        args.outer_fold,
    )
    print(f"Selected outer fold: {args.outer_fold}")
    print(f"Held-out DNE TEST groups: {len(heldout_dne_groups)}")
    for g in sorted(heldout_dne_groups):
        print(" ", g)

    print("\n[2/6] Loading DNE manifest...")
    dne_all = load_dne_rows(args.dne_manifest)
    print(f"DNE manifest rows: {len(dne_all)}")

    dne_rows, dne_excluded = exclude_dne_test_groups(
        dne_all,
        heldout_dne_groups,
    )

    if not dne_excluded:
        raise RuntimeError(
            "No DNE rows matched the selected outer TEST groups. "
            "Check group_id formatting and the supplied manifest."
        )

    # Hard safety check before any windows are built.
    check_no_heldout_dne_in_training(dne_rows, heldout_dne_groups)

    print(f"DNE training-pool rows: {len(dne_rows)}")
    print(f"DNE held-out rows removed: {len(dne_excluded)}")

    print("\n[3/6] Loading side-view PASS TRAIN...")
    side_rows = make_side_rows(args.side_train_root)
    check_side_train_paths(side_rows)
    print(f"Side samples: {len(side_rows)}")

    side_meta = pd.DataFrame([r[3] for r in side_rows])
    print("Side labels:")
    print(side_meta["label"].map({0: "HC", 1: "PD"}).value_counts().to_string())

    rows = dne_rows + side_rows

    print("\n[4/6] Building F3 training windows...")
    x, meta, coverage, rejected = build_windows(rows, cfg, extract_windows)

    # Safety checks.
    if meta[["group_id", "label"]].drop_duplicates("group_id").groupby("group_id").label.nunique().max() > 1:
        raise ValueError("A subject has conflicting labels.")

    if meta["group_id"].isna().any():
        raise ValueError("Missing group_id.")

    print(f"Windows: {len(meta)}")
    print("Window labels:")
    print(meta["label"].map({0: "HC", 1: "PD"}).value_counts().to_string())
    print("\nWindows by source:")
    print(meta["source"].value_counts().to_string())

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "training_windows.npy", x)
    meta.to_csv(out / "training_windows.csv", index=False)
    coverage.to_csv(out / "coverage.csv", index=False)
    rejected.to_csv(out / "window_rejections.csv", index=False)
    (out / "protocol.json").write_text(json.dumps(cfg.dict(), indent=2))

    split_audit = {
        "outer_fold": int(args.outer_fold),
        "heldout_dne_test_groups": sorted(heldout_dne_groups),
        "heldout_dne_rows_removed": int(len(dne_excluded)),
        "dne_manifest_rows": int(len(dne_all)),
        "dne_training_pool_rows": int(len(dne_rows)),
        "side_training_rows": int(len(side_rows)),
    }
    (out / "heldout_test.json").write_text(
        json.dumps(split_audit, indent=2)
    )

    # The selected DNE TEST subjects must not appear anywhere in the
    # training/validation metadata.
    present_groups = set(map(str, meta["group_id"].unique()))
    leaked_groups = sorted(present_groups & heldout_dne_groups)
    if leaked_groups:
        raise RuntimeError(
            "LEAKAGE: selected DNE TEST groups appear in training metadata:\n"
            + "\n".join(leaked_groups)
        )

    print("\n[5/6] Creating subject-disjoint validation split...")
    tr, va, val_groups = group_split(meta, args.validation_fraction, args.seed)

    split_info = {
        "seed": args.seed,
        "train_windows": int(len(tr)),
        "validation_windows": int(len(va)),
        "train_groups": sorted(meta.iloc[tr].group_id.unique().tolist()),
        "validation_groups": val_groups,
        "train_label_counts": {
            str(k): int(v) for k, v in meta.iloc[tr].label.value_counts().sort_index().items()
        },
        "validation_label_counts": {
            str(k): int(v) for k, v in meta.iloc[va].label.value_counts().sort_index().items()
        },
    }
    (out / "split.json").write_text(json.dumps(split_info, indent=2))

    # Explicitly prevent side TEST/UNTOUCHED leakage if the root was mis-specified.
    side_groups = set(meta.loc[meta.source == "SIDE", "group_id"])
    if any(("TEST" in str(g).upper() or "UNTOUCHED" in str(g).upper()) for g in side_groups):
        raise RuntimeError("Possible TEST/UNTOUCHED leakage detected in side group IDs.")

    print("Train groups:", len(split_info["train_groups"]))
    print("Validation groups:", len(split_info["validation_groups"]))
    print("Train labels:", split_info["train_label_counts"])
    print("Validation labels:", split_info["validation_label_counts"])

    print("\n[6/6] Training NEW TCN...")
    model = make_model(
        args.model,
        args.feature_set,
        {"width": args.width},
        args.seed,
        cfg,
    )

    start = time.perf_counter()
    model.fit(
        x[tr],
        meta.iloc[tr].reset_index(drop=True),
        validation=(
            x[va],
            meta.iloc[va].reset_index(drop=True),
        ),
        epochs=args.epochs,
    )
    elapsed = time.perf_counter() - start

    model_dir = out / "model"
    save_model(model, model_dir)

    # Final validation metrics.
    from pdml.metrics import clip_score
    scores = model.scores(x[va])
    val_meta = meta.iloc[va].copy().reset_index(drop=True)
    val_bal_acc = float(clip_score(val_meta, scores, model.threshold))

    pred = val_meta[["window_id", "sample_id", "group_id", "label"]].copy()
    pred["score"] = scores
    pred["threshold"] = model.threshold
    pred.to_csv(out / "validation_predictions.csv", index=False)

    info = {
        "status": "complete",
        "model": args.model,
        "feature_set": args.feature_set,
        "width": args.width,
        "training_seconds": elapsed,
        "best_epoch": int(model.best_epoch),
        "validation_clip_balanced_accuracy": val_bal_acc,
        "dne_manifest": str(Path(args.dne_manifest).resolve()),
        "dne_splits": str(Path(args.dne_splits).resolve()),
        "outer_fold": int(args.outer_fold),
        "heldout_dne_test_groups": sorted(heldout_dne_groups),
        "heldout_dne_rows_removed": int(len(dne_excluded)),
        "side_train_root": str(Path(args.side_train_root).resolve()),
        "original_model_untouched": True,
    }
    (out / "run.json").write_text(json.dumps(info, indent=2))

    print("\nDONE")
    print(json.dumps(info, indent=2))
    print(f"\nNEW MODEL: {model_dir}")
    print("Original weights.npz/scaler.npz were not modified.")

if __name__ == "__main__":
    main()
