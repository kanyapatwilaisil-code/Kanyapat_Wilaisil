#!/usr/bin/env python3
"""
pose_quality_report.py

Scan extracted pose.npz files recursively and create pose_quality_report.csv.

Usage:
    python pose_quality_report.py --root /path/to/side_pose --output pose_quality_report.csv

The report is designed for the side-view 17-joint pipeline.
It reports:
- frame count / duration / source FPS
- valid frames
- required-joint validity
- complete-pose rate
- longest consecutive bad-pose run
- per-required-joint validity
- automatic QC decision

Recommended initial rules:
    PASS:
      required_joint_valid_rate >= 0.80
      AND complete_required_frame_rate >= 0.70
      AND longest_bad_run_seconds <= 2.0

    REVIEW:
      not PASS, but required_joint_valid_rate >= 0.50

    REJECT:
      required_joint_valid_rate < 0.50

These thresholds are QC heuristics, not model-performance thresholds.
Review borderline clips before final training.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
JOINT_NAMES = {
    1: "right_hip", 2: "right_knee", 3: "right_ankle",
    4: "left_hip", 5: "left_knee", 6: "left_ankle",
    11: "left_shoulder", 12: "left_elbow", 13: "left_wrist",
    14: "right_shoulder", 15: "right_elbow", 16: "right_wrist",
}

def longest_run(mask):
    best = cur = 0
    for x in np.asarray(mask, dtype=bool):
        if x:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

def scalar(z, default=np.nan):
    try:
        a = np.asarray(z)
        if a.size == 1:
            return float(a.reshape(-1)[0])
    except Exception:
        pass
    return default

def inspect(path):
    try:
        d = np.load(path, allow_pickle=True)
        seq = np.asarray(d["sequence"], dtype=float)
        ts = np.asarray(d["timestamps"], dtype=float) if "timestamps" in d else None

        fps = scalar(d["source_fps"]) if "source_fps" in d else np.nan
        if not np.isfinite(fps) or fps <= 0:
            if ts is not None and len(ts) > 1:
                dt = np.nanmedian(np.diff(ts))
                fps = 1.0 / dt if dt > 0 else np.nan

        T = seq.shape[0] if seq.ndim >= 1 else 0

        required = seq[:, REQUIRED, :]
        required_joint_ok = np.all(np.isfinite(required), axis=2)  # T x 12
        complete_required = np.all(required_joint_ok, axis=1)      # T

        any_pose = np.any(np.isfinite(seq), axis=(1, 2))
        complete_pose = np.all(np.isfinite(seq), axis=(1, 2))

        if "required_joints_valid" in d:
            stored = np.asarray(d["required_joints_valid"], dtype=bool)
            if stored.shape == complete_required.shape:
                complete_required = np.all(stored, axis=1)
                required_joint_ok = stored

        valid_joint_rate = float(np.mean(required_joint_ok)) if required_joint_ok.size else 0.0
        complete_required_rate = float(np.mean(complete_required)) if T else 0.0
        complete_pose_rate = float(np.mean(complete_pose)) if T else 0.0
        any_pose_rate = float(np.mean(any_pose)) if T else 0.0

        longest_bad = longest_run(~complete_required)
        longest_bad_sec = longest_bad / fps if np.isfinite(fps) and fps > 0 else np.nan

        per_joint = {
            f"{JOINT_NAMES[j]}_valid_rate": (
                float(np.mean(required_joint_ok[:, i]))
                if required_joint_ok.size else 0.0
            )
            for i, j in enumerate(REQUIRED)
        }

        if valid_joint_rate >= 0.80 and complete_required_rate >= 0.70 and (
            not np.isfinite(longest_bad_sec) or longest_bad_sec <= 2.0
        ):
            decision = "PASS"
        elif valid_joint_rate >= 0.50:
            decision = "REVIEW"
        else:
            decision = "REJECT"

        source_video = str(d["source_video"]) if "source_video" in d else ""
        direction = scalar(d["walking_direction"]) if "walking_direction" in d else np.nan

        duration = (T / fps) if np.isfinite(fps) and fps > 0 else (
            float(ts[-1] - ts[0]) if ts is not None and len(ts) > 1 else np.nan
        )

        row = {
            "pose_file": str(path),
            "source_video": source_video,
            "frames": T,
            "source_fps": fps,
            "duration_sec": duration,
            "any_pose_rate": any_pose_rate,
            "complete_pose_rate": complete_pose_rate,
            "required_joint_valid_rate": valid_joint_rate,
            "complete_required_frame_rate": complete_required_rate,
            "longest_bad_run_frames": longest_bad,
            "longest_bad_run_sec": longest_bad_sec,
            "walking_direction": direction,
            "decision": decision,
            "error": "",
        }
        row.update(per_joint)
        return row

    except Exception as e:
        return {
            "pose_file": str(path),
            "source_video": "",
            "frames": np.nan,
            "source_fps": np.nan,
            "duration_sec": np.nan,
            "any_pose_rate": np.nan,
            "complete_pose_rate": np.nan,
            "required_joint_valid_rate": np.nan,
            "complete_required_frame_rate": np.nan,
            "longest_bad_run_frames": np.nan,
            "longest_bad_run_sec": np.nan,
            "walking_direction": np.nan,
            "decision": "ERROR",
            "error": repr(e),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing extracted pose.npz files")
    ap.add_argument("--output", default="pose_quality_report.csv")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    files = sorted(root.rglob("pose.npz"))
    if not files:
        raise SystemExit(f"No pose.npz found under: {root}")

    rows = [inspect(p) for p in files]
    df = pd.DataFrame(rows)

    # Put the most useful columns first.
    first = [
        "pose_file", "source_video", "frames", "source_fps", "duration_sec",
        "required_joint_valid_rate", "complete_required_frame_rate",
        "longest_bad_run_sec", "complete_pose_rate", "any_pose_rate",
        "walking_direction", "decision", "error"
    ]
    remaining = [c for c in df.columns if c not in first]
    df = df[first + remaining]

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Scanned: {len(files)} pose.npz files")
    print(f"Saved:   {out}")
    print()
    print(df["decision"].value_counts(dropna=False).to_string())
    print()
    print("QC rules:")
    print("  PASS   = required valid >= 80%, complete required frames >= 70%, bad run <= 2 sec")
    print("  REVIEW = required valid >= 50% but not PASS")
    print("  REJECT = required valid < 50%")
    print()
    print("Do not delete REVIEW/REJECT automatically; inspect them before training.")

if __name__ == "__main__":
    main()
