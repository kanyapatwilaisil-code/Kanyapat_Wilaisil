#!/usr/bin/env python3
"""
pose_quality_report_v2.py

Quality control for side-view gait pose.npz files.

Why v2:
- Does NOT reject a clip just because the whole video is not valid.
- Looks for usable consecutive gait-like segments.
- Reports global quality AND best usable segment quality.
- Uses required joints used by the current F3 pipeline.
- Keeps REVIEW clips instead of aggressively rejecting them.

Usage:
    python pose_quality_report_v2.py \
        --root /path/to/side_pose \
        --output pose_quality_report_v2.csv

Optional:
    --min-segment-sec 3.0
    --min-joint-valid 0.50
    --min-segment-valid 0.60

QC philosophy:
    PASS   = at least one usable segment >= min_segment_sec,
             segment valid rate >= min_segment_valid,
             and required-joint validity >= min_joint_valid.

    REVIEW = usable evidence exists but is weaker than PASS.

    REJECT = essentially no usable segment.

This is a QC screening tool, not a medical or model-performance metric.
Do not automatically delete REVIEW/REJECT clips.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Same 17-joint convention used by the side-view extractor/F3 pipeline.
REQUIRED = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
JOINT_NAMES = {
    1: "right_hip", 2: "right_knee", 3: "right_ankle",
    4: "left_hip", 5: "left_knee", 6: "left_ankle",
    11: "left_shoulder", 12: "left_elbow", 13: "left_wrist",
    14: "right_shoulder", 15: "right_elbow", 16: "right_wrist",
}

def scalar(z, default=np.nan):
    try:
        a = np.asarray(z)
        if a.size == 1:
            return float(a.reshape(-1)[0])
    except Exception:
        pass
    return default

def longest_run(mask):
    best = cur = 0
    start = best_start = end = 0
    for i, x in enumerate(np.asarray(mask, dtype=bool)):
        if x:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best = cur
                best_start = start
                end = i
        else:
            cur = 0
    return best, best_start, end

def segments(mask, min_len):
    """Return contiguous True segments of at least min_len frames."""
    out = []
    mask = np.asarray(mask, dtype=bool)
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        s = i
        while i < n and mask[i]:
            i += 1
        e = i - 1
        if e - s + 1 >= min_len:
            out.append((s, e))
    return out

def inspect(path, min_segment_sec, min_joint_valid, min_segment_valid):
    try:
        d = np.load(path, allow_pickle=True)
        seq = np.asarray(d["sequence"], dtype=float)

        if seq.ndim != 3 or seq.shape[1] < 17 or seq.shape[2] < 3:
            raise ValueError(f"unexpected sequence shape: {seq.shape}")

        fps = scalar(d["source_fps"]) if "source_fps" in d else np.nan
        if not np.isfinite(fps) or fps <= 0:
            if "timestamps" in d:
                ts = np.asarray(d["timestamps"], dtype=float)
                if len(ts) > 1:
                    dt = np.nanmedian(np.diff(ts))
                    if np.isfinite(dt) and dt > 0:
                        fps = 1.0 / dt

        T = len(seq)
        required = seq[:, REQUIRED, :]
        joint_ok = np.all(np.isfinite(required), axis=2)  # T x 12

        # Frame is considered usable if enough of the required joints are valid.
        per_frame_joint_rate = np.mean(joint_ok, axis=1) if T else np.array([])
        usable_frame = per_frame_joint_rate >= min_joint_valid

        # A stricter frame: all required joints available.
        complete_frame = np.all(joint_ok, axis=1) if T else np.array([], dtype=bool)

        # Find contiguous usable regions rather than judging the entire video.
        min_frames = max(1, int(round(min_segment_sec * fps))) if np.isfinite(fps) and fps > 0 else 45
        segs = segments(usable_frame, min_frames)

        best = None
        for s, e in segs:
            seg_ok = usable_frame[s:e+1]
            seg_joint_rate = float(np.mean(per_frame_joint_rate[s:e+1]))
            complete_rate = float(np.mean(complete_frame[s:e+1]))
            length = e - s + 1
            score = (length, seg_joint_rate, complete_rate)
            if best is None or score > best["score"]:
                best = {
                    "start": s, "end": e, "frames": length,
                    "duration": length / fps if np.isfinite(fps) and fps > 0 else np.nan,
                    "joint_rate": seg_joint_rate,
                    "complete_rate": complete_rate,
                    "score": score,
                }

        # Global metrics
        global_joint_rate = float(np.mean(joint_ok)) if joint_ok.size else 0.0
        global_usable_rate = float(np.mean(usable_frame)) if T else 0.0
        global_complete_rate = float(np.mean(complete_frame)) if T else 0.0

        bad_run, bad_s, bad_e = longest_run(~usable_frame) if T else (0, 0, 0)
        bad_run_sec = bad_run / fps if np.isfinite(fps) and fps > 0 else np.nan

        # Per-joint validity, now correctly calculated independently.
        per_joint = {
            f"{JOINT_NAMES[j]}_valid_rate": (
                float(np.mean(joint_ok[:, i])) if T else 0.0
            )
            for i, j in enumerate(REQUIRED)
        }

        # Decision is based on the best usable segment, not the whole clip.
        if best is None:
            decision = "REJECT"
            reason = "no_contiguous_usable_segment"
        elif best["duration"] >= min_segment_sec and best["joint_rate"] >= 0.70:
            decision = "PASS"
            reason = "usable_gait_segment_found"
        elif best["duration"] >= min_segment_sec and best["joint_rate"] >= 0.50:
            decision = "REVIEW"
            reason = "usable_segment_but_moderate_pose_quality"
        else:
            decision = "REVIEW"
            reason = "short_or_low_quality_usable_segment"

        source_video = ""
        if "source_video" in d:
            try:
                source_video = str(np.asarray(d["source_video"]).reshape(-1)[0])
            except Exception:
                source_video = ""

        direction = ""
        if "walking_direction" in d:
            try:
                direction = str(np.asarray(d["walking_direction"]).reshape(-1)[0])
            except Exception:
                direction = ""

        duration = T / fps if np.isfinite(fps) and fps > 0 else np.nan

        row = {
            "pose_file": str(path),
            "source_video": source_video,
            "frames": T,
            "source_fps": fps,
            "duration_sec": duration,
            "global_joint_valid_rate": global_joint_rate,
            "global_usable_frame_rate": global_usable_rate,
            "global_complete_frame_rate": global_complete_rate,
            "longest_unusable_run_sec": bad_run_sec,
            "best_segment_start_frame": best["start"] if best else np.nan,
            "best_segment_end_frame": best["end"] if best else np.nan,
            "best_segment_duration_sec": best["duration"] if best else 0.0,
            "best_segment_joint_valid_rate": best["joint_rate"] if best else 0.0,
            "best_segment_complete_frame_rate": best["complete_rate"] if best else 0.0,
            "walking_direction": direction,
            "decision": decision,
            "reason": reason,
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
            "global_joint_valid_rate": np.nan,
            "global_usable_frame_rate": np.nan,
            "global_complete_frame_rate": np.nan,
            "longest_unusable_run_sec": np.nan,
            "best_segment_start_frame": np.nan,
            "best_segment_end_frame": np.nan,
            "best_segment_duration_sec": np.nan,
            "best_segment_joint_valid_rate": np.nan,
            "best_segment_complete_frame_rate": np.nan,
            "walking_direction": "",
            "decision": "ERROR",
            "reason": "",
            "error": repr(e),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", default="pose_quality_report_v2.csv")
    ap.add_argument("--min-segment-sec", type=float, default=3.0)
    ap.add_argument("--min-joint-valid", type=float, default=0.50)
    ap.add_argument("--min-segment-valid", type=float, default=0.60)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    files = sorted(root.rglob("pose.npz"))
    if not files:
        raise SystemExit(f"No pose.npz found under: {root}")

    rows = [
        inspect(
            p,
            args.min_segment_sec,
            args.min_joint_valid,
            args.min_segment_valid,
        )
        for p in files
    ]

    df = pd.DataFrame(rows)

    first = [
        "pose_file", "source_video", "frames", "source_fps", "duration_sec",
        "global_joint_valid_rate", "global_usable_frame_rate",
        "global_complete_frame_rate", "longest_unusable_run_sec",
        "best_segment_start_frame", "best_segment_end_frame",
        "best_segment_duration_sec", "best_segment_joint_valid_rate",
        "best_segment_complete_frame_rate", "walking_direction",
        "decision", "reason", "error"
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
    print("v2 QC:")
    print(f"  minimum usable segment: {args.min_segment_sec:.1f} sec")
    print(f"  usable frame threshold: {args.min_joint_valid:.0%} required joints valid")
    print("  PASS   = best segment >= minimum duration and >= 70% joint validity")
    print("  REVIEW = usable evidence exists but quality is moderate")
    print("  REJECT = no sufficiently long usable segment")
    print()
    print("Do NOT automatically delete REVIEW/REJECT clips.")

if __name__ == "__main__":
    main()
