#!/usr/bin/env python3
"""
Batch side-view pose extraction for separate HC/PD folders.

Examples from the project layout:
    model/
      extract_side_pose.py
      batch_extract_side_pose.py
      pose_landmarker_full.task

    Video_normal_gait/
      *.mp4

    Video_parkinson_gait/
      *.mp4

Run from model/:

    python batch_extract_side_pose.py \
      --input ../Video_normal_gait \
      --output ../side_pose_validation/HC \
      --label HC \
      --model pose_landmarker_full.task

    python batch_extract_side_pose.py \
      --input ../Video_parkinson_gait \
      --output ../side_pose_validation/PD \
      --label PD \
      --model pose_landmarker_full.task

Each video gets its own directory containing pose.npz.
A manifest.csv is written for each group.
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def find_videos(root):
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def subject_from_filename(stem):
    """
    Default subject parsing.

    Examples:
        PD001_01 -> PD001
        PD001_02 -> PD001
        HC001    -> HC001

    If filenames use another convention, the script keeps the whole
    filename stem as the subject ID rather than guessing.
    """
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].upper().startswith(("PD", "HC")):
        return parts[0]
    return stem


def main():
    ap = argparse.ArgumentParser(
        description="Batch side-view pose extraction"
    )
    ap.add_argument("--input", required=True,
                    help="Folder containing videos")
    ap.add_argument("--output", required=True,
                    help="Output folder for pose.npz files")
    ap.add_argument("--label", required=True, choices=["HC", "PD"],
                    help="Ground-truth group for this input folder")
    ap.add_argument("--model", required=True,
                    help="Path to pose_landmarker_full.task")
    ap.add_argument("--extractor", default=None,
                    help="Path to extract_side_pose.py")
    ap.add_argument("--no-overlay", action="store_true",
                    help="Do not create pose_overlay.mp4")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract clips whose pose.npz already exists")
    args = ap.parse_args()

    root = Path(args.input).resolve()
    out = Path(args.output).resolve()
    model = Path(args.model).resolve()

    if args.extractor:
        extractor = Path(args.extractor).resolve()
    else:
        extractor = Path(__file__).resolve().with_name(
            "extract_side_pose.py"
        )

    if not root.exists():
        raise SystemExit(f"Input directory not found: {root}")
    if not model.exists():
        raise SystemExit(f"Pose model not found: {model}")
    if not extractor.exists():
        raise SystemExit(f"Extractor not found: {extractor}")

    videos = find_videos(root)
    if not videos:
        raise SystemExit(f"No supported video files found under {root}")

    out.mkdir(parents=True, exist_ok=True)

    rows = []
    ok = skipped = failed = 0

    print(f"Group  : {args.label}")
    print(f"Input  : {root}")
    print(f"Output : {out}")
    print(f"Videos : {len(videos)}")
    print()

    for i, video in enumerate(videos, 1):
        clip_dir = out / video.stem
        pose_file = clip_dir / "pose.npz"
        subject = subject_from_filename(video.stem)

        print(f"[{i}/{len(videos)}] {video.name}")

        if pose_file.exists() and not args.overwrite:
            status = "skipped"
            skipped += 1
            print("  -> SKIP (pose.npz exists)")
        else:
            clip_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(extractor),
                "--video", str(video),
                "--output", str(clip_dir),
                "--model", str(model),
            ]

            if args.no_overlay:
                cmd.append("--no-overlay")

            result = subprocess.run(cmd)

            if result.returncode == 0 and pose_file.exists():
                status = "ok"
                ok += 1
                print("  -> OK")
            else:
                status = "failed"
                failed += 1
                print("  -> FAILED")

        rows.append({
            "clip": video.name,
            "video_path": str(video),
            "subject": subject,
            "label": 0 if args.label == "HC" else 1,
            "group": args.label,
            "view": "side",
            "status": status,
            "pose_file": str(pose_file) if pose_file.exists() else "",
        })

    manifest = out / "manifest.csv"
    fields = [
        "clip", "video_path", "subject", "label",
        "group", "view", "status", "pose_file"
    ]

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("========== SUMMARY ==========")
    print(f"Total   : {len(videos)}")
    print(f"Success : {ok}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
