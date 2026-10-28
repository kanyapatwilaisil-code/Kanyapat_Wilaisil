#!/usr/bin/env python3
"""
Side-view pose extractor compatible with frozen_pdml/features.py.

Keeps native video FPS. Resampling to 30 Hz is deferred to features.py.
Output sequence shape: (T, 17, 3).

17-joint convention:
0 root/pelvis
1 R hip, 2 R knee, 3 R ankle
4 L hip, 5 L knee, 6 L ankle
7 shoulder center, 8 nose
9 R eye, 10 L eye
11 L shoulder, 12 L elbow, 13 L wrist
14 R shoulder, 15 R elbow, 16 R wrist
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MP = {
    "NOSE": 0, "RIGHT_EYE": 5, "LEFT_EYE": 2,
    "RIGHT_SHOULDER": 12, "LEFT_SHOULDER": 11,
    "RIGHT_ELBOW": 14, "LEFT_ELBOW": 13,
    "RIGHT_WRIST": 16, "LEFT_WRIST": 15,
    "RIGHT_HIP": 24, "LEFT_HIP": 23,
    "RIGHT_KNEE": 26, "LEFT_KNEE": 25,
    "RIGHT_ANKLE": 28, "LEFT_ANKLE": 27,
}

JOINT_NAMES = [
    "root", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle", "shoulder_center",
    "nose", "right_eye", "left_eye",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
]
REQUIRED = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]


def convert_landmarks(frames):
    out = np.full((len(frames), 17, 3), np.nan, np.float32)
    for t, lm in enumerate(frames):
        if len(lm) != 33:
            continue
        p = np.asarray([[q.x, q.y, q.z] for q in lm], np.float32)
        rh, lh = p[MP["RIGHT_HIP"]], p[MP["LEFT_HIP"]]
        rs, ls = p[MP["RIGHT_SHOULDER"]], p[MP["LEFT_SHOULDER"]]
        out[t] = np.array([
            (rh + lh) / 2, p[MP["RIGHT_HIP"]], p[MP["RIGHT_KNEE"]],
            p[MP["RIGHT_ANKLE"]], p[MP["LEFT_HIP"]], p[MP["LEFT_KNEE"]],
            p[MP["LEFT_ANKLE"]], (rs + ls) / 2, p[MP["NOSE"]],
            p[MP["RIGHT_EYE"]], p[MP["LEFT_EYE"]],
            p[MP["LEFT_SHOULDER"]], p[MP["LEFT_ELBOW"]],
            p[MP["LEFT_WRIST"]], p[MP["RIGHT_SHOULDER"]],
            p[MP["RIGHT_ELBOW"]], p[MP["RIGHT_WRIST"]],
        ], dtype=np.float32)
    return out


def walking_direction(seq):
    root = seq[:, 0, :]
    good = np.isfinite(root[:, 0])
    if good.sum() < 10:
        return 1.0
    x = root[good, 0]
    n = max(3, len(x) // 10)
    return 1.0 if np.nanmedian(x[-n:]) - np.nanmedian(x[:n]) >= 0 else -1.0


def canonicalize(seq, img):
    direction = walking_direction(seq)
    if direction >= 0:
        return seq, img, direction

    seq = seq.copy()
    img = img.copy()
    seq[..., 0] *= -1
    img[..., 0] *= -1

    for a, b in [(1,4), (2,5), (3,6), (11,14), (12,15), (13,16)]:
        seq[:, [a,b]] = seq[:, [b,a]]
        img[:, [a,b]] = img[:, [b,a]]
    return seq, img, direction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--model", default="pose_landmarker_full.task")
    ap.add_argument("--min-pose", type=float, default=0.35)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    video = Path(args.video).resolve()
    out = Path(args.output).resolve() if args.output else video.with_name(video.stem + "_pose")
    out.mkdir(parents=True, exist_ok=True)
    model = Path(args.model).resolve()
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = mp.tasks.BaseOptions(model_asset_path=str(model))
    opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=args.min_pose,
        min_pose_presence_confidence=args.min_pose,
        min_tracking_confidence=args.min_pose,
    )

    world, image, vis_frames, frames, times = [], [], [], [], []
    with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and i >= args.max_frames):
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(round(i * 1000.0 / fps))
            r = landmarker.detect_for_video(mpimg, ts)

            if r.pose_world_landmarks:
                world.append(r.pose_world_landmarks[0])
                image.append(r.pose_landmarks[0])
                vis_frames.append(np.asarray(
                    [getattr(q, "visibility", 1.0) for q in r.pose_landmarks[0]],
                    np.float32))
            else:
                world.append([])
                image.append([])
                vis_frames.append(np.full(33, np.nan, np.float32))

            if not args.no_overlay:
                frames.append(frame)
            times.append(ts / 1000.0)
            i += 1
    cap.release()

    seq = convert_landmarks(world)
    img = convert_landmarks(image)
    vis = np.asarray(vis_frames, np.float32)
    valid = np.isfinite(vis[:, REQUIRED]) & (vis[:, REQUIRED] >= args.min_pose)
    seq, img, direction = canonicalize(seq, img)

    overlay = out / "pose_overlay.mp4"
    if not args.no_overlay and frames:
        writer = cv2.VideoWriter(
            str(overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        edges = [
            (1,2),(2,3),(4,5),(5,6),(11,12),(12,13),(14,15),(15,16),
            (11,7),(14,7),(0,7),(0,1),(0,4),(7,8)
        ]
        for frame, lm in zip(frames, img):
            for a,b in edges:
                if np.all(np.isfinite(lm[[a,b], :2])):
                    p1 = tuple(np.round(lm[a,:2] * [w,h]).astype(int))
                    p2 = tuple(np.round(lm[b,:2] * [w,h]).astype(int))
                    cv2.line(frame, p1, p2, (0,255,0), 2)
            for x,y,_ in lm:
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(frame, (int(x*w), int(y*h)), 3, (0,0,255), -1)
            writer.write(frame)
        writer.release()

    np.savez_compressed(
        out / "pose.npz",
        sequence=seq.astype(np.float32),
        timestamps=np.asarray(times, np.float64),
        image_landmarks=img.astype(np.float32),
        people=np.ones(len(seq), np.int32),
        required_joints_valid=valid.astype(bool),
        source_video=str(video),
        source_fps=np.float64(fps),
        resample_hz=np.float64(np.nan),
        walking_direction=np.float64(direction),
        joint_count=np.int32(17),
    )

    metadata = {
        "format": "side_pose_v2",
        "joint_count": 17,
        "joint_names": JOINT_NAMES,
        "required_joints": REQUIRED,
        "source_video": str(video),
        "source_fps": fps,
        "frame_count": len(seq),
        "walking_direction": direction,
        "walking_direction_label": "left_to_right" if direction >= 0 else "right_to_left",
        "resampling": "deferred_to_frozen_pdml.features",
        "target_hz": 30.0,
        "coordinate_space": "mediapipe_world_landmarks",
        "f3_compatible_joint_mapping": True,
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {out / 'pose.npz'}")
    print(f"Shape: {seq.shape}")
    print(f"Source FPS: {fps:.6g}")
    print("Resample: deferred to features.py (target 30 Hz)")
    print("Walking direction:", "left_to_right" if direction >= 0 else "right_to_left")
    if not args.no_overlay:
        print(f"Overlay: {overlay}")


if __name__ == "__main__":
    main()
