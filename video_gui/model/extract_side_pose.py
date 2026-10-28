#!/usr/bin/env python3
"""
Extract a side-view walking pose into pose.npz for the video_gui pipeline.

Input:
    --video path/to/video.mp4
Output:
    pose.npz containing:
      sequence              float32 [T,17,3] MediaPipe world coordinates
      timestamps            float64 [T]
      image_landmarks      float32 [T,17,3] normalized image x,y,z
      people                int32 [T]
      required_joints_valid bool [T,12]

The 17-joint representation follows the app's downstream convention:
0=root, 1-6 lower limbs, 7-10 trunk/upper body, 11-16 wrists/feet.

The extractor:
- uses MediaPipe Pose Landmarker if available
- keeps the largest/only detected person
- converts the 33 MediaPipe landmarks to 17 joints
- canonicalizes left/right for a person walking either left-to-right or right-to-left
- resamples to 30 Hz
- does NOT invent missing poses; missing frames remain NaN
- writes a side-view overlay video when OpenCV is available
"""

import argparse
from pathlib import Path
import json
import math
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit(
        "mediapipe is required. Install with: pip install mediapipe"
    ) from e


# MediaPipe Pose landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32


def safe_mid(a, b):
    return (a + b) / 2.0


def map17(p):
    """Map MediaPipe 33 world/image landmarks to the app's 17-joint array."""
    out = np.full((17, 3), np.nan, dtype=np.float32)

    def put(i, j):
        if j is not None and np.isfinite(p[j]).all():
            out[i] = p[j]

    # Root / pelvis
    if np.isfinite(p[L_HIP]).all() and np.isfinite(p[R_HIP]).all():
        out[0] = safe_mid(p[L_HIP], p[R_HIP])

    put(1, R_HIP)
    put(2, R_KNEE)
    put(3, R_ANKLE)
    put(4, L_HIP)
    put(5, L_KNEE)
    put(6, L_ANKLE)

    # Upper body
    if np.isfinite(p[L_SHOULDER]).all() and np.isfinite(p[R_SHOULDER]).all():
        out[7] = safe_mid(p[L_SHOULDER], p[R_SHOULDER])
    put(8, NOSE)

    put(9, R_SHOULDER)
    put(10, L_SHOULDER)
    put(11, R_WRIST)
    put(12, L_WRIST)
    put(13, R_HEEL)
    put(14, L_HEEL)
    put(15, R_FOOT)
    put(16, L_FOOT)

    return out


def visibility_mask(vis, threshold=0.35):
    """Required joints used by the app's feature extractor."""
    required = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
    return np.isfinite(vis[:, required]) & (vis[:, required] >= threshold)


def resample_linear(sequence, timestamps, hz=30.0):
    """Resample finite pose runs to a regular 30-Hz grid."""
    t = np.asarray(timestamps, dtype=float)
    x = np.asarray(sequence, dtype=float)

    if len(t) < 2:
        return x, t

    grid = np.arange(
        0.0,
        t[-1] - t[0] + 1e-9,
        1.0 / hz,
        dtype=float,
    )

    y = np.full((len(grid),) + x.shape[1:], np.nan, dtype=np.float32)

    for j in range(x.shape[1]):
        for k in range(x.shape[2]):
            v = x[:, j, k]
            good = np.isfinite(v)
            if good.sum() >= 2:
                y[:, j, k] = np.interp(
                    grid,
                    t[good] - t[0],
                    v[good],
                    left=np.nan,
                    right=np.nan,
                )

    return y, grid + t[0]


def walking_direction(sequence):
    """
    Estimate image-horizontal walking direction from pelvis/root.
    Returns +1 for left->right and -1 for right->left.
    Used only to canonicalize the sequence.
    """
    root = sequence[:, 0, :]
    good = np.isfinite(root[:, 0])
    if good.sum() < 10:
        return 1.0

    x = root[good, 0]
    n = max(3, len(x) // 10)
    delta = np.nanmedian(x[-n:]) - np.nanmedian(x[:n])
    return 1.0 if delta >= 0 else -1.0


def canonicalize_side_view(sequence, image_landmarks):
    """
    Canonicalize horizontal walking direction.

    If the person walks right->left, reflect the horizontal coordinate
    and swap anatomical left/right joint pairs so the downstream feature
    convention remains consistent.
    """
    seq = sequence.copy()
    img = image_landmarks.copy()

    direction = walking_direction(seq)
    if direction >= 0:
        return seq, img, direction

    # Reflect x coordinate around zero.
    seq[..., 0] *= -1.0
    img[..., 0] *= -1.0

    # Swap anatomical pairs.
    pairs = [
        (1, 4), (2, 5), (3, 6),
        (9, 10), (11, 12),
        (13, 14), (15, 16),
    ]
    for a, b in pairs:
        seq[:, [a, b]] = seq[:, [b, a]]
        img[:, [a, b]] = img[:, [b, a]]

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

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    outdir = Path(args.output or (video.stem + "_side_pose"))
    outdir.mkdir(parents=True, exist_ok=True)

    if not Path(args.model).exists():
        raise SystemExit(
            f"Pose model not found: {args.model}\n"
            "Download a MediaPipe Pose Landmarker .task model and pass "
            "--model /path/to/pose_landmarker_full.task"
        )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=args.min_pose,
        min_pose_presence_confidence=args.min_pose,
        min_tracking_confidence=args.min_pose,
        output_segmentation_masks=False,
    )

    sequences = []
    images = []
    visibilities = []
    times = []

    writer = None
    if not args.no_overlay and cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(outdir / "pose_overlay.mp4"),
            fourcc,
            fps,
            (width, height),
        )

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_id = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_id >= args.max_frames:
                break

            t_ms = int(round(frame_id * 1000.0 / fps))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            result = landmarker.detect_for_video(mp_image, t_ms)

            world = np.full((33, 3), np.nan, dtype=np.float32)
            image = np.full((33, 3), np.nan, dtype=np.float32)
            vis = np.zeros(33, dtype=np.float32)

            people = 0
            if result.pose_world_landmarks:
                people = 1
                wl = result.pose_world_landmarks[0]
                il = result.pose_landmarks[0]

                for j in range(min(33, len(wl))):
                    world[j] = [wl[j].x, wl[j].y, wl[j].z]
                    image[j] = [il[j].x, il[j].y, il[j].z]
                    vis[j] = float(
                        getattr(il[j], "visibility", 1.0)
                    )

            pose17 = map17(world)
            image17 = map17(image)

            sequences.append(pose17)
            images.append(image17)
            visibilities.append(vis)
            times.append(frame_id / fps)

            if writer is not None:
                # Draw simple 17-joint skeleton for visual QA.
                connections = [
                    (1,2),(2,3),(4,5),(5,6),
                    (1,0),(4,0),(0,7),
                    (9,10),(9,11),(10,12),
                    (3,13),(6,14),(13,15),(14,16),
                ]
                for a, b in connections:
                    if np.isfinite(image17[[a,b], :2]).all():
                        xa, ya = int(image17[a,0]*width), int(image17[a,1]*height)
                        xb, yb = int(image17[b,0]*width), int(image17[b,1]*height)
                        cv2.line(frame, (xa,ya), (xb,yb), (0,255,0), 2)

                for j in range(17):
                    if np.isfinite(image17[j,:2]).all():
                        xj = int(image17[j,0]*width)
                        yj = int(image17[j,1]*height)
                        cv2.circle(frame, (xj,yj), 4, (0,0,255), -1)

                writer.write(frame)

            frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()

    sequence = np.stack(sequences).astype(np.float32)
    image_landmarks = np.stack(images).astype(np.float32)
    visibilities = np.stack(visibilities).astype(np.float32)
    times = np.asarray(times, dtype=np.float64)

    sequence, image_landmarks, direction = canonicalize_side_view(
        sequence, image_landmarks
    )

    # Regularize to the 30-Hz convention used by the app.
    sequence30, timestamps30 = resample_linear(
        sequence, times, hz=30.0
    )

    image30, _ = resample_linear(
        image_landmarks, times, hz=30.0
    )

    # Recompute required-joint validity on the resampled sequence.
    required = [1,2,3,4,5,6,11,12,13,14,15,16]
    required_valid = np.isfinite(
        sequence30[:, required, :]
    ).all(axis=2)

    pose_path = outdir / "pose.npz"
    np.savez_compressed(
        pose_path,
        sequence=sequence30.astype(np.float32),
        timestamps=timestamps30.astype(np.float64),
        image_landmarks=image30.astype(np.float32),
        people=np.ones(len(sequence30), dtype=np.int32),
        required_joints_valid=required_valid.astype(bool),
        source_video=np.array(str(video)),
        source_fps=np.array(float(fps)),
        resample_hz=np.array(30.0),
        walking_direction=np.array(float(direction)),
        joint_count=np.array(17, dtype=np.int32),
    )

    metadata = {
        "source_video": str(video),
        "source_fps": float(fps),
        "source_frames": int(nframes),
        "output_frames": int(len(sequence30)),
        "resample_hz": 30.0,
        "joints": 17,
        "walking_direction": (
            "left_to_right" if direction >= 0 else "right_to_left"
        ),
        "pose_file": str(pose_path),
        "overlay": None if args.no_overlay else str(outdir / "pose_overlay.mp4"),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {pose_path}")
    print(f"Shape: {sequence30.shape}")
    print(f"Walking direction: {metadata['walking_direction']}")
    if writer is not None:
        print(f"Overlay: {outdir / 'pose_overlay.mp4'}")


if __name__ == "__main__":
    main()
