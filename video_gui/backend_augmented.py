"""Two-pass offline video analysis and annotated H.264 export using source PTS."""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction
import csv
import importlib.metadata
import json
import platform
import time
import uuid
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from inference_augmented import HERE, AugmentedPredictor, sha256
from frozen_pdml.features import NAMES, REQUIRED

print("[BACKEND LOADED]", __file__)

CONNECTIONS = [
    # Left leg
    (1, 2),
    (2, 3),

    # Right leg
    (4, 5),
    (5, 6),

    # Left arm
    (11, 12),
    (12, 13),

    # Right arm
    (14, 15),
    (15, 16),

    # Pelvis / torso
    (1, 0),
    (4, 0),
    (0, 7),

    # Shoulders / torso
    (11, 7),
    (14, 7),

    # Head / neck
    (7, 8),
    (8, 9),
    (9, 10)
    ]
LABELS = {0: "NORMAL GROUP", 1: "PD GROUP"}
NOTE = "Experimental MediaPipe transfer | PD score is not clinical probability"

class Cancelled(Exception):
    pass

def check_cancel(cancel):
    if cancel():
        raise Cancelled("Cancelled by user")

def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def save_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def map_world(landmarks):
    """Map MediaPipe 33-point WORLD coordinates to the frozen 17-joint schema.

    IMPORTANT: do not threshold MediaPipe visibility/presence here.  The
    augmented training pose extractor stores the world coordinates first and
    lets frozen_pdml.features decide whether a frame/window is usable.
    MediaPipe's visibility is an image-space confidence signal; applying it
    directly to world coordinates was causing otherwise finite leg joints to
    become NaN and produced ``invalid_leg_scale`` with zero valid windows.
    """
    a = np.array([[r.x, -r.y, r.z] for r in landmarks], dtype=float)
    out = np.full((17, 3), np.nan)
    for j, k in {1:24,2:26,3:28,4:23,5:25,6:27,11:11,12:13,13:15,14:12,15:14,16:16}.items():
        out[j] = a[k]
    out[0] = (out[1] + out[4]) / 2
    out[8] = (out[11] + out[14]) / 2
    out[7] = (out[0] + out[8]) / 2
    out[10] = a[0]
    out[9] = (out[8] + out[10]) / 2
    return out

def mapped_visibility(landmarks):
    """Return image/world landmark visibility diagnostics in the 17-joint schema."""
    vals = np.full(17, np.nan, dtype=float)
    if not landmarks:
        return vals
    def v(i):
        x = getattr(landmarks[i], "visibility", None)
        return float(x) if x is not None and np.isfinite(x) else np.nan
    for j, k in {1:24,2:26,3:28,4:23,5:25,6:27,11:11,12:13,13:15,14:12,15:14,16:16}.items():
        vals[j] = v(k)
    if np.isfinite(vals[[1,4]]).all(): vals[0] = np.nanmean(vals[[1,4]])
    if np.isfinite(vals[[11,14]]).all(): vals[8] = np.nanmean(vals[[11,14]])
    if np.isfinite(vals[[0,8]]).all(): vals[7] = np.nanmean(vals[[0,8]])
    vals[10] = v(0)
    if np.isfinite(vals[[8,10]]).all(): vals[9] = np.nanmean(vals[[8,10]])
    return vals

def oriented_rgb(frame, stream):
    rgb = frame.to_ndarray(format="rgb24")
    # FFmpeg display-matrix rotation is counterclockwise. Older PyAV may only expose metadata.
    angle = getattr(frame, "rotation", None)
    if angle is None:
        angle = -float(stream.metadata.get("rotate", 0))
    angle = float(angle)
    if abs(angle / 90 - round(angle / 90)) > .01:
        raise ValueError("Unsupported non-right-angle video rotation")
    return np.ascontiguousarray(np.rot90(rgb, int(round(angle / 90)) % 4))

@lru_cache(maxsize=16)
def font(size):
    for path in ["/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)

def local_score(t, rows):
    active = [r["score"] for r in rows if r["start_s"] <= t < r["end_s"]]
    return float(np.mean(active)) if active else None

def draw_frame(rgb, landmarks, lines, visibility=.5):
    """
    Draw the same 17-joint gait skeleton used by the TCN.

    landmarks:
        MediaPipe 33-joint landmarks with:
        x, y, z, visibility, presence
    """

    im = Image.fromarray(rgb).convert("RGB")
    im.thumbnail((1280, 1280))

    w, h = im.size

    # Keep enough width for text
    canvas_w = max(720, w)

    size = max(
        15,
        min(23, canvas_w // 43)
    )

    line_h = size + 9
    f = font(size)

    # -------------------------------------------------
    # Prepare banner text
    # -------------------------------------------------

    wrapped = []

    for text in lines:
        line = ""

        for word in text.split():

            candidate = (
                (line + " " + word).strip()
            )

            if (
                line
                and f.getlength(candidate) > canvas_w - 24
            ):
                wrapped.append(line)
                line = word
            else:
                line = candidate

        if line:
            wrapped.append(line)

    banner_h = line_h * len(wrapped) + 20
    canvas_h = banner_h + h

    canvas = Image.new(
        "RGB",
        (
            canvas_w + canvas_w % 2,
            canvas_h + canvas_h % 2
        ),
        "#111827"
    )

    x0 = (canvas_w - w) // 2

    canvas.paste(
        im,
        (x0, banner_h)
    )

    draw = ImageDraw.Draw(canvas)

    # -------------------------------------------------
    # Draw banner
    # -------------------------------------------------

    for i, line in enumerate(wrapped):

        draw.text(
            (12, 8 + i * line_h),
            line,
            font=f,
            fill="#F3F4F6"
        )

    # -------------------------------------------------
    # Draw each detected person
    # -------------------------------------------------

    for person in landmarks:

        # -------------------------------------------------
        # Convert MediaPipe 33 joints -> TCN 17 joints
        # -------------------------------------------------

        person17 = map_image17(person)

        # -------------------------------------------------
        # IMPORTANT:
        # points belongs to THIS person
        # -------------------------------------------------

        points = {}

        for j, a in enumerate(person17):

            # a = x, y, z, visibility, presence
            if a.shape[0] < 5:
                continue

            if not np.isfinite(a[:3]).all():
                continue

            if not (
                0 <= a[0] <= 1
                and 0 <= a[1] <= 1
            ):
                continue

            # For visualization we don't reject the joint
            # solely because visibility is low.
            points[j] = (
                x0 + int(a[0] * (w - 1)),
                banner_h + int(a[1] * (h - 1))
            )

        # -------------------------------------------------
        # Color
        # -------------------------------------------------

        color = (
            "#34D399"
            if len(landmarks) == 1
            else "#FB7185"
        )

        # -------------------------------------------------
        # Draw 17-joint gait skeleton
        # -------------------------------------------------

        for a, b in CONNECTIONS:

            if a in points and b in points:

                draw.line(
                    [
                        points[a],
                        points[b]
                    ],
                    fill=color,
                    width=max(
                        3,
                        w // 250
                    )
                )

        # -------------------------------------------------
        # Draw joints
        # -------------------------------------------------

        radius = max(
            3,
            w // 320
        )

        for x, y in points.values():

            draw.ellipse(
                (
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius
                ),
                fill=color
            )

    return np.asarray(canvas)

def extract(path, cfg, progress, preview, cancel):
    import av
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HERE / "model/pose_landmarker_full.task"),
                                        delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_poses=2,
        min_pose_detection_confidence=.5, min_tracking_confidence=.5)
    sequence, times, poses, counts, frame_quality, visibility17 = [], [], [], [], [], []
    last_ms, nudged, last_preview = -1, 0, 0.
    with av.open(str(path)) as container, vision.PoseLandmarker.create_from_options(options) as detector:
        if not container.streams.video:
            raise ValueError("No video stream")
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        nominal = float(stream.average_rate) if stream.average_rate else None
        for i, frame in enumerate(container.decode(stream)):
            check_cancel(cancel)
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Missing video PTS; cannot verify timing")
            t = float(frame.pts * frame.time_base)
            if times and t <= times[-1]:
                raise ValueError("Non-increasing video PTS")
            times.append(t)
            relative = t - times[0]
            ms = round(relative * 1000)
            if ms <= last_ms:
                ms = last_ms + 1
                nudged += 1
            last_ms = ms
            rgb = oriented_rgb(frame, stream)
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
            count = len(result.pose_world_landmarks)
            counts.append(count)
            
            if count > 1:
                print(
                    f"[DEBUG MULTI] frame={i} "
                    f"time={relative:.3f}s "
                    f"poses={count}",
                    flush=True,
                )
                for pi, person in enumerate(result.pose_landmarks[:count]):
                    xs = np.array([p.x for p in person], dtype=np.float32)
                    ys = np.array([p.y for p in person], dtype=np.float32)

                    finite = (
                        np.isfinite(xs) &
                        np.isfinite(ys)
                    )
                    if finite.any():
                        print(
                             f"    person {pi}: "
                             f"x=({xs[finite].min():.3f}, {xs[finite].max():.3f}) "
                             f"y=({ys[finite].min():.3f}, {ys[finite].max():.3f})"
                        )
            
            mapped = map_world(result.pose_world_landmarks[0]) if count == 1 else np.full((17,3), np.nan)
            sequence.append(mapped)
            frame_quality.append(bool(np.isfinite(mapped[REQUIRED]).all()))
            visibility17.append(mapped_visibility(result.pose_landmarks[0]) if count == 1 else np.full(17, np.nan))
            stored = np.full((2,33,5), np.nan, dtype=np.float32)
            for j, person in enumerate(result.pose_landmarks[:2]):
                stored[j] = [[p.x, p.y, p.z,
                              1. if p.visibility is None else p.visibility,
                              1. if p.presence is None else p.presence] for p in person]
            poses.append(stored)
            if time.monotonic() - last_preview >= .15:
                progress(min(59, int(60 * relative / duration)) if duration else 0,
                         f"Extracting pose: {i+1} frames | {relative:.1f} s")
                preview(draw_frame(rgb, stored[:count], [
                    "TCN-F3 | Extracting MediaPipe pose...",
                    f"Time: {relative:.2f} s | Detected people: {count}",
                    "Prediction is available after full-clip analysis."], cfg.visibility))
                last_preview = time.monotonic()
    if len(times) < 2:
        raise ValueError("Video must contain at least two decoded frames")
    times = np.asarray(times) - times[0]
    return np.stack(sequence), times, np.stack(poses), np.asarray(counts), np.asarray(frame_quality), dict(
        frames=len(times), nominal_fps=nominal,
        median_effective_fps=float(1 / np.median(np.diff(times))),
        duration_s=float(times[-1] + np.median(np.diff(times))),
        detected_single_person_fraction=float(np.mean(np.asarray(counts) == 1)),
        required_joints_valid_fraction=float(np.mean(frame_quality)),
        visibility_threshold=float(cfg.visibility),
        required_joints_visibility_fraction=float(np.mean(np.isfinite(np.asarray(visibility17)[:, REQUIRED]) & (np.asarray(visibility17)[:, REQUIRED] >= cfg.visibility))),
        multiple_person_frames=int(np.sum(np.asarray(counts) > 1)),
        missing_pose_frames=int(np.sum(np.asarray(counts) == 0)),
        timing="source_presentation_timestamps", timestamp_nudges=nudged)

def map_image17(landmarks33):
    """
    Convert MediaPipe 33 landmarks to the
    17-joint schema used by the TCN.
    """

    a = np.asarray(
        landmarks33,
        dtype=np.float32
    )

    out = np.full(
        (17, 5),
        np.nan,
        dtype=np.float32
    )

    mapping = {
        1: 24,
        2: 26,
        3: 28,

        4: 23,
        5: 25,
        6: 27,

        11: 11,
        12: 13,
        13: 15,

        14: 12,
        15: 14,
        16: 16,
    }

    for dst, src in mapping.items():

        out[dst] = a[src]

    # Center hip
    if np.isfinite(out[[1, 4], :3]).all():
        out[0] = np.nanmean(
            out[[1, 4]],
            axis=0
        )

    # Center shoulder
    if np.isfinite(out[[11, 14], :3]).all():
        out[8] = np.nanmean(
            out[[11, 14]],
            axis=0
        )

    # Center torso
    if np.isfinite(out[[0, 8], :3]).all():
        out[7] = np.nanmean(
            out[[0, 8]],
            axis=0
        )

    # Nose
    out[10] = a[0]

    # Neck/head center
    if np.isfinite(out[[8, 10], :3]).all():
        out[9] = np.nanmean(
            out[[8, 10]],
            axis=0
        )

    return out

def export_video(path, out, times, poses, counts, prediction, rows, cfg, progress, preview, cancel):
    import av
    partial = out / "annotated.partial.mp4"
    output = out / "annotated.mp4"
    end_time = float(times[-1] + np.median(np.diff(times)))
    output_frames = max(1, int(np.ceil(end_time * 30 - 1e-8)))
    threshold = prediction["threshold"]
    with av.open(str(path)) as source, av.open(str(partial), "w", format="mp4", options={"movflags":"+faststart"}) as dest:
        stream = source.streams.video[0]
        decoded = iter(source.decode(stream))
        current = next(decoded)
        next_frame = next(decoded, None)
        source_i = 0
        encoder = None
        last_preview = 0.
        for index in range(output_frames):
            check_cancel(cancel)
            t = index / 30
            while next_frame is not None and source_i + 1 < len(times) and times[source_i+1] <= t + 1e-8:
                current = next_frame
                source_i += 1
                next_frame = next(decoded, None)
            if prediction["status"] == "ok":
                whole = f"Clip prediction: {LABELS[prediction['label']]} | PD score: {prediction['score']:.3f}"
                score = local_score(t, rows)
                segment = "Current interval: no valid window" if score is None else f"Current interval: {LABELS[int(score >= threshold)]} | PD score: {score:.3f}"
            else:
                whole = "Clip prediction: UNABLE TO ASSESS"
                segment = "Reason: " + prediction["reason"]
            lines = ["TCN-F3 | Whole-clip offline analysis", whole, segment,
                     f"Time: {t:.2f} s | Threshold: {threshold:.2f} | Valid windows: {prediction['windows']}",
                     NOTE]
            rgb = draw_frame(oriented_rgb(current, stream), poses[source_i,:counts[source_i]],
                             lines, cfg.visibility)
            if encoder is None:
                encoder = dest.add_stream("libx264", rate=30)
                encoder.width, encoder.height = rgb.shape[1], rgb.shape[0]
                encoder.pix_fmt = "yuv420p"
                encoder.options = {"crf":"20", "preset":"veryfast"}
            rendered = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            rendered.pts = index
            rendered.time_base = Fraction(1, 30)
            for packet in encoder.encode(rendered):
                dest.mux(packet)
            if time.monotonic() - last_preview >= .15:
                preview(rgb)
                progress(65 + int(34 * (index+1)/output_frames), f"Saving annotated video: {index+1}/{output_frames}")
                last_preview = time.monotonic()
        for packet in encoder.encode():
            dest.mux(packet)
    check_cancel(cancel)
    partial.replace(output)
    return dict(filename=output.name, fps=30, frames=output_frames, audio=False,
                duration_s=output_frames/30, codec="H.264", timing="CFR, previous source frame held by PTS",
                max_image_dimension=1280)

def save_plot(path, rows, threshold):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(10,4), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for i, r in enumerate(rows):
        ax.plot([r["start_s"], r["end_s"]], [r["score"]]*2, color="#2563eb",
                alpha=.7, label="Accepted window PD score" if i == 0 else None)
    ax.axhline(threshold, color="#dc2626", linestyle="--", label="Decision threshold")
    if not rows:
        ax.text(.5,.5,"No valid prediction windows", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Time (s)", ylabel="Uncalibrated PD score", ylim=(-.03,1.03),
           title="TCN-F3 — accepted windows (gaps are not predictions)")
    ax.legend()
    fig.savefig(path, dpi=160)

def analyze(video, output_root, progress=lambda *a: None, preview=lambda *a: None,
            cancel=lambda: False, predictor=None, extractor=None):
    
    print("========== [DEBUG] analyze() CALLED ==========")
    print("[DEBUG] backend:", __file__)

    """No filename/folder labels are read. Each run is independent and saved automatically."""
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    out = Path(output_root).expanduser() / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        save_json(out / "status.json", {"status":"running", "input":str(video)})
        progress(0, "Loading augmented TCN-F3 model and preprocessing...")
        check_cancel(cancel)
        predictor = predictor or AugmentedPredictor()
        sequence, times, poses, counts, quality, info = (extractor or extract)(
            video, predictor.cfg, progress, preview, cancel)
        check_cancel(cancel)
        progress(60, "Preparing 30 Hz features and predicting...")
        rows, rejected = [], []
        windows = np.empty((0,predictor.cfg.window,22), np.float32)
        signals = np.empty((0,22), np.float32)
        
        counts_arr = np.asarray(counts)
        
        multiple_person_frames = int(np.sum(counts_arr > 1))
        multiple_person_fraction = (
            multiple_person_frames / len(counts_arr)
            if len(counts_arr)
            else 0.0
        )

        print(
            f"[DEBUG analyze] counts shape={counts_arr.shape}",
            flush=True,
        )
        
        print(
            f"[DEBUG analyze] multiple_person_frames="
            f"{multiple_person_frames}/{len(counts_arr)} "
            f"({multiple_person_fraction:.2%})",
            flush=True,
            )
        print("[DEBUG analyze] people count distribution:",
              {
                  int(x): int(np.sum(counts_arr == x))
                  for x in np.unique(counts_arr)
                  },
                  flush=True,
        )
        
        if multiple_person_fraction > 0.10:
            prediction = dict(
                status="unable_to_assess",
                reason="multiple_people_detected",
                detail=(
                    f"{multiple_person_frames}/{len(counts_arr)} frames "
                    f"({multiple_person_fraction:.1%}) detected multiple people"
                ),
                score=None,
                label=None,
                threshold=predictor.threshold,
                windows=0,
                rejected_windows=0,
            )
        else:
            try:
                prediction, rows, rejected, windows, signals = predictor.predict(sequence, times)
            except ValueError as e:
                if str(e).startswith("Invalid leg scale"):
                    prediction = dict(
                        status="unable_to_assess",
                        reason="invalid_leg_scale",
                        detail=str(e),
                        score=None,
                        label=None,
                        threshold=predictor.threshold,
                        windows=0,
                        rejected_windows=0,
                    )
                    rows = []
                    rejected = []
                    windows = np.empty(
                        (0, predictor.cfg.window, 22), np.float32
                    )
                    signals = np.empty((0, 22), np.float32)
                else:
                    raise
        np.savez_compressed(out / "pose.npz", sequence=sequence, timestamps=times,
                            image_landmarks=poses, people=counts, required_joints_valid=quality)
        np.savez_compressed(out / "features.npz", windows=windows, signals=signals,
                            feature_names=np.array(NAMES))
        save_csv(out / "window_scores.csv", rows,
                 ["start_frame","end_frame","start_s","end_s","interpolated_fraction","score","label"])
        save_csv(out / "rejected_windows.csv", rejected,
                 ["start_frame","end_frame","start_s","reason","interpolated_fraction"])
        save_csv(out / "frame_quality.csv",
                 [dict(time_s=float(t),people=int(c),required_joints_valid=bool(q)) for t,c,q in zip(times,counts,quality)],
                 ["time_s","people","required_joints_valid"])
        save_plot(out / "scores.png", rows, predictor.threshold)
        check_cancel(cancel)
        export = export_video(video, out, times, poses, counts, prediction, rows,
                              predictor.cfg, progress, preview, cancel)
        packages = {}
        for package in ["numpy","pandas","scipy","torch","mediapipe","av","PySide6","Pillow","matplotlib"]:
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        report = dict(prediction=prediction, quality=info, video_export=export,
                      input=str(video), input_sha256=sha256(video),
                      model=predictor.manifest, reference_max_error=predictor.reference_max_error,
                      preprocessing=predictor.cfg.dict(), packages=packages,
                      python=platform.python_version(), platform=platform.platform(),
                      source_sha256={str(p.relative_to(HERE)):sha256(p) for p in sorted(HERE.glob("*.py"))},
                      elapsed_seconds=time.perf_counter()-started,
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      evaluation_metrics=None, ground_truth=None,
                      notes=["MediaPipe world landmarks mapped to h36m joint indices; not SMPL reconstruction.",
                             "Unvalidated domain transfer from CARE-PD to MediaPipe.",
                             "Scores are uncalibrated; no clinical probability or diagnosis.",
                             "Offline noncausal features; whole-clip prediction is shown throughout export.",
                             "No accuracy metrics: ground truth is not supplied.",
                             "Export has no audio; video is resampled to 30 FPS by source timestamps."])
        save_json(out / "prediction.json", report)
        save_json(out / "status.json", {"status":"complete","prediction_status":prediction["status"]})
        progress(100, "Complete — video and results saved")
        return out, report
    except BaseException as e:
        state = "cancelled" if isinstance(e, Cancelled) else "failed"
        save_json(out / "status.json", {"status":state,"reason":str(e)})
        # Incomplete files remain clearly marked and are never presented as completed exports.
        if (out / "annotated.mp4").exists():
            (out / "annotated.mp4").rename(out / "annotated.incomplete.mp4")
        raise


def analyze_pose(pose_path, output_root, progress=lambda *a: None,
                 cancel=lambda: False, predictor=None):
    print("========== [BACKEND DEBUG] analyze_pose ENTERED ==========", flush=True)
    print("[BACKEND DEBUG] file =", __file__, flush=True)
    """Run the augmented TCN-F3 directly from a pre-extracted pose.npz.

    Expected keys: sequence, timestamps. Optional keys are preserved in the
    output report when present. No MediaPipe extraction is performed.
    """
    pose_path = Path(pose_path).resolve()
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)
    out = Path(output_root).expanduser() / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_pose_" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        save_json(out / "status.json", {"status":"running", "input":str(pose_path), "input_type":"pose_npz"})
        progress(0, "Loading augmented TCN-F3 model and preprocessing...")
        check_cancel(cancel)
        predictor = predictor or AugmentedPredictor()
        with np.load(pose_path, allow_pickle=False) as z:
            if "sequence" not in z or "timestamps" not in z:
                raise ValueError("pose.npz must contain 'sequence' and 'timestamps'")
            sequence = np.asarray(z["sequence"], dtype=np.float64)
            times = np.asarray(z["timestamps"], dtype=np.float64)
            poses = np.asarray(z["image_landmarks"], dtype=np.float32) if "image_landmarks" in z else np.empty((len(sequence), 0, 33, 5), np.float32)
            counts = np.asarray(z["people"], dtype=np.int32) if "people" in z else np.ones(len(sequence), np.int32)
            quality = np.asarray(z["required_joints_valid"], dtype=bool)
            # Normalize required_joints_valid to one boolean per frame.
            # # Some pose.npz files store validity per joint, e.g. (T, 17).
            # # The inference/reporting pipeline expects (T,).
            if quality.ndim > 1:
                quality = np.all(
                    quality,
                    axis=tuple(range(1, quality.ndim))
                )
            
            if quality.ndim != 1 or len(quality) != len(sequence):
                raise ValueError(
                    "required_joints_valid must resolve to one boolean per frame: "
                    f"got shape {quality.shape}, expected ({len(sequence)},)"
                )
            
        if sequence.ndim != 3 or sequence.shape[1:] != (17,3):
            raise ValueError(f"Expected sequence shape (T,17,3), got {sequence.shape}")
        if len(times) != len(sequence) or len(times) < 2:
            raise ValueError("Invalid pose timestamps")
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Pose timestamps must be finite and strictly increasing")
        times = times - times[0]
        progress(40, "Using pre-extracted pose — preparing 30 Hz features and predicting...")
        check_cancel(cancel)
        counts_arr = np.asarray(counts)
        
        multiple_person_frames = int(np.sum(counts_arr > 1))
        multiple_person_fraction = (
            multiple_person_frames / len(counts_arr)
            if len(counts_arr)
            else 0.0
        )
        print(
            f"[DEBUG analyze_pose] multiple_person_frames="
            f"{multiple_person_frames}/{len(counts_arr)} "
            f"({multiple_person_fraction:.2%})"
        )
        
        print(
            "[DEBUG analyze_pose] people count distribution:",
            {
                int(x): int(np.sum(counts_arr == x))
                for x in np.unique(counts_arr)
            }
        )
        
        if multiple_person_fraction > 0.10:
            prediction = dict(status="unable_to_assess", reason="multiple_people_detected",
                              score=None, label=None, threshold=predictor.threshold, windows=0, rejected_windows=0)
            rows=[]; rejected=[]; windows=np.empty((0,predictor.cfg.window,22), np.float32); signals=np.empty((0,22), np.float32)
        else:
            try:
                prediction, rows, rejected, windows, signals = predictor.predict(sequence, times)
            except ValueError as e:
                if str(e).startswith("Invalid leg scale"):
                    prediction = dict(status="unable_to_assess", reason="invalid_leg_scale",
                                      detail=str(e), score=None,label=None,threshold=predictor.threshold,windows=0,rejected_windows=0)
                    rows=[]; rejected=[]; windows=np.empty((0,predictor.cfg.window,22), np.float32); signals=np.empty((0,22), np.float32)
                else:
                    raise
        np.savez_compressed(out / "pose.npz", sequence=sequence, timestamps=times,
                            image_landmarks=poses, people=counts, required_joints_valid=quality)
        np.savez_compressed(out / "features.npz", windows=windows, signals=signals,
                            feature_names=np.array(NAMES))
        save_csv(out / "window_scores.csv", rows,
                 ["start_frame","end_frame","start_s","end_s","interpolated_fraction","score","label"])
        save_csv(out / "rejected_windows.csv", rejected,
                 ["start_frame","end_frame","start_s","reason","interpolated_fraction"])
        save_csv(out / "frame_quality.csv",
                 [dict(time_s=float(t),people=int(c),required_joints_valid=bool(q)) for t,c,q in zip(times,counts,quality)],
                 ["time_s","people","required_joints_valid"])
        save_plot(out / "scores.png", rows, predictor.threshold)
        progress(90, "Saving prediction results...")
        packages = {}
        for package in ["numpy","pandas","scipy","torch","mediapipe","av","PySide6","Pillow","matplotlib"]:
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        report = dict(prediction=prediction, quality=dict(
                      frames=len(times), duration_s=float(times[-1] + np.median(np.diff(times))),
                      effective_fps=float(1 / np.median(np.diff(times))),
                      required_joints_valid_fraction=float(np.mean(quality))),
                      input=str(pose_path), input_sha256=sha256(pose_path), input_type="pose_npz",
                      model=predictor.manifest, reference_max_error=predictor.reference_max_error,
                      preprocessing=predictor.cfg.dict(), packages=packages,
                      python=platform.python_version(), platform=platform.platform(),
                      elapsed_seconds=time.perf_counter()-started,
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      evaluation_metrics=None, ground_truth=None,
                      notes=["Inference used pre-extracted pose.npz; MediaPipe extraction was skipped.",
                             "The pose file must use the same 17-joint/world-coordinate convention expected by frozen_pdml.features.",
                             "Scores are uncalibrated; no clinical probability or diagnosis."])
        save_json(out / "prediction.json", report)
        save_json(out / "status.json", {"status":"complete","prediction_status":prediction["status"]})
        progress(100, "Complete — pose prediction and results saved")
        return out, report
    except BaseException as e:
        state = "cancelled" if isinstance(e, Cancelled) else "failed"
        save_json(out / "status.json", {"status":state,"reason":str(e)})
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze one video with augmented TCN-F3")
    parser.add_argument("video")
    parser.add_argument("--output", default=str(Path.home() / "Movies/PD_Gait_Results"))
    args = parser.parse_args()
    folder, report = analyze(args.video, args.output, progress=lambda p,s: print(f"{p}% {s}",flush=True))
    print(folder)
    print(report["prediction"])
