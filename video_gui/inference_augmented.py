"""Inference adapter for the newly trained augmented TCN-F3 model.

Loads the model bundle produced by train_augmented.py:
    model.joblib
    weights.pt
    scaler.joblib
    architecture.json
    model_info.json

The path can be set with PD_AUGMENTED_MODEL_DIR. Default:
    <project-root>/training_results/augmented_f3_tcn_v4/model

Preprocessing/features remain the frozen video_gui/frozen_pdml pipeline,
which uses the same F3=22-channel representation and 45/22 windowing used
by the augmented training run.
"""
from pathlib import Path
import hashlib
import json
import os

import joblib
import numpy as np
import torch
from torch import nn

from frozen_pdml.protocol import Protocol
from frozen_pdml.features import extract_windows, NAMES, columns

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Residual(nn.Module):
    def __init__(self, cin, width, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, width, 3, padding=dilation, dilation=dilation),
            nn.ReLU(), nn.Dropout(.3),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.ReLU(), nn.Dropout(.3),
        )
        self.skip = nn.Conv1d(cin, width, 1) if cin != width else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class TCN(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.blocks = nn.Sequential(*[
            Residual(channels if i == 0 else width, width, d)
            for i, d in enumerate([1, 2, 4, 8])
        ])
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        return self.head(self.blocks(x.transpose(1, 2)).mean(-1)).squeeze(-1)


def default_model_dir():
    # Project layout: experiments_v2/video_gui/model/inference_augmented.py
    # -> experiments_v2/training_results/augmented_f3_tcn_v4/model
    return PROJECT_ROOT / "training_results" / "augmented_f3_tcn_v4" / "model"


class AugmentedPredictor:
    """Drop-in replacement for FixedPredictor used by backend.py."""

    def __init__(self, model_dir=None):
        self.model_dir = Path(
            model_dir or os.environ.get("PD_AUGMENTED_MODEL_DIR", default_model_dir())
        ).expanduser().resolve()

        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                "Augmented model directory not found: "
                f"{self.model_dir}\n"
                "Set PD_AUGMENTED_MODEL_DIR to the directory containing "
                "weights.pt, scaler.joblib and architecture.json."
            )

        required = [
            "weights.pt",
            "scaler.joblib",
            "architecture.json",
            "model_info.json",
        ]
        missing = [x for x in required if not (self.model_dir / x).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Augmented model bundle is incomplete: {missing}\n"
                f"Directory: {self.model_dir}"
            )

        self.info = json.loads((self.model_dir / "model_info.json").read_text())
        self.manifest = {
            "model": self.info.get("model", "tcn"),
            "feature_set": self.info.get("feature_set", "F3"),
            "threshold": float(self.info.get("threshold", 0.5)),
            "model_dir": str(self.model_dir),
            "bundle_type": "pdml_temporal_augmented",
            "training_run": str(self.model_dir.parent),
        }

        if self.info.get("model") != "tcn" or self.info.get("feature_set") != "F3":
            raise ValueError(
                "This app adapter expects the newly trained augmented TCN-F3 "
                f"but found model={self.info.get('model')!r}, "
                f"feature_set={self.info.get('feature_set')!r}."
            )

        self.threshold = float(self.info.get("threshold", 0.5))

        # The current training protocol and the frozen GUI preprocessing agree:
        # hz=30, window=45, stride=22, max_gap_s=.1, max_missing=.2, smoothing 7/2.
        preprocessing = HERE / "model" / "preprocessing.json"
        if preprocessing.exists():
            self.cfg = Protocol(**json.loads(preprocessing.read_text()))
        else:
            raise FileNotFoundError(f"Missing GUI preprocessing config: {preprocessing}")

        if self.cfg.window != 45 or self.cfg.stride != 22:
            raise ValueError(
                "GUI preprocessing does not match the augmented training run: "
                f"window={self.cfg.window}, stride={self.cfg.stride}; "
                "expected 45/22."
            )

        architecture = json.loads((self.model_dir / "architecture.json").read_text())
        channels = int(architecture["channels"])
        width = int(architecture["width"])
        if channels != 22:
            raise ValueError(f"Expected 22 input channels for F3, got {channels}.")

        self.scaler = joblib.load(self.model_dir / "scaler.joblib")
        if not hasattr(self.scaler, "mean_") or not hasattr(self.scaler, "scale_"):
            raise ValueError("Invalid scaler.joblib: expected a fitted StandardScaler.")
        if self.scaler.mean_.shape != (22,) or self.scaler.scale_.shape != (22,):
            raise ValueError(
                f"Invalid scaler shape: mean={self.scaler.mean_.shape}, "
                f"scale={self.scaler.scale_.shape}"
            )
        if not np.isfinite(self.scaler.scale_).all() or np.any(self.scaler.scale_ <= 0):
            raise ValueError("Invalid scaler: non-finite or non-positive scale.")

        torch.set_num_threads(self.cfg.threads)
        self.net = TCN(channels, width).eval()
        state = torch.load(
            self.model_dir / "weights.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.net.load_state_dict(state)
        self.net.eval()

        # Metadata useful for reports/UI.
        self.reference_max_error = None


    @staticmethod
    def diagnose_sequence(sequence):
        """Return geometry/quality diagnostics before feature extraction.

        This is deliberately based on finite WORLD coordinates, not MediaPipe
        visibility.  It mirrors the quantities that frozen_pdml.features needs
        for F3 and makes failures auditable instead of collapsing to a generic
        ``invalid_leg_scale`` message.
        """
        seq = np.asarray(sequence, dtype=float)
        required = np.array([1,2,3,4,5,6,11,12,13,14,15,16], dtype=int)
        out = {
            "frames": int(len(seq)) if seq.ndim >= 1 else 0,
            "shape": list(seq.shape),
        }
        if seq.ndim != 3 or seq.shape[1:] != (17,3):
            out["error"] = "invalid_sequence_shape"
            return out

        req = np.isfinite(seq[:, required, :]).all(axis=2)
        out["required_joints_valid_fraction"] = float(req.all(axis=1).mean())
        out["required_joint_valid_fraction"] = {
            str(int(j)): float(req[:, i].mean()) for i, j in enumerate(required)
        }

        lengths = []
        for hip, knee, ankle in [(1,2,3),(4,5,6)]:
            leg = np.linalg.norm(seq[:, hip] - seq[:, knee], axis=1) + np.linalg.norm(seq[:, knee] - seq[:, ankle], axis=1)
            lengths.append(leg)
        length_series = np.nanmean(np.stack(lengths), axis=0)
        finite = np.isfinite(length_series)
        out["finite_leg_scale_fraction"] = float(finite.mean())
        out["median_leg_scale"] = float(np.nanmedian(length_series)) if finite.any() else None
        out["has_finite_leg_scale"] = bool(finite.any())
        out["finite_coordinate_fraction"] = float(np.isfinite(seq).mean())
        if not finite.any():
            out["failure_reason"] = "invalid_leg_scale"
        return out

    def scores(self, windows):
        """Match pdml.models.Temporal._array + _predict exactly."""
        a = np.asarray(windows[:, :, columns("F3")], dtype=np.float32)
        if a.ndim != 3 or a.shape[1:] != (self.cfg.window, 22):
            raise ValueError(
                f"Expected (N,{self.cfg.window},22) tensor, got {a.shape}"
            )
        if len(a) == 0:
            return np.empty(0, dtype=np.float32)
        if not np.isfinite(a).all():
            raise ValueError("Expected finite feature windows.")

        shape = a.shape
        flat = a.reshape(-1, shape[-1])

        # sklearn StandardScaler transform, same as pdml.models.Temporal._array.
        z = self.scaler.transform(flat).reshape(shape).astype("float32")

        parts = []
        with torch.inference_mode():
            for start in range(0, len(z), 256):
                xb = torch.from_numpy(z[start:start + 256])
                parts.append(torch.sigmoid(self.net(xb)).numpy())

        return np.concatenate(parts)

    def predict(self, sequence, timestamps):
        diagnostics = self.diagnose_sequence(sequence)
        if not diagnostics.get("has_finite_leg_scale", False):
            prediction = dict(
                status="unable_to_assess",
                reason="invalid_leg_scale",
                detail="No finite hip-knee-ankle leg scale was available in the WORLD-coordinate sequence.",
                threshold=self.threshold,
                windows=0,
                rejected_windows=0,
                score=None,
                label=None,
                diagnostics=diagnostics,
            )
            return prediction, [], [], np.empty((0, self.cfg.window, 22), np.float32), np.empty((0,22), np.float32)

        windows, accepted, rejected, signals = extract_windows(
            sequence, timestamps, self.cfg
        )
        scores = self.scores(windows)

        rows = []
        for window, score in zip(accepted, scores):
            rows.append(
                dict(
                    window,
                    end_s=window["end_frame"] / self.cfg.hz,
                    score=float(score),
                    label=int(score >= self.threshold),
                )
            )

        prediction = dict(
            status="ok" if len(scores) else "unable_to_assess",
            reason="" if len(scores) else "no_valid_windows",
            threshold=self.threshold,
            windows=len(rows),
            rejected_windows=len(rejected),
            score=None,
            label=None,
            diagnostics=diagnostics,
        )
        if len(scores):
            clip_score = float(scores.mean())
            prediction.update(
                score=clip_score,
                label=int(clip_score >= self.threshold),
            )

        return prediction, rows, rejected, windows, signals
