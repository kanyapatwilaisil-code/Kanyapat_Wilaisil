"""Portable, CPU-only inference for the fixed final TCN-F3 (no pickle at runtime)."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn
from frozen_pdml.protocol import Protocol
from frozen_pdml.features import extract_windows, NAMES

HERE = Path(__file__).resolve().parent

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
            nn.ReLU(), nn.Dropout(.3))
        self.skip = nn.Conv1d(cin, width, 1) if cin != width else nn.Identity()
    def forward(self, x):
        return self.net(x) + self.skip(x)

class TCN(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.blocks = nn.Sequential(*[
            Residual(channels if i == 0 else width, width, d)
            for i, d in enumerate([1, 2, 4, 8])])
        self.head = nn.Linear(width, 1)
    def forward(self, x):
        return self.head(self.blocks(x.transpose(1, 2)).mean(-1)).squeeze(-1)

class FixedPredictor:
    def __init__(self):
        folder = HERE / "model"
        self.manifest = json.loads((folder / "manifest.json").read_text())
        for relative, expected in self.manifest["files"].items():
            if sha256(HERE / relative) != expected:
                raise ValueError("Model/preprocessing checksum mismatch: " + relative)
        self.cfg = Protocol(**json.loads((folder / "preprocessing.json").read_text()))
        if self.manifest["feature_names"] != NAMES or self.manifest["model"] != "tcn" or self.manifest["feature_set"] != "F3":
            raise ValueError("This app only supports the packaged TCN-F3.")
        self.threshold = float(self.manifest["threshold"])
        architecture = json.loads((folder / "architecture.json").read_text())
        torch.set_num_threads(self.cfg.threads)
        self.net = TCN(architecture["channels"], architecture["width"]).eval()
        with np.load(folder / "weights.npz", allow_pickle=False) as z:
            self.net.load_state_dict({k: torch.from_numpy(z[k].copy()) for k in z.files})
        with np.load(folder / "scaler.npz", allow_pickle=False) as z:
            self.mean, self.scale = z["mean"].copy(), z["scale"].copy()
        if self.mean.shape != (22,) or self.scale.shape != (22,) or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("Invalid saved scaler")
        fixture = np.load(folder / "reference_windows.npy", allow_pickle=False)
        expected = np.loadtxt(folder / "reference_scores.csv", delimiter=",", skiprows=1)
        actual = self.scores(fixture)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6,
                                   err_msg="Portable model does not reproduce saved reference scores")
        self.reference_max_error = float(np.max(np.abs(actual - expected)))

    def scores(self, windows):
        a = np.asarray(windows, dtype=np.float32).copy()
        if a.ndim != 3 or a.shape[1:] != (self.cfg.window, 22) or not np.isfinite(a).all():
            raise ValueError("Expected finite (N, window, 22) tensor")
        if not len(a):
            return np.empty(0, dtype=np.float32)
        # Match StandardScaler.transform's separate in-place float32 operations.
        a -= self.mean
        a /= self.scale
        with torch.inference_mode():
            return np.concatenate([
                torch.sigmoid(self.net(torch.from_numpy(a[i:i+256]))).numpy()
                for i in range(0, len(a), 256)])

    def predict(self, sequence, timestamps):
        windows, accepted, rejected, signals = extract_windows(sequence, timestamps, self.cfg)
        scores = self.scores(windows)
        rows = []
        for window, score in zip(accepted, scores):
            rows.append(dict(window, end_s=window["end_frame"] / self.cfg.hz,
                             score=float(score), label=int(score >= self.threshold)))
        prediction = dict(status="ok" if len(scores) else "unable_to_assess",
                          reason="" if len(scores) else "no_valid_windows",
                          threshold=self.threshold, windows=len(rows),
                          rejected_windows=len(rejected), score=None, label=None)
        if len(scores):
            prediction.update(score=float(scores.mean()), label=int(scores.mean() >= self.threshold))
        return prediction, rows, rejected, windows, signals
