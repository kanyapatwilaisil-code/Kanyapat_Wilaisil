# Augmented TCN-F3 integration patch

This patch is designed for the uploaded `video_gui` folder.

It does NOT replace the existing frozen model files. It adds parallel files:

- `inference_augmented.py`
- `backend_augmented.py`
- `app_augmented.py`

Run `app_augmented.py` to use the model produced by `train_augmented.py`.

Default model directory:
`../ผลการเทรน/augmented_f3_tcn_v2/model`

Override with:
`export PD_AUGMENTED_MODEL_DIR="/absolute/path/to/model"`

Required model files:
- model.joblib (not loaded at runtime; kept as training artifact)
- weights.pt
- scaler.joblib
- architecture.json
- model_info.json

The adapter uses the GUI's existing `frozen_pdml/features.py` for the same 22-channel F3 extraction and 45-frame/22-stride protocol. The trained TCN state and fitted StandardScaler are loaded from the new training output.

Start:
`python video_gui/app_augmented.py`

This leaves the old `app.py`, `backend.py`, and frozen model bundle untouched.
