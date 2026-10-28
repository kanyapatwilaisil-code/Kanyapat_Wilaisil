"""Maintainer utility: export only the existing final TCN; never retrains it."""
from pathlib import Path
import json
import shutil
import numpy as np
from inference import HERE, sha256, FixedPredictor

def main():
    import joblib
    import torch
    bundle = HERE.parent / "artifacts/final_models/20260930T032616_final_bundle_02f2dc"
    info = json.loads((bundle / "bundle.json").read_text())
    winner = json.loads((bundle / "estimator/model_info.json").read_text())
    assert not info["smoke"]
    assert winner["model"] == "tcn" and winner["feature_set"] == "F3"
    for file, key in [("estimator/model.joblib", "model_sha256"),
                      ("preprocessing.json", "preprocessing_sha256"),
                      ("feature_schema.json", "feature_schema_sha256")]:
        assert sha256(bundle / file) == info[key], file
    dest = HERE / "model"
    dest.mkdir(exist_ok=True)
    scaler = joblib.load(bundle / "estimator/scaler.joblib")
    state = torch.load(bundle / "estimator/weights.pt", map_location="cpu", weights_only=True)
    np.savez_compressed(dest / "weights.npz", **{k: v.numpy() for k, v in state.items()})
    np.savez_compressed(dest / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    for file in ["preprocessing.json", "feature_schema.json", "reference_windows.npy",
                 "reference_scores.csv", "model_card.md", "environment.json"]:
        shutil.copy2(bundle / file, dest / file)
    shutil.copy2(bundle / "estimator/architecture.json", dest / "architecture.json")
    task = HERE.parents[1] / "parkinson_gait_app/video/models/pose_landmarker_full.task"
    shutil.copy2(task, dest / task.name)
    snapshot = bundle / "source_parent/pdml"
    for name in ["features.py", "common.py", "protocol.py"]:
        shutil.copy2(snapshot / name, HERE / "frozen_pdml" / name)
    files = [p for p in dest.iterdir() if p.is_file() and p.name != "manifest.json"]
    files += [HERE / "frozen_pdml" / n for n in ["features.py", "common.py", "protocol.py"]]
    manifest = dict(model="tcn", feature_set="F3", threshold=winner["threshold"],
                    feature_names=info["feature_names"], source_bundle=bundle.name,
                    source_model_sha256=info["model_sha256"],
                    source_weights_sha256=sha256(bundle / "estimator/weights.pt"),
                    source_scaler_sha256=sha256(bundle / "estimator/scaler.joblib"),
                    calibrated=False, device="cpu",
                    files={str(p.relative_to(HERE)): sha256(p) for p in files})
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    predictor = FixedPredictor()
    print("Portable model reference maximum error:", predictor.reference_max_error)

if __name__ == "__main__":
    main()
