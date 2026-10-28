# Model card

Status: Research prototype

- Task: DNE healthy-control vs parkinson; no severity prediction.
- Source evaluation run: /home/cis/Desktop/PD_ML/experiments_v2/artifacts/runs/20260929T115842_DNE_full_958574
- Selected: tcn / F3
- Selection: subject-disjoint CV on all eligible development data, not external video.
- Final refit uses all eligible subjects; training scores are not held-out accuracy.
- Evaluation: see outer-fold predictions in source run.
- Sampling: 30 Hz; window 45 samples; stride 22.
- Scores: uncalibrated_probability; threshold 0.5; not clinical risk.
- Common joints differ between SMPL/h36m and MediaPipe; transfer is unvalidated until external evaluation.
- External video inventory may confound disease with activity/source; no clinical diagnostic claim.
- Offline noncausal smoothing; not zero-latency live inference.
- No absolute gait speed/stride length, toe clearance, or angular acceleration.
- Use only trusted joblib/PT bundles.

## Loading on another machine
Use matching package versions from environment.json. Add bundle/source_parent to PYTHONPATH if using the copied pdml package, or install/use the experiments_v2 source tree.
