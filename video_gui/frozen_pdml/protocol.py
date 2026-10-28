"""Frozen defaults. Changes create new feature/split/run identities."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json, hashlib
from .common import SUITE, dump_json

@dataclass(frozen=True)
class Protocol:
    seed: int = 42
    hz: int = 30
    window: int = 60
    stride: int = 30
    max_gap_s: float = 0.1
    max_missing: float = 0.2
    smooth_window: int = 7
    smooth_poly: int = 2
    visibility: float = 0.5
    outer_folds: int = 5
    inner_folds: int = 3
    epochs: int = 60
    patience: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    bootstrap: int = 2000
    threads: int = 2
    def dict(self): return asdict(self)
    @classmethod
    def read(cls, path=None):
        return cls(**json.loads(Path(path or SUITE/'configs/protocol.json').read_text()))

CLASSICAL = ['logistic','svm','random_forest','extra_trees','xgboost']
TEMPORAL = ['cnn_lstm','tcn']
FEATURE_SETS = ['F1','F2','F3']

def identity(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()[:16]

def candidates(model, smoke=False):
    if model=='dummy': return [{}]
    if model in ['logistic','svm']: out=[{'C':v} for v in [0.1,1.,10.]]
    elif model in ['random_forest','extra_trees']:out=[{'max_depth':d,'min_samples_leaf':l,'n_estimators':200} for d,l in [(4,2),(8,2),(None,4)]]
    elif model=='xgboost':out=[{'max_depth':d,'n_estimators':200} for d in [2,3,4]]
    elif model in TEMPORAL:out=[{'width':32},{'width':64}]
    else:raise ValueError(model)
    if smoke:
        out=out[:1]
        if 'n_estimators' in out[0]:out[0]['n_estimators']=10
    return out
