from pathlib import Path
import hashlib, json, platform, importlib.metadata, datetime, subprocess
import numpy as np

SUITE = Path(__file__).resolve().parents[1]
ROOT = SUITE.parent
DATA = ROOT / 'CARE-PD/assets/datasets'
ARTIFACTS = SUITE / 'artifacts'
SEED = 42

def dump_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=convert), encoding='utf-8')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def environment():
    packages = {}
    for p in ['numpy','pandas','scipy','scikit-learn','joblib','matplotlib','mediapipe','opencv-python','xgboost','nbformat']:
        try: packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: packages[p] = None
    return {'python':platform.python_version(), 'platform':platform.platform(), 'packages':packages,
            'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'code_sha256':{str(p.relative_to(SUITE)):sha256(p) for p in sorted((SUITE/'pdml').glob('*.py'))}}

def new_run(tag, base=None):
    import re, uuid
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', tag): raise ValueError('Use a simple run tag')
    path = Path(base or ARTIFACTS/'runs') / (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+tag+'_'+uuid.uuid4().hex[:6])
    path.mkdir(parents=True, exist_ok=False)
    dump_json(path/'environment.json', environment())
    # Preserve the exact implementation with each run, not just its hash.
    import shutil
    shutil.copytree(SUITE/'pdml', path/'source', ignore=shutil.ignore_patterns('__pycache__'))
    return path
