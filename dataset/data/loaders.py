
import os
import pickle
from io import BytesIO
from PIL import Image

EXPECTED_SPLITS = [
    "vflute_train", "vflute_val", "vflute_test",
    "mmsd2_train", "mmsd2_val", "mmsd2_test",
    "capcon_train", "capcon_val", "capcon_test",
    "vflute_train_dev50",
]

# The file is retained under its original name for backwards compatibility,
# while the public split name makes its train-derived provenance explicit.
SPLIT_FILES = {
    "vflute_train_dev50": "dev_split",
}

def _package_root():
    here = os.path.abspath(os.path.dirname(__file__))
    if os.path.isdir(os.path.join(here, "data", "processed")):
        return here
    parent = os.path.abspath(os.path.join(here, os.pardir))
    if os.path.isdir(os.path.join(parent, "data", "processed")):
        return parent
    raise FileNotFoundError("Could not locate data/processed relative to loaders.py")

BASE = _package_root()
PROCESSED_DIR = os.path.join(BASE, "data", "processed")

def _load(split_name):
    if split_name not in EXPECTED_SPLITS:
        raise ValueError(f"Unknown split: {split_name}")
    stored_name = SPLIT_FILES.get(split_name, split_name)
    path = os.path.join(PROCESSED_DIR, f"{stored_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)

def decode_image(image_bytes):
    return Image.open(BytesIO(image_bytes)).convert("RGB")

def load_vflute_train(): return _load("vflute_train")
def load_vflute_val(): return _load("vflute_val")
def load_vflute_test(): return _load("vflute_test")
def load_mmsd2_train(): return _load("mmsd2_train")
def load_mmsd2_val(): return _load("mmsd2_val")
def load_mmsd2_test(): return _load("mmsd2_test")
def load_capcon_train(): return _load("capcon_train")
def load_capcon_val(): return _load("capcon_val")
def load_capcon_test(): return _load("capcon_test")
def load_vflute_train_dev50(): return _load("vflute_train_dev50")

# Legacy alias for older scripts. New experiments should use the explicit name.
def load_dev_split(): return load_vflute_train_dev50()

def load_split(split_name): return _load(split_name)

def load_all():
    return {split: _load(split) for split in EXPECTED_SPLITS}

def by_phenomenon(samples, phenomenon):
    return [s for s in samples if s.get("phenomenon") == phenomenon]
