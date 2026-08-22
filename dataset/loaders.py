
import os
import pickle
from io import BytesIO
from PIL import Image

EXPECTED_SPLITS = [
    "vflute_train_dev50", "vflute_val", "vflute_test",
]

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
    path = os.path.join(PROCESSED_DIR, f"{split_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)

def decode_image(image_bytes):
    return Image.open(BytesIO(image_bytes)).convert("RGB")

def load_split(split_name): return _load(split_name)
