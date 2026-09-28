"""Repository-local assets shared by setup and launchers (standard library only)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def configure():
    # Keep normal user authentication separate from the repository-local cache.
    default_auth_home = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface'))
    os.environ.setdefault('HF_TOKEN_PATH', str(default_auth_home / 'token'))
    # Supported launchers deliberately isolate this checkout from stale shell paths.
    os.environ['FIGDEBATE_DATA_ROOT'] = str(ROOT / 'dataset')
    os.environ['FIGDEBATE_MODEL_ROOT'] = str(ROOT / 'models')
    os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
    os.environ['HF_HUB_CACHE'] = str(ROOT / '.cache' / 'huggingface' / 'hub')
    os.environ['HF_DATASETS_CACHE'] = str(ROOT / '.cache' / 'huggingface' / 'datasets')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    # HTTP downloads also work on networks where the optional Xet transport stalls.
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')


def python_path():
    return ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
