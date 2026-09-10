"""Order-independent deterministic execution helpers."""

from __future__ import annotations

import hashlib
import os
import random


def derived_seed(global_seed, sample_id, stage, attempt=0):
    payload = f"{int(global_seed)}|{sample_id}|{stage}|{int(attempt)}"
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    ) % (2**31 - 1)


def seed_stage(global_seed, sample_id, stage, attempt=0):
    """Reset every available RNG for one sample-stage boundary."""
    seed = derived_seed(global_seed, sample_id, stage, attempt)
    from engine.runtime_accounting import set_scope
    set_scope(sample_id, stage, attempt)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def deterministic_environment():
    """Describe settings that must be applied before Python for strict runs."""
    return {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
    }
