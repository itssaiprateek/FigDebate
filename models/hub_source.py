"""Prefer a pinned local Hub snapshot and go online only when it is absent."""

import os


def hub_cache_root():
    from huggingface_hub.constants import HF_HUB_CACHE

    return HF_HUB_CACHE


def cached_snapshot_or_hub(model_id, revision):
    repository = "models--" + str(model_id).replace("/", "--")
    snapshot = os.path.join(
        hub_cache_root(), repository, "snapshots", str(revision)
    )
    required = (
        "config.json", "model.safetensors.index.json", "tokenizer_config.json",
    )
    if all(os.path.isfile(os.path.join(snapshot, name)) for name in required):
        return snapshot, {"local_files_only": True}
    return model_id, {"revision": revision}
