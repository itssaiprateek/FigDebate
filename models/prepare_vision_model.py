"""Download and validate the pinned Qwen3-VL Agent 1 model."""

from __future__ import annotations

import json
from pathlib import Path

from models.vision_model import (
    VISION_MODEL_ARCHITECTURE,
    VISION_MODEL_FILES,
    VISION_MODEL_ID,
    VISION_MODEL_REVISION,
    local_vision_model_path,
)


def revision_metadata_path(model_root: Path) -> Path:
    return (
        model_root
        / ".cache"
        / "huggingface"
        / "download"
        / "config.json.metadata"
    )


def validate_model(model_root: Path) -> list[str]:
    failures = [
        f"missing {filename}"
        for filename in VISION_MODEL_FILES
        if not (model_root / filename).is_file()
    ]
    config_path = model_root / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            failures.append(f"unreadable config.json: {error}")
        else:
            architecture = (config.get("architectures") or [None])[0]
            if architecture != VISION_MODEL_ARCHITECTURE:
                failures.append(
                    f"architecture {architecture!r}, expected "
                    f"{VISION_MODEL_ARCHITECTURE!r}"
                )
    metadata_path = revision_metadata_path(model_root)
    if metadata_path.is_file():
        try:
            revision = metadata_path.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError) as error:
            failures.append(f"unreadable revision metadata: {error}")
        else:
            if revision != VISION_MODEL_REVISION:
                failures.append(
                    f"revision {revision!r}, expected {VISION_MODEL_REVISION!r}"
                )
    else:
        failures.append("missing pinned revision metadata")
    return failures


def main() -> int:
    model_root = local_vision_model_path()
    failures = validate_model(model_root)
    if failures:
        from huggingface_hub import snapshot_download

        print(
            f"Downloading {VISION_MODEL_ID}@{VISION_MODEL_REVISION} to "
            f"{model_root}"
        )
        model_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=VISION_MODEL_ID,
            revision=VISION_MODEL_REVISION,
            local_dir=str(model_root),
            allow_patterns=list(VISION_MODEL_FILES),
        )
        failures = validate_model(model_root)
    if failures:
        print("Qwen3-VL 4B Instruct preparation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"Qwen3-VL 4B Instruct is ready: {model_root}")
    print(f"Revision: {VISION_MODEL_REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
