"""Fail-fast local readiness check for the FigDebate runtime."""

import argparse
import importlib
from importlib import metadata
import json
import os
import platform
import sys

from models.vision_model import (
    VISION_MODEL_ARCHITECTURE,
    VISION_MODEL_DIRECTORY,
    VISION_MODEL_FILES,
    VISION_MODEL_REVISION,
)
from models.judge_model import (
    JUDGE_MODEL_ARCHITECTURE,
    JUDGE_MODEL_DIRECTORY,
    JUDGE_MODEL_FILES,
    JUDGE_MODEL_REVISION,
)


REQUIRED_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "datasets": "datasets",
    "huggingface_hub": "huggingface-hub",
    "pandas": "pandas",
    "PIL": "pillow",
    "safetensors": "safetensors",
    "sklearn": "scikit-learn",
    "sentencepiece": "sentencepiece",
}
EXPECTED_VERSIONS = {
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "transformers": "5.15.0",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.0",
    "datasets": "5.0.1",
    "huggingface-hub": "1.27.0",
    "pandas": "3.0.5",
    "pillow": "12.2.0",
    "safetensors": "0.8.0",
    "scikit-learn": "1.9.0",
    "sentencepiece": "0.2.2",
}
REQUIRED_DATA = (
    "dataset/data/processed/vflute_train_dev50.pkl",
    "dataset/data/processed/vflute_val.pkl",
    "dataset/data/processed/vflute_test.pkl",
)


def check_control_plane(failures):
    """Verify tribunal modules and contracts without loading model weights."""
    print("\nPIPELINE CONTROL PLANE")
    checks = (
        ("engine.batch_runner", "StagewiseRunner"),
        ("engine.question_router", "build_question_plan"),
        ("engine.tribunal", "apply_tribunal_resolution"),
        ("engine.decision_trace", "append_decision_checkpoint"),
        ("utils.judge_parser", "parse_tribunal_review_response"),
        ("agents.visual_adapter", "AtomicVisualQuestionController"),
    )
    for module_name, attribute in checks:
        try:
            module = importlib.import_module(module_name)
            getattr(module, attribute)
        except Exception as error:
            print(f"[FAIL] {module_name}.{attribute}: {error}")
            failures.append(
                f"Unusable pipeline component: {module_name}.{attribute}"
            )
        else:
            print(f"[OK]   {module_name}.{attribute}")
def check_vision_files(root, failures):
    model_root = os.path.join(
        root, "models", "vision", VISION_MODEL_DIRECTORY
    )
    display_root = f"models/vision/{VISION_MODEL_DIRECTORY}"
    print("\nMANDATORY AGENT 1 MODEL")
    for filename in VISION_MODEL_FILES:
        path = os.path.join(model_root, filename)
        if os.path.isfile(path):
            print(f"[OK]   {display_root}/{filename}")
        else:
            print(f"[FAIL] {display_root}/{filename}")
            failures.append(f"Missing Agent 1 model file: {filename}")
    config_path = os.path.join(model_root, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, ValueError) as error:
            failures.append(f"Agent 1 config is unreadable: {error}")
        else:
            architecture = (config.get("architectures") or [None])[0]
            if architecture != VISION_MODEL_ARCHITECTURE:
                failures.append(f"Unexpected Agent 1 architecture: {architecture}")
            else:
                print(f"[OK]   Agent 1 architecture: {architecture}")
    metadata_path = os.path.join(
        model_root, ".cache", "huggingface", "download", "config.json.metadata"
    )
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            revision = handle.readline().strip()
        if revision != VISION_MODEL_REVISION:
            failures.append(
                "Agent 1 revision mismatch: "
                f"{revision} (expected {VISION_MODEL_REVISION})"
            )
        else:
            print(f"[OK]   Agent 1 revision: {revision}")
    else:
        failures.append("Agent 1 revision metadata is missing.")


def check_judge_files(root, failures):
    model_root = os.path.join(root, JUDGE_MODEL_DIRECTORY)
    print("\nOPTIONAL MULTIMODAL JUDGE")
    for filename in JUDGE_MODEL_FILES:
        path = os.path.join(model_root, filename)
        if os.path.isfile(path):
            print(f"[OK]   models/judge/Qwen3.5-4B/{filename}")
        else:
            print(f"[FAIL] models/judge/Qwen3.5-4B/{filename}")
            failures.append(f"Missing judge model file: {filename}")
    config_path = os.path.join(model_root, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, ValueError) as error:
            failures.append(f"Judge config is unreadable: {error}")
        else:
            architecture = (config.get("architectures") or [None])[0]
            if architecture != JUDGE_MODEL_ARCHITECTURE:
                failures.append(f"Unexpected judge architecture: {architecture}")
            else:
                print(f"[OK]   judge architecture: {architecture}")
    metadata_path = os.path.join(
        model_root, ".cache", "huggingface", "download", "config.json.metadata"
    )
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            revision = handle.readline().strip()
        if revision != JUDGE_MODEL_REVISION:
            failures.append(
                "Judge revision mismatch: "
                f"{revision} (expected {JUDGE_MODEL_REVISION})"
            )
        else:
            print(f"[OK]   judge revision: {revision}")
    else:
        failures.append("Judge revision metadata is missing.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate the pinned FigDebate environment."
    )
    parser.add_argument(
        "--check-judge",
        action="store_true",
        help="Also validate the optional pinned local Qwen judge files.",
    )
    args = parser.parse_args()
    failures = []
    print("FIGDEBATE ENVIRONMENT CHECK")
    print("=" * 40)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    if sys.version_info[:2] != (3, 11):
        failures.append("Python 3.11 is required.")

    for module_name, distribution_name in REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            print(f"[FAIL] {module_name}: {error}")
            failures.append(f"Missing or unusable module: {module_name}")
        else:
            version = metadata.version(distribution_name)
            expected = EXPECTED_VERSIONS[distribution_name]
            if version != expected:
                print(
                    f"[FAIL] {distribution_name}: {version} "
                    f"(expected {expected})"
                )
                failures.append(
                    f"Version mismatch: {distribution_name}=={expected} required."
                )
            else:
                print(f"[OK]   {distribution_name}: {version}")

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Torch CUDA: {torch.version.cuda}")
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU memory: {total_gb:.2f} GB")
            if total_gb < 7.0:
                failures.append(
                    "At least 7 GB of GPU memory is required for the "
                    "validated 4-bit stagewise runtime."
                )
        else:
            failures.append("A CUDA-capable GPU is required for the validated runtime.")
    except Exception:
        pass

    root = os.path.dirname(os.path.abspath(__file__))
    for relative_path in REQUIRED_DATA:
        path = os.path.join(root, relative_path)
        if os.path.exists(path):
            print(f"[OK]   {relative_path}")
        else:
            print(f"[FAIL] {relative_path}")
            failures.append(f"Missing dataset file: {relative_path}")

    check_vision_files(root, failures)
    check_control_plane(failures)

    if args.check_judge:
        check_judge_files(root, failures)

    if failures:
        print("\nNOT READY")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nREADY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
