"""Fail-fast local readiness check for the FigDebate runtime."""

import importlib
from importlib import metadata
import os
import platform
import sys


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
    "transformers": "4.46.3",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.0",
    "datasets": "5.0.1",
    "huggingface-hub": "0.26.2",
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


def main():
    failures = []
    print("FIGDEBATE ENVIRONMENT CHECK")
    print("=" * 40)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    if sys.version_info[:2] != (3, 11):
        failures.append("Python 3.11 is required.")

    broken_metadata = []
    for distribution in metadata.distributions():
        try:
            distribution.metadata["Name"]
            distribution.entry_points
        except Exception as error:
            broken_metadata.append(
                (str(getattr(distribution, "_path", "unknown")), str(error))
            )
    if broken_metadata:
        print("[FAIL] Installed-package metadata integrity")
        for path, error in broken_metadata:
            print(f"       {path}: {error}")
        failures.append(
            "Unreadable package metadata was found. Virtual environments "
            "inside cloud-synced folders must remain fully local."
        )
    else:
        print("[OK]   Installed-package metadata integrity")

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

    # A top-level Transformers import does not load the model implementation.
    # Exercise every API used by the two model loaders so incompatible
    # Torch/Transformers combinations fail here rather than during inference.
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
            BitsAndBytesConfig,
            LlavaForConditionalGeneration,
        )

        _ = (
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
            BitsAndBytesConfig,
            LlavaForConditionalGeneration,
        )
        print("[OK]   FigDebate Transformers model APIs")
    except Exception as error:
        print(f"[FAIL] FigDebate Transformers model APIs: {error}")
        failures.append(
            "The installed Torch/Transformers combination cannot load the "
            "FigDebate model APIs. Rebuild .venv from requirements.txt."
        )

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

    if failures:
        print("\nNOT READY")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nREADY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
