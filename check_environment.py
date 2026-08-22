"""Fail-fast local readiness check for the FigDebate runtime."""

import importlib
import os
import platform
import sys


REQUIRED_MODULES = (
    "torch", "transformers", "accelerate", "bitsandbytes", "datasets",
    "pandas", "PIL", "sentencepiece",
)
OPTIONAL_MODULES = ("sklearn",)
REQUIRED_DATA = (
    "dataset/data/processed/dev_split.pkl",
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

    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            print(f"[FAIL] {module_name}: {error}")
            failures.append(f"Missing or unusable module: {module_name}")
        else:
            version = getattr(module, "__version__", "available")
            print(f"[OK]   {module_name}: {version}")

    for module_name in OPTIONAL_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            print(
                f"[INFO] {module_name}: optional; internal evaluation "
                "fallback will be used"
            )
        else:
            print(
                f"[OK]   {module_name}: "
                f"{getattr(module, '__version__', 'available')}"
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
