"""Launch FigDebate with process-start determinism settings."""

from __future__ import annotations

import os
import subprocess
import sys


REQUIRED_ENVIRONMENT = {
    "PYTHONHASHSEED": "42",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
}


def main():
    requested_seed = "42"
    if "--seed" in sys.argv:
        position = sys.argv.index("--seed")
        if position + 1 < len(sys.argv):
            requested_seed = str(int(sys.argv[position + 1]))
    expected = dict(REQUIRED_ENVIRONMENT)
    expected["PYTHONHASHSEED"] = requested_seed
    if os.environ.get("FIGDEBATE_REPRO_EXEC") != "1" or any(
        os.environ.get(key) != value for key, value in expected.items()
    ):
        os.environ.update(expected)
        os.environ["FIGDEBATE_REPRO_EXEC"] = "1"
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
            check=False,
            env=os.environ.copy(),
        )
        raise SystemExit(completed.returncode)

    from run_figdebate import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()
