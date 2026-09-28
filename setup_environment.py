"""Create and validate the single supported FigDebate environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import venv

from project_environment import configure


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_DIR = PROJECT_ROOT / ".venv"
STATE_FILE = ENV_DIR / ".figdebate-environment.json"
SETUP_SCHEMA_VERSION = 4


def environment_python() -> Path:
    if sys.platform == "win32":
        return ENV_DIR / "Scripts" / "python.exe"
    return ENV_DIR / "bin" / "python"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def environment_is_usable() -> bool:
    python = environment_python()
    if not python.exists():
        return False
    try:
        completed = subprocess.run(
            [str(python), "-c", "import sys; raise SystemExit(sys.version_info[:2] != (3, 11))"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return completed.returncode == 0


def requirements_checksum() -> str:
    digest = hashlib.sha256()
    with (PROJECT_ROOT / "requirements.lock").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_environment_state() -> None:
    state = {
        "setup_schema_version": SETUP_SCHEMA_VERSION,
        "python": "3.11",
        "requirements_sha256": requirements_checksum(),
    }
    temporary = STATE_FILE.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, STATE_FILE)


def _remove_readonly(function, path, _error_info) -> None:
    """Allow removal of copied or OneDrive-marked virtualenv entries."""
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    function(path)


def remove_environment() -> None:
    resolved_root = PROJECT_ROOT.resolve()
    resolved_env = ENV_DIR.resolve()
    if resolved_env.parent != resolved_root or resolved_env.name != ".venv":
        raise RuntimeError(f"Refusing to remove unexpected path: {resolved_env}")
    if ENV_DIR.exists():
        last_error = None
        for attempt in range(3):
            try:
                shutil.rmtree(ENV_DIR, onerror=_remove_readonly)
                return
            except PermissionError as error:
                last_error = error
                time.sleep(attempt + 1)
        raise RuntimeError(
            "Windows could not remove the old .venv after three attempts. "
            "Close terminals using that environment, close Explorer windows "
            "inside .venv, pause OneDrive syncing briefly, and rerun with "
            "--recreate."
        ) from last_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create FigDebate's canonical Python 3.11 environment."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild only the project-local .venv directory.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Install and validate dependencies without running unit tests.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Install dependencies only; skip all assets and do not mark setup ready.",
    )
    args = parser.parse_args()
    configure()

    if sys.version_info[:2] != (3, 11):
        print(
            "Python 3.11 is required. Run this script with py -3.11 on "
            "Windows or python3.11 on Linux.",
            file=sys.stderr,
        )
        return 2

    if args.recreate:
        remove_environment()
    elif ENV_DIR.exists() and not environment_is_usable():
        print("Existing .venv is not usable; rebuilding it.")
        remove_environment()

    if not environment_is_usable():
        print(f"Creating {ENV_DIR}")
        venv.EnvBuilder(with_pip=True).create(ENV_DIR)

    python = str(environment_python())
    run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([python, "-m", "pip", "install", "-r", "requirements.lock"])
    if args.skip_data:
        print("Dependencies installed; asset download skipped. Setup is incomplete.")
        return 0
    run([python, '-c',
         "import torch; "
         "assert torch.cuda.is_available(), 'An NVIDIA GPU with a compatible driver is required'; "
         "assert torch.cuda.get_device_properties(0).total_memory >= 7 * 1024**3, 'Use an NVIDIA GPU with at least 8 GB VRAM'; "
         "x = torch.ones(1, device='cuda'); print('CUDA execution verified:', (x + 1).item())"])
    run([python, "scripts/prepare_assets.py"])
    run([python, "check_environment.py", "--check-judge"])
    run([python, "-m", "pip", "check"])
    if not args.skip_tests:
        run([
            python,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ])

    write_environment_state()

    print("\nFigDebate environment is ready.")
    print(f"Python: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
