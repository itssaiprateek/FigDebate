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


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_DIR = PROJECT_ROOT / ".venv"
STATE_FILE = ENV_DIR / ".figdebate-environment.json"
SETUP_SCHEMA_VERSION = 1
REQUIRED_DATA = (
    PROJECT_ROOT / "dataset" / "data" / "processed" / "vflute_train_dev50.pkl",
    PROJECT_ROOT / "dataset" / "data" / "processed" / "vflute_val.pkl",
    PROJECT_ROOT / "dataset" / "data" / "processed" / "vflute_test.pkl",
)


def environment_python() -> Path:
    if sys.platform == "win32":
        return ENV_DIR / "Scripts" / "python.exe"
    return ENV_DIR / "bin" / "python"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def pin_environment_locally() -> None:
    """Prevent OneDrive Files On-Demand from offloading virtualenv files."""
    if sys.platform != "win32" or not ENV_DIR.exists():
        return
    if "onedrive" not in str(PROJECT_ROOT).lower():
        return
    completed = subprocess.run(
        ["attrib", "+P", "-U", str(ENV_DIR), "/S", "/D"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Could not mark .venv as Always Available on this device. "
            "Move the repository outside OneDrive or disable Files On-Demand "
            "for .venv before continuing."
        )


def environment_is_usable() -> bool:
    python = environment_python()
    if not python.exists():
        return False
    try:
        completed = subprocess.run(
            [str(python), "--version"],
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
    with (PROJECT_ROOT / "requirements.txt").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_is_current() -> bool:
    if not environment_is_usable() or not STATE_FILE.exists():
        return False
    if not all(path.exists() for path in REQUIRED_DATA):
        return False
    try:
        with STATE_FILE.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    return (
        state.get("setup_schema_version") == SETUP_SCHEMA_VERSION
        and state.get("python") == "3.11"
        and state.get("requirements_sha256") == requirements_checksum()
    )


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
        help="Do not download and prepare the required V-FLUTE splits.",
    )
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 11):
        print(
            "Python 3.11 is required. Run this script with py -3.11 on "
            "Windows or python3.11 on Linux.",
            file=sys.stderr,
        )
        return 2

    if not args.recreate and environment_is_current():
        print("FigDebate environment is already current.")
        print(f"Python: {environment_python()}")
        return 0

    if args.recreate:
        remove_environment()
    elif ENV_DIR.exists() and not environment_is_usable():
        print("Existing .venv is not usable; rebuilding it.")
        remove_environment()

    if not environment_is_usable():
        print(f"Creating {ENV_DIR}")
        venv.EnvBuilder(with_pip=True).create(ENV_DIR)
    pin_environment_locally()

    python = str(environment_python())
    run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([python, "-m", "pip", "install", "-r", "requirements.txt"])
    if not args.skip_data:
        run([python, "-m", "dataset.prepare_vflute"])
    run([python, "check_environment.py"])
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

    pin_environment_locally()

    write_environment_state()

    print("\nFigDebate environment is ready.")
    print(f"Python: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
