"""One-time setup from any installed Python 3.10+ on Windows/Linux x86-64."""
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    if platform.system() not in ('Windows', 'Linux') or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise SystemExit('Supported inference platform: Windows/Linux x86-64 with an NVIDIA CUDA GPU.')
    env = dict(os.environ)
    env['UV_PYTHON_INSTALL_DIR'] = str(ROOT / '.runtime')
    env['UV_CACHE_DIR'] = str(ROOT / '.cache/uv')
    def run(args):
        subprocess.run([str(a) for a in args], cwd=ROOT, env=env, check=True)
    run([sys.executable, '-m', 'pip', 'install', '--upgrade', '--target', ROOT / '.tools', 'uv==0.12.19'])
    uv = ROOT / '.tools/bin' / ('uv.exe' if os.name == 'nt' else 'uv')
    run([uv, 'python', 'install', '3.11.16'])
    python = subprocess.check_output(
        [str(uv), 'python', 'find', '--system', '--managed-python', '3.11.16'],
        cwd=ROOT, env=env, text=True).strip()
    run([python, ROOT / 'setup_environment.py', *sys.argv[1:]])



if __name__ == '__main__':
    main()
