"""Launch using this clone's environment: python run.py [runner arguments]."""
import subprocess
import sys
from project_environment import ROOT, configure, python_path


def main():
    configure()
    python = python_path()
    if not python.exists():
        print('Environment missing. Run Python 3.11 setup_environment.py first.', file=sys.stderr)
        return 2
    return subprocess.call([str(python), str(ROOT / 'run_reproducible.py'), *sys.argv[1:]], cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
