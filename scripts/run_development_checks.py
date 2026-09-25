"""Portable launcher for the fixed development cohort; delegates to the official runner."""
from pathlib import Path
import argparse
from datetime import datetime
import os
import runpy
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-only', action='store_true', help='Validate selection without model inference.')
    parser.add_argument('--run-dir', type=Path, help='New output directory; existing runs are not overwritten.')
    args = parser.parse_args()
    # PYTHONHASHSEED must be set before interpreter startup, as in the old launcher.
    environment = dict(os.environ, PYTHONHASHSEED='42', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_SYMLINKS_WARNING='1')
    environment.setdefault('HF_HUB_OFFLINE', '1')
    if os.environ.get('PYTHONHASHSEED') != '42':
        raise SystemExit(subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                                        env=environment).returncode)
    os.environ.update(environment)
    output = args.run_dir or root / 'outputs' / ('development_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    sys.path.insert(0, str(root))
    sys.argv = [str(root / 'run_figdebate.py'),
        '--dataset-split', 'vflute_train', '--num-samples', '10',
        '--sample-ids-file', str(root / 'config' / 'diagnostic10_ids.json'),
        '--selection-strategy', 'prefix', '--seed', '42', '--run-purpose', 'diagnostic',
        '--execution-mode', 'stagewise', '--batch-size', '10', '--hardware-profile', 'paper-8gb-review5',
        '--judge-mode', 'tribunal', '--judge-scope', 'all', '--feedback-mode', 'disabled',
        '--debate-mode', 'enabled', '--evidence-mode', 'enabled', '--candidate-mode', 'independent',
        '--semantic-bridge-mode', 'corroborated', '--tribunal-repair-mode', 'bounded',
        '--tribunal-audit-mode', 'baseline', '--run-dir', str(output.resolve())]
    if args.selection_only:
        sys.argv.append('--selection-only')
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
