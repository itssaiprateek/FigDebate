"""Build and verify a source-only handoff archive from the current working tree."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {'agents', 'arbiter', 'comparators', 'config', 'dataset', 'docs',
               'engine', 'evaluation', 'models', 'scripts', 'tests', 'utils'}
ROOT_FILES = {'README.md', 'AGENTS.md', '.gitignore', 'requirements.txt',
              'requirements-diagnostics.txt', 'requirements-evaluation.txt', 'run_compact10.ps1'}
EXCLUDED_PARTS = {'__pycache__', '.venv', '.git', '.pytest_cache', 'cache', '.cache', '.env'}
EXCLUDED_SUFFIXES = {'.pyc', '.pyo', '.pkl', '.bin', '.safetensors', '.pt', '.pth', '.onnx', '.gguf', '.log', '.env'}


def source_files():
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts) or path.suffix in EXCLUDED_SUFFIXES:
            continue
        if relative.parts[:2] in (('dataset', 'data'), ('models', 'vision'), ('models', 'judge')):
            continue
        if len(relative.parts) == 1:
            if path.name in ROOT_FILES or path.suffix == '.py':
                yield path
        elif relative.parts[0] in SOURCE_DIRS:
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist/FigDebate_team_handoff.zip')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new archive path; existing handoffs are not overwritten.')
    output.parent.mkdir(parents=True, exist_ok=True)
    files = list(source_files())
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = dict(source='current_working_tree_including_uncommitted_source',
                    contains_model_weights=False, contains_datasets=False, contains_run_outputs=False,
                    file_sha256=hashes)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo('FigDebate/' + path.relative_to(ROOT).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
        archive.writestr('FigDebate/SOURCE_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive integrity check failed')
        for relative, digest in hashes.items():
            if hashlib.sha256(archive.read('FigDebate/' + relative)).hexdigest() != digest:
                raise RuntimeError('Archived source differs: ' + relative)
    print(json.dumps(dict(archive=str(output), files=len(files), bytes=output.stat().st_size,
                          sha256=hashlib.sha256(output.read_bytes()).hexdigest(), verified=True), indent=2))


if __name__ == '__main__':
    main()
