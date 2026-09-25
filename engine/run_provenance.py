"""Archive exact source bytes for a run, including dirty and new modules."""
import hashlib
import json
from pathlib import Path
import zipfile


def snapshot_source(root, directory):
    root, directory = Path(root), Path(directory)
    manifest_path = directory / 'source_manifest.json'
    archive_path = directory / 'source_snapshot.zip'
    paths = list(root.glob('*.py'))
    for folder in ('agents', 'arbiter', 'comparators', 'dataset', 'engine', 'evaluation', 'models', 'utils', 'tests', 'config'):
        paths.extend(p for p in (root / folder).rglob('*') if p.is_file()
            and p.suffix in {'.py', '.json'} and not any(x in p.relative_to(root).parts for x in ('data', '__pycache__')))
    paths.extend(p for name in ('requirements.txt', 'pyproject.toml', 'AGENTS.md') if (p := root / name).is_file())
    content = {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(set(paths))}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = {'schema_version': '1', 'files': hashes, 'source_manifest_sha256': digest}
    if manifest_path.exists() or archive_path.exists():
        if not manifest_path.exists() or not archive_path.exists():
            raise ValueError('Incomplete source archive; retain it and use a fresh run directory')
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Source archive differs from current code; do not resume this run')
        with zipfile.ZipFile(archive_path) as archive:
            if set(archive.namelist()) != set(hashes) or any(hashlib.sha256(archive.read(name)).hexdigest() != h for name, h in hashes.items()):
                raise ValueError('Source archive integrity mismatch')
        return manifest
    temporary = directory / 'source_snapshot.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in content.items():
            archive.writestr(name, value)
    temporary.replace(archive_path)
    from engine.result_store import atomic_json
    atomic_json(manifest_path, manifest)
    return manifest
