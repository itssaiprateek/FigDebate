"""Download all pinned runtime assets; verify newly built dataset provenance."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_environment import ROOT, configure


def main():
    configure()
    import os
    os.environ.pop('HF_HUB_OFFLINE', None)
    from huggingface_hub import snapshot_download
    from models.prepare_vision_model import main as vision
    from models.judge_model import JUDGE_MODEL_ID, JUDGE_MODEL_REVISION, default_judge_model_path
    from models.nli_model import NliVerifier
    from dataset.protocol import DATASET_ID, DATASET_REVISION
    from dataset.prepare_vflute import prepare
    from dataset.verify_upstream import verify, file_sha
    upstream = snapshot_download(DATASET_ID, repo_type='dataset', revision=DATASET_REVISION,
                                 local_dir=ROOT / '.cache/upstream-vflute')
    vision()
    snapshot_download(JUDGE_MODEL_ID, revision=JUDGE_MODEL_REVISION,
                      local_dir=default_judge_model_path())
    for model, revision in [
        ('mistralai/Mistral-7B-Instruct-v0.2', '63a8b081895390a26e140280378bc85ec8bce07a'),
        (NliVerifier.MODEL_ID, NliVerifier.REVISION),
    ]:
        snapshot_download(model, revision=revision,
                          allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])
    import json
    destination = ROOT / 'dataset/data/provenance/prepared' / ('vflute_' + DATASET_REVISION[:8])
    reuse = None
    try:
        prior = json.loads((destination / 'verification.json').read_text())
        hashes = {row['split']: row['local_file_sha256'] for row in prior['splits']}
        hashes['vflute_train_dev50'] = prior['dev50_file_sha256']
        if (prior['all_source_lineage_verified'] and prior['revision'] == DATASET_REVISION
                and all(file_sha(ROOT / 'dataset/data/processed' / (name + '.pkl')) == digest
                        for name, digest in hashes.items())):
            reuse = str(destination)
    except (OSError, ValueError, KeyError):
        pass
    if reuse is None:
        print('Validating/preparing all V-FLUTE splits; image checks may take several minutes.', flush=True)
        prepare(upstream_dir=upstream)
    import tempfile
    staging = Path(tempfile.mkdtemp(prefix='provenance-', dir=ROOT / '.cache')) / 'verified'
    print('Verifying every dataset row against the pinned original shards.', flush=True)
    result = verify(upstream, ROOT / 'dataset/data/processed', staging, reuse)
    if not result['all_source_lineage_verified']:
        raise RuntimeError('Dataset source verification failed; setup is not complete.')
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('provenance.jsonl', 'verification.json'):
        os.replace(staging / name, destination / name)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        from huggingface_hub.errors import GatedRepoError
        if isinstance(error, GatedRepoError):
            raise SystemExit('Dataset/model access required. Request V-FLUTE access on Hugging Face, '
                             'then run .venv/Scripts/hf.exe auth login (Windows) or '
                             '.venv/bin/hf auth login (Linux), and rerun bootstrap.py. '
                             'Alternatively set HF_TOKEN in your shell. Never commit your token.') from None
        raise
