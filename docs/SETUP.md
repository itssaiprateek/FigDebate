# Fresh clone setup

Supported inference hardware: Windows or Linux x86-64, NVIDIA CUDA GPU with
at least 8 GB VRAM and a CUDA 12.1-compatible driver. Reserve approximately
80 GB of free disk space for environments, models, dataset and caches.
macOS, AMD GPUs and CPU-only inference are not supported by this validated runtime.

Install Python 3.10 or newer with pip, clone the repository, and enter its folder.
Then run:

```text
python bootstrap.py
```

On Linux use `python3 bootstrap.py` if `python` is unavailable. The bootstrap
downloads Python 3.11 locally, creates `.venv`, installs dependencies from the cross-platform `requirements.lock`,
downloads all four runtime models and V-FLUTE, verifies dataset lineage, checks
CUDA and runs the software tests. It does not run inference. Internet access is
needed for setup. V-FLUTE requires Hugging Face access. Request access to
[ColumbiaNLP/V-FLUTE](https://huggingface.co/datasets/ColumbiaNLP/V-FLUTE), then
sign in using `.venv/Scripts/hf.exe auth login` (Windows) or
`.venv/bin/hf auth login` (Linux), and rerun bootstrap. Setup creates these
commands before downloading assets. Your normal user login is reused; credentials
are not stored in Git. Alternatively set `HF_TOKEN` in your environment. Never
commit tokens.
Rerun bootstrap after an interrupted download. Existing invalid dataset files
are rejected rather than silently replaced.

If Python 3.11 is already installed, `python3.11 setup_environment.py` (Linux)
or `py -3.11 setup_environment.py` (Windows) skips the managed-Python bootstrap.

Run from any installed Python using the launcher, which selects `.venv` itself:

```text
python run.py --dataset-split vflute_train --num-samples 10 --selection-strategy random --selection-seed 20260928 --seed 42 --run-purpose diagnostic --execution-mode stagewise --batch-size 10 --hardware-profile paper-8gb-review5 --judge-mode tribunal --judge-scope all --feedback-mode disabled --debate-mode enabled --evidence-mode enabled --candidate-mode independent --semantic-bridge-mode corroborated --tribunal-repair-mode bounded --tribunal-audit-mode baseline
```

Change `--selection-seed` for a different reproducible random cohort. Add
`--selection-only` to validate data selection without loading models.

## Local directory layout

| Directory | Purpose |
|---|---|
| `.runtime/`, `.tools/` | Bootstrap Python and environment manager |
| `.venv/` | Dependencies for this clone |
| `models/vision/`, `models/judge/` | Pinned Qwen weights |
| `.cache/huggingface/` | Pinned Mistral/NLI snapshots and dataset cache |
| `.cache/upstream-vflute/` | Original dataset shards and download hashes |
| `dataset/data/processed/` | Prepared splits |
| `dataset/data/provenance/prepared/` | Verification for this clone's files |
| `outputs/` | Run outputs |

All these directories are generated and ignored by Git. Do not copy `.venv`
between computers or move an installed checkout; recreate the environment at
the new location. Downloaded weights are not included in `git clone`.
The supported setup and run launchers isolate asset paths from stale shell
settings. No previous project directory is needed.

Software tests and readiness checks do not establish semantic accuracy.
The pipeline's decision logic and model revisions are unchanged by this setup.

Dependency maintainers: edit `requirements.txt`, regenerate `requirements.lock`
with the command recorded in its header (preserve the CUDA index directive),
and rerun setup and tests. Do not update model revisions during setup.

Fresh preparations preserve upstream image bytes. Historical experiments may
have used re-encoded images; do not treat their results as a paired comparison
with a newly prepared dataset. Labels and official split membership are unchanged.
Pinned software and seeds aid reproducibility; different GPU hardware can still
produce numerical differences.
