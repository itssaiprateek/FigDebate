# Runbook

Run commands from the repository root. Use Python 3.11, the pinned packages in
`requirements.txt`, an NVIDIA CUDA GPU and sufficient disk space for the models.
The research environment used an RTX 4060 Laptop GPU with 8 GB VRAM. Laptop clock
variation can affect time-budget decisions; runtime is not a semantic metric.

## Environment and assets

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Obtain the team's verified dataset files and provenance sidecars. The default
layout is `dataset/data/processed/{vflute_train,vflute_train_dev50,vflute_val,vflute_test}.pkl`
with sidecars under `dataset/data/provenance/`. For a separate dataset location,
set `FIGDEBATE_DATA_ROOT` to its parent containing `data/processed`, not the
`processed` directory itself. Do not modify official labels or split membership.
The existing `dataset.prepare_vflute` and `dataset.verify_upstream` modules provide
preparation and provenance tooling; inspect their help before preparing new data.

Prepare the pinned vision model with `python -m models.prepare_vision_model`.
The Qwen judge uses revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`:

```powershell
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir models/judge/Qwen3.5-4B
hf download mistralai/Mistral-7B-Instruct-v0.2 --revision 63a8b081895390a26e140280378bc85ec8bce07a
hf download cross-encoder/nli-MiniLM2-L6-H768 --revision b95119ce93d3e065de6214e38cd4a97b0f2f2c6d
```

Mistral and NLI use pinned Hugging Face snapshots. Qwen directories default to
`models/vision/Qwen3-VL-4B-Instruct` and `models/judge/Qwen3.5-4B`.
`FIGDEBATE_MODEL_ROOT` can select the parent containing `vision` and `judge`.
Set `HF_HUB_OFFLINE=1` only after the required snapshots are available locally.
Keep tokens and local path settings in your environment, not committed scripts.

`python check_environment.py --check-judge` validates installed versions and
local model files. Its legacy dataset presence check expects repository-relative
PKLs even if `FIGDEBATE_DATA_ROOT` is configured; selection-only execution below
checks the actual configured dataset. The handoff deliberately preserves this
existing helper rather than changing unrelated runtime behavior.

## Checks before inference

```powershell
python -m unittest discover -s tests
python scripts/run_development_checks.py --selection-only
```

The portable launcher has no personal filesystem paths. It delegates to the
official runner with the existing fixed ten-case cohort and current tribunal
profile. `run_compact10.ps1` remains a compatibility wrapper and accepts
`-PythonExecutable`, `-SelectionOnly` and `-RunDir`.

## Bounded development run

```powershell
python scripts/run_development_checks.py --run-dir outputs/development_check
```

The directory must be new. This runs ten selected training examples with seed
42, stagewise execution, `paper-8gb-review5`, tribunal scope `all`, independent
candidates, corroborated bridges, bounded repair, baseline audit and feedback
disabled. It does not prove held-out accuracy. A full dataset run is a separate
team decision, not part of the cleanup checks.

For other cohorts, use `run_figdebate.py --help` and state all relevant modes.
Use the same manifest, images, captions, seed, model revisions and budgets for a
paired comparison, and always include `--feedback-mode disabled`. A code change
requires a new run; do not resume a pre-OCR run with the changed source.

## Results and reasoning evaluation

Keep `records.jsonl`, `predictions.csv`, `run_config.json`, `sample_manifest.json`,
`selected_ids.json`, completion/progress files, checkpoints and timing records.
Only a completed manifest establishes run completion. Empty or interrupted outputs
must not disappear from reported denominators.

Install optional scoring dependencies with `python -m pip install -r requirements-evaluation.txt`.
Follow [reasoning similarity](REASONING_SIMILARITY.md) to score initial, proposed
and final explanations against human references after inference. BERTScore is
similarity, not logical correctness. Check image faithfulness, roles, polarity,
modality and figurative meaning separately, alongside helpful/harmful changes.

The resumed full-pipeline OCR comparison remains outstanding. Preserve its original
control, interrupted candidate checkpoint and untested six-case cohort as research
evidence; do not report them as completed validation or reuse their artifacts in a
new-source run without the existing compatibility checks.
