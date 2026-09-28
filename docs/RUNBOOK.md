# Runbook

Run commands from the repository root. Use Python 3.11, the pinned packages in
`requirements.txt`, an NVIDIA CUDA GPU and sufficient disk space for the models.
The research environment used an RTX 4060 Laptop GPU with 8 GB VRAM. Laptop clock
variation can affect time-budget decisions; runtime is not a semantic metric.

## Environment and assets

Follow [SETUP](SETUP.md) and run `python bootstrap.py` on a fresh clone.
Use `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Linux
for the direct Python commands below. `python run.py` selects that environment
automatically for pipeline runs.

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
