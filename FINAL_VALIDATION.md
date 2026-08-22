# FigDebate Final Validation

## 1. Repair The Local Runtime

The existing `.venv311` references a Python 3.11 installation that is no
longer present. Install Python 3.11, then create a clean environment without
overwriting historical folders:

```powershell
py -3.11 -m venv .venv_final
.\.venv_final\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cu121.txt
python check_environment.py
```

The validated hardware target is Python 3.11, CUDA 12.1, and an NVIDIA GPU
with at least 7 GB memory. The local reference machine is an RTX 4060 Laptop
GPU with torch `2.5.1+cu121`.

## 2. Run Contract Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Expected result: `Ran 101 tests` followed by `OK` before any model run.

## 3. Run A Three-Sample Smoke Test

```powershell
python run_figdebate.py --dataset-split vflute_train_dev50 --num-samples 3 --selection-strategy stratified --seed 42 --feedback-mode disabled --debate-mode enabled --evidence-mode enabled --run-dir outputs\final_smoke3
```

Check `metrics_summary.txt`, `debate_log.csv`, and `records.jsonl`. There must
be no invalid labels, exceptions, missing output artifacts, or evidence IDs
that refer to another sample.

## 4. Build New Procedural Memory

Do not reuse a pre-v2 feedback file.

```powershell
python run_figdebate.py --dataset-split vflute_train_dev50 --num-samples 50 --selection-strategy stratified --seed 42 --feedback-mode calibrate --debate-mode enabled --evidence-mode enabled --run-dir outputs\final_dev50_calibration
```

The output `outputs\final_dev50_calibration\calibrated_feedback.json` must
contain `procedural_case` entries and must not contain `verified_relation`.

## 5. Run Matched Validation Ablations

No debate or memory:

```powershell
python run_figdebate.py --dataset-split vflute_val --num-samples 100 --selection-strategy stratified --seed 42 --feedback-mode disabled --debate-mode disabled --evidence-mode enabled --run-dir outputs\final_val100_core
```

Selective debate:

```powershell
python run_figdebate.py --dataset-split vflute_val --num-samples 100 --selection-strategy stratified --seed 42 --feedback-mode disabled --debate-mode enabled --evidence-mode enabled --run-dir outputs\final_val100_debate
```

Selective debate plus procedural memory:

```powershell
python run_figdebate.py --dataset-split vflute_val --num-samples 100 --selection-strategy stratified --seed 42 --feedback-mode verified --verified-feedback-file outputs\final_dev50_calibration\calibrated_feedback.json --debate-mode enabled --evidence-mode enabled --run-dir outputs\final_val100_feedback
```

## 6. Generate Paired Comparisons

```powershell
python -m evaluation.compare_runs --control outputs\final_val100_core\predictions.csv --treatment outputs\final_val100_debate\predictions.csv --output-dir outputs\compare_core_vs_debate
python -m evaluation.compare_runs --control outputs\final_val100_debate\predictions.csv --treatment outputs\final_val100_feedback\predictions.csv --output-dir outputs\compare_debate_vs_feedback
```

## 7. Acceptance Gates

- Invalid prediction rate is zero.
- Debate harm is zero or lower than debate correction, with paired cases shown.
- Procedural-memory harm is zero or lower than its correction count.
- Memory match rate is selective; a return to the historical 97% rate is a
  failure of retrieval specificity.
- No generic NLI candidate is promoted to decision-grade evidence.
- Every accepted revision cites current-sample decision-grade evidence.
- Level 2 visual reinspection evidence is separately counted and identified;
  weak, negated, or entity-unbound observations remain non-decision-grade.
- Claim-contract failures and review-board statuses are reported, not hidden.
- Runtime comparison includes model loading and per-stage inference.
- Accuracy, balanced accuracy, macro F1, calibration, explanation diagnostics,
  and phenomenon breakdowns are present in each run folder.

Only after these validation comparisons are locked should `vflute_test` be run
once for the final paper table.
