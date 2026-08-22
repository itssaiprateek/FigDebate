FigDebate
=========

Official pipeline
-----------------

`run_figdebate.py` is the single official experiment runner.

It runs:

    image -> Agent 1 visual evidence
    caption -> Agent 2 immutable claim frame
    both -> Comparator V2 -> Arbiter -> selective two-level Debate
         -> deterministic Review Board -> prediction and paper artifacts

The default `stagewise` mode loads LLaVA once for the visual stage and Mistral
once for the language/Arbiter stage. This reduces repeated model loading while
keeping the two large models separate in GPU memory.

Quick verification
------------------

    .\.venv311\Scripts\activate
    python check_environment.py
    python -m unittest discover -s tests -p "test_*.py"
    python run_figdebate.py --num-samples 3 --selection-strategy stratified

Current reasoning flow
----------------------

    image -> structured visual evidence
    caption -> structured intended claim and relation
    relation candidates -> generic NLI diagnostic routing (never visual proof)
    initial Arbiter -> debate-need score
    Level 1 -> independent decision-grade evidence deliberation
    Level 2 -> independent targeted multimodal reinspection
    procedural feedback -> diagnostic question and debate routing, never a label
    deterministic Review Board -> accept only stronger current-image evidence
    final binary decision -> complete audit and paper artifacts

The previous label is hidden from both debate reviewers. Generic text NLI is
retained as a diagnostic signal but cannot create decision-grade visual
evidence. Explicit object-to-region text bindings are verified by a
deterministic relation checker before they can change a label.

Run integrity
-------------

An existing run directory cannot be reused accidentally. `--resume` verifies
the dataset, seed, modes, model revisions, feedback checksum, source checksum,
and evidence-ledger version before processing any missing samples.

Paired ablation comparison
--------------------------

    python -m evaluation.compare_runs --control outputs\CONTROL\predictions.csv --treatment outputs\TREATMENT\predictions.csv --output-dir outputs\comparison

The runner prints a unique run folder, for example:

    outputs\run_YYYYMMDD_HHMMSS

That folder contains `records.jsonl` (durable per-sample checkpoint),
`predictions.csv`, `run_config.json`, and `paper_assets`.

Resume an interrupted run
-------------------------

    python run_figdebate.py --num-samples 50 --selection-strategy stratified --run-dir outputs\run_YYYYMMDD_HHMMSS --resume

Evaluate a completed run
------------------------

    python evaluation\evaluate_predictions.py --input outputs\run_YYYYMMDD_HHMMSS\predictions.csv

The evaluator writes metrics, phenomenon breakdown, error analysis, and invalid
decision audit files into the same run folder by default.

Environment
-----------

Use Python 3.11. Install PyTorch for your CUDA version first, then install the
remaining packages from `requirements.txt`.
For the validated CUDA 12.1 environment, use `requirements-cu121.txt` after
installing Python 3.11. This pins torch 2.5.1 and torchvision 0.20.1 to the
combination previously verified on the RTX 4060 Laptop GPU.
`current_requirements.txt` is an older environment snapshot and is not the
installation contract.

The pinned `cross-encoder/nli-MiniLM2-L6-H768` model remains a CPU diagnostic
for text-relation uncertainty. Its output is never promoted to visual proof.

Paper protocol
--------------

Use separate run directories and the same seed/selection strategy:

    1. Baseline ablation: --debate-mode disabled --feedback-mode disabled
    2. FigDebate: --debate-mode enabled --feedback-mode disabled
    3. Development calibration: --feedback-mode calibrate
    4. Held-out procedural memory: --feedback-mode verified --verified-feedback-file FILE

Never calibrate on `vflute_test`. Compare matched runs with
`python -m evaluation.compare_runs`; report accuracy, balanced accuracy,
macro F1, debate corrections/harms, feedback corrections/harms, claim-contract
validity, directional evidence coverage, calibration, explanation diagnostics,
and stage/runtime profiles.

Feedback files created before memory schema version 2 contain legacy
gold-direction fields and must not be reused. Rebuild `calibrated_feedback.json`
with the current runner before a verified run.

Legacy baselines
----------------

Earlier Phase 3/Phase 4 scripts and results are retained for reproducibility.
They are not the official runner; see `legacy/README.md`.
