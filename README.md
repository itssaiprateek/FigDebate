# FigDebate

Official pipeline
-----------------

`run_figdebate.py` is the single official experiment runner.

It runs:

    image -> Agent 1 visual evidence
    caption -> Agent 2 immutable claim frame
    both -> Evidence Comparator -> Arbiter -> selective two-level Debate
         -> deterministic Review Board -> prediction and paper artifacts

The default `stagewise` mode loads LLaVA once for the visual stage and Mistral
once for the language/Arbiter stage. This reduces repeated model loading while
keeping the two large models separate in GPU memory.

Quick verification
------------------

    py -3.11 setup_environment.py
    .\.venv\Scripts\activate
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

Python 3.11, a CUDA-capable NVIDIA GPU, and at least 7 GB of GPU memory are
required for the validated stagewise runtime. `requirements.txt` is the only
dependency contract. It pins the CUDA 12.1 PyTorch build and every direct
Python dependency used by the project.

Windows setup:

    py -3.11 setup_environment.py
    .\.venv\Scripts\activate

Linux setup:

    python3.11 setup_environment.py
    source .venv/bin/activate

The setup script creates only `.venv`, installs the pinned dependencies,
downloads and validates the required V-FLUTE splits, runs
`check_environment.py`, and executes the unit suite. Model weights are fetched
from their pinned Hugging Face revisions on the first pipeline run. Virtual
environments, model caches, and processed datasets are machine-local and are
deliberately not committed to Git.

When the folder is trusted and opened in VS Code, the committed workspace task
runs this setup automatically. The Python extension selects `.venv` and
activates it in newly opened terminals. A setup fingerprint prevents repeated
downloads when Python, the datasets, and `requirements.txt` are unchanged.
Python 3.11 must already be installed, and the full pipeline still requires a
compatible NVIDIA GPU with at least 7 GB of VRAM.

To rebuild only the dataset later:

    python -m dataset.prepare_vflute --force

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

Project structure
-----------------

    agents/          visual grounding and claim extraction
    arbiter/         final language-model decision component
    comparators/     evidence-aware image-caption comparison
    dataset/         loaders, split manifest, and deterministic preparation
    engine/          orchestration, debate, feedback, evidence, and review
    evaluation/      metrics, audits, comparisons, and paper artifacts
    models/          vision, language, and NLI model wrappers
    tests/           contract and regression tests
    utils/           structured response parsers and decision utilities
    docs/            validation protocol and compatibility decisions

`run_figdebate.py` is the sole experiment entry point. `figdebate.py` exposes
the small programmatic API. Historical phase implementations and generated
root-level result files are intentionally excluded from the runtime tree.
