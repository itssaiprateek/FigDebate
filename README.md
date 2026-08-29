# FigDebate

Official pipeline
-----------------

`run_figdebate.py` is the single official experiment runner.

It runs:

    image -> Agent 1 visual evidence
    caption -> Agent 2 immutable claim frame
    both -> Evidence Comparator -> Arbiter -> selective two-level Debate
         -> deterministic Review Board -> optional independent Qwen Judge
         -> optional bounded Tribunal -> deterministic revision gate
         -> prediction and paper artifacts

The default `stagewise` mode loads Qwen3-VL 4B Instruct once for the visual stage and Mistral
once for the language/Arbiter stage. This reduces repeated model loading while
keeping the two large models separate in GPU memory.

The judge is disabled by default, so established runs keep the same model loads,
decision path, and predictions. Shadow and appellate load Qwen after the debate.
Mediated mode loads Qwen once before debate. Tribunal mode uses the same
label-blind question plan, reviews both agent answers, and permits at most one
targeted follow-up. Large GPU runtimes are unloaded between every stage.

Quick verification
------------------

    py -3.11 setup_environment.py
    .\.venv\Scripts\activate
    python check_environment.py
    python check_environment.py --check-judge
    python -m unittest discover -s tests -p "test_*.py"
    python run_figdebate.py --num-samples 3 --selection-strategy stratified

`setup_environment.py` also downloads and validates the mandatory pinned Agent
1 model at `models/vision/Qwen3-VL-4B-Instruct`. To prepare only that model later:

    python -m models.prepare_vision_model

Validate the already-downloaded optional judge as well:

    python check_environment.py --check-judge

On a new teammate system, download the optional judge once after activating
`.venv` (the weights remain outside Git):

    hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir models/judge/Qwen3.5-4B

Authentication is optional for these public models, but removes Hub rate-limit
warnings during the first download. The token is stored in the user profile,
never in this repository:

    hf auth login
    hf auth whoami

After the pinned Qwen3-VL model has been prepared, Agent 1 uses only that local
directory. Other pinned model snapshots are also resolved locally when cached.
This avoids repeat Hub metadata requests and unauthenticated warnings on later
runs while retaining online first-run setup.

Current reasoning flow
----------------------

    image -> short Qwen3-VL visual questions -> validated atomic answers
          -> deterministic Agent 1 evidence schema
    caption -> structured intended claim and relation
    relation candidates -> generic NLI diagnostic routing (never visual proof)
    initial Arbiter -> debate-need score
    Level 1 -> independent decision-grade evidence deliberation
    Level 2 -> independent targeted multimodal reinspection
    procedural feedback -> diagnostic question and debate routing, never a label
    deterministic Review Board -> accept only stronger current-image evidence
    optional Qwen judge -> independent raw-image and full-debate audit
    appellate gate -> require stronger cited decision-grade ledger evidence
    mediated mode -> Qwen issue map -> targeted agent checks -> verified gate
    tribunal mode -> issue map -> both witnesses -> review -> optional follow-up
                  -> independent verification -> deterministic Review Board
    final binary decision -> complete audit and paper artifacts

The previous label is hidden from both debate reviewers. Generic text NLI is
retained as a diagnostic signal but cannot create decision-grade visual
evidence. Explicit object-to-region text bindings are verified by a
deterministic relation checker before they can change a label.

Debate evidence safeguards
--------------------------

Agent 1 does not ask Qwen to author the complete evidence schema. A fixed,
caption-blind initial question plan collects the scene, entities, OCR, direct
facts, relationships, scene type, and visible symbolic cues separately. Python
validates each answer, permits one simplified retry, and assembles the public
schema. Model self-confidence is deliberately not accepted as calibrated
evidence. The same Agent 1 works when debate and the judge are disabled.

Level 2 review now asks Agent 1 one visual question at a time. Agent 1 records a
typed visual observation but never classifies it as entailment or contradiction.
Formatting failures, token-limit truncation, genuine absence, and valid
observations are logged as distinct outcomes. Only the failed field is retried.

Targeted recovery creates a fresh evidence generation instead of combining new
answers with disputed old observations. Ledger entries carry lifecycle status,
and only `ACTIVE` or `RECONFIRMED` evidence can support a decision. Agent 2's
support and conflict requirements are checked for opposing states and preserved
claim outcomes. Entity or theme word overlap remains diagnostic; the comparator
requires a subject-bound state or polarity cue for directional evidence.

The Arbiter keeps the public binary label contract but uses `INSUFFICIENT`
internally when evidence proves neither direction. An invalid Agent 1 review or
invalid Agent 2 requirement produces `NO_VISUAL_REVISION`; it cannot flip the
existing decision. The deterministic Review Board remains the final revision
gate.

Judge experiment modes
----------------------

Use shadow mode first. It records Qwen's independent verdict and disagreement
without changing a single prediction:

    python run_figdebate.py --num-samples 10 --judge-mode shadow --judge-scope escalated

After comparing the paired shadow run against the established baseline,
appellate mode can be tested on a development split:

    python run_figdebate.py --num-samples 10 --judge-mode appellate --judge-scope escalated

Mediated mode inserts Qwen before the debate. The agents receive only targeted
questions; Qwen's provisional vote and rationale remain hidden from them:

    python run_figdebate.py --num-samples 10 --judge-mode mediated --judge-scope escalated

Tribunal mode adds post-response review and at most one targeted follow-up:

    python run_figdebate.py --num-samples 10 --judge-mode tribunal --judge-scope escalated

The judge never sees the gold label or the current Arbiter label. A Qwen answer
cannot become evidence by itself. A proposed label change is accepted only if
the response is valid JSON, confidence is at least 0.75, every citation belongs
to the current sample, and the judge cites independently verified evidence whose
provenance-weighted strength exceeds the current direction. The validated Agent
1/Agent 2 exchange can add one cross-agent verified relation before tribunal
resolution; the judge itself cannot add evidence. The ordinary Review Board
still rejects the proposal when opposing verified evidence is stronger. See `docs/judge_architecture.md` and
`docs/tribunal_implementation.md` for the contracts and rollout protocol.

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
downloads and validates the required V-FLUTE splits, downloads and validates
the pinned Qwen3-VL 4B Instruct Agent 1 weights, runs `check_environment.py`, and
executes the unit suite. Other model weights are fetched from their pinned
Hugging Face revisions when required. Virtual
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
See `docs/agent1_qwen3vl.md` for the Agent 1 contract and acceptance protocol.

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
    models/          vision, language, NLI, and optional judge model wrappers
    tests/           contract and regression tests
    utils/           structured response parsers and decision utilities
    docs/            validation protocol and compatibility decisions

`run_figdebate.py` is the sole experiment entry point. `figdebate.py` exposes
the small programmatic API. Historical phase implementations and generated
root-level result files are intentionally excluded from the runtime tree.
