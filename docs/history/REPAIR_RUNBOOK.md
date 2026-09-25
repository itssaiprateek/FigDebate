# One-system qualification runbook

Run from the actual FigDebate_main repository using its .venv Python. These
commands are a qualification workflow, not permission to claim a research gain.
Use a new output directory each time; resume only unchanged code/config/inputs.
Do not launch simultaneous GPU runs. Keep the three-phenomenon data unchanged.

## Local software gate

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s tests -p 'test_*.py'
& '.\.venv\Scripts\python.exe' -B check_environment.py
```

The latest development smoke manifest is
outputs/qualification_manifest_20260906_d.json. It contains one source-order
case per phenomenon and is not an adequately sized held-out benchmark.

```powershell
& '.\.venv\Scripts\python.exe' -B run_reproducible.py --dataset-split vflute_train_dev50 --num-samples 3 --sample-manifest outputs/qualification_manifest_20260906_d.json --selection-strategy prefix --selection-seed 42 --seed 42 --hardware-profile paper-8gb --judge-mode tribunal --judge-scope all --semantic-bridge-mode corroborated --feedback-mode disabled --run-dir outputs/NEW_full_tribunal
```

## Fairer control qualification

Use the same manifest, model revisions, seed and frozen source. For a base-only
run, set debate, judge and semantic bridge to disabled. For independent voting,
also set --control-mode independent_vote; for conventional two-round debate,
use --control-mode conventional_debate. Active controls require disabled hearing,
judge, bridge and feedback, and independent candidate mode.

--reuse-stage-dir accepts a completed source run's stage_checkpoints directory.
It reads compatible upstream outputs only and records reuse/rejection events;
it never reuses tribunal/hearing decisions across conditions. Source data and
source checkpoints are not overwritten. Check run_timing.json and trace runtime
accounting: executed-only counts exclude cached work. This is not evidence that
two methods have matched total compute.

evaluation.research_protocol generates the broader condition/seed matrix from a
locked manifest. Comparison requires exact declared treatment changes using
evaluation.compare_runs --allow-setting-change. Never use --allow-legacy to
present a confounded comparison as a controlled experiment.

## Fresh internal development and calibration

The audited manifests are separate selections of the existing vflute_train
records. Their provenance sidecars identify the audit and explicitly state that
the dataset was not changed. Development and calibration image groups exclude
dev50 and external validation/test groups. Final test remains untouched.

Training-split execution requires an explicit manifest. Calibration fitting must
use only the calibration manifest; tuning/extraction qualification must use the
development manifest. Freeze method and acceptance policies before final testing.

evaluation.confidence_calibration supports fit and apply on records.jsonl plus
run_config.json. Fit also requires the audited protocol; apply requires the saved
calibrator. It emits a separate probability and never changes predictions/gates.
It rejects overlapping image groups, wrong input identities and method changes.
No fitted calibrator is currently a project result.

## Human explanation review

```powershell
& '.\.venv\Scripts\python.exe' -B -m evaluation.human_review --records outputs/QUALIFIED_RUN/records.jsonl --split vflute_train_dev50 --output outputs/NEW_blinded_review
```

Give each rater their JSON, image files and instructions, not the private mapping.
Actual independent raters must enter their identities and ratings. The displayed
answer is shown solely to assess explanation faithfulness; it is not gold.
Report agreement and unassessable cases before adjudicating disagreements.

## Release limitations

Passing software tests, producing valid JSON or finishing a small GPU run does
not establish semantic accuracy. Require held-out paired image-group comparisons,
corrections and harms, calibrated uncertainty and independent human evidence.
Consult implementation_status.md and all 116 entries in change_tracker.json.
