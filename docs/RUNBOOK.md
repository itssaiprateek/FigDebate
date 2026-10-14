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
candidates, corroborated bridges, bounded repair, focused audit, grounded initial reasoning and feedback
disabled. It does not prove held-out accuracy. A full dataset run is a separate
team decision, not part of the cleanup checks.

For other cohorts, use `run_figdebate.py --help` and state all relevant modes.
Use the same manifest, images, captions, seed, model revisions and budgets for a
paired comparison, and always include `--feedback-mode disabled`. A code change
requires a new run; do not resume a pre-OCR run with the changed source.

## Reasoning and audit modes

As authorized on 2026-09-30, stagewise runs default to `--reasoning-mode grounded`
for a performance trial. This includes bounded explanation-completion checks.
Sequential execution retains baseline reasoning.

Stagewise tribunal runs using `paper-8gb-review5`, enabled debate and disabled
feedback default to `--tribunal-audit-mode focused-audit-1`. This includes the
aligned-reading protocol; aligned and focused audits are alternative modes,
not switches to stack. Other configurations retain baseline audit behavior.
The development launcher explicitly selects grounded reasoning and focused audit.

This is a user-authorized trial default, not a demonstrated accuracy improvement.
Earlier selected-case semantic results were mixed. For the previous control,
pass `--reasoning-mode completion --tribunal-audit-mode baseline` explicitly.
Keep feedback disabled. Use a fresh run directory and compare the same manifest
under both settings before attributing any change to the new modes.
See the [implementation and qualification report](semantic_protocol_implementation_20260928.md)
for prior results and limits; its original default descriptions are historical.

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

## Reliability qualification and semantic experiments (2026-09-30)

The current grounded path checks the assessment input budget before generation.
For oversized inputs it preserves the exact caption and every catalogue ID/text,
groups duplicate observations only with identical panel/region/speaker attachments,
and removes repeated fallible summaries. Relation scoring uses the same evidence
packet when needed. Inputs that still cannot fit are explicitly unresolved; an
identical oversized generation request is not retried. This is evidence-preserving
repacking, not a claim that all upstream prompt wording is unchanged.

Inline citations must refer to known catalogue IDs and agree with declared IDs.
Alternative-choice visual questions are routed as open questions. A proposer reason
at the old 480-character boundary fails the contract even if it ends in punctuation;
the existing single retry regenerates only the unfinished text fields, preserves
the other proposal fields, and still requires independent verification. The trace
retains the original value and patch. Final artifacts report binary prediction,
explanation completeness and label/reason alignment separately. A structurally
aligned or model-verified answer is not necessarily semantically correct.

The following options are experimental and do not replace the current defaults:

| Option | Mechanism |
| --- | --- |
| `--reasoning-mode compact` | Shorter evidence/meaning/relation/reason assessment schema. |
| `--tribunal-audit-mode questions-audit-1` | One open deciding question and pixel-grounded answer inside independent visual selection. |
| `--tribunal-audit-mode mapping-audit-1` | Explicit pictured situation, conventional connection and attributed property; mapping passed as a fallible hypothesis. |
| `--tribunal-audit-mode coherent-audit-1` | Audit candidate reasoning, roles, property, polarity and specific objections. |
| `--tribunal-audit-mode reinspect-audit-1` | Route an eligible cited visual objection to one fresh visual-witness inspection under bounded repair. |
| `--tribunal-audit-mode combined-audit-1` | Combine the four tribunal mechanisms. |

Reinspection uses the existing full-image witness and existing call/time/novelty
limits. It does not add an object detector, guarantee a useful crop, or establish
independence between models. Use `--tribunal-repair-mode bounded` to exercise it.
All tribunal experiments require the review5 profile, enabled debate and disabled
feedback. Experiment identity is bound into verification provenance. Keep the same
models, sample IDs, source images, seed, feedback setting and budget for paired arms;
use fresh output directories and proof caches. Gold labels and reference reasoning
are read only for evaluation after each prediction.

`evaluation.qualify_reasoning_completion` replays the initial stage using frozen
upstream outputs. `evaluation.qualify_grounded_gate` replays first review and the
actual acceptance gate; it does not exercise a repair hearing or the whole pipeline.
Its `direct` arm is a cheaper independent image/caption control, not a gate ablation.
Report generation requests, tokens, wall time, missing/invalid outputs, helpful and
harmful changes, and label-plus-reasoning adequacy separately. Rotation of arm order
reduces order bias but does not establish equal compute or deterministic GPU output.
Selected development cases do not establish held-out improvement. Never select the
best rerun or promote a correct-label answer with unsupported reasoning.

### Qualification results from this iteration

All results below are selected development diagnostics, not held-out performance.
The before-change source/patch and the initial test plan are retained under
`.cache/iteration_20260930_before/`. Every live arm has its source snapshot and
manifest. No gold labels or reference explanations were supplied to inference.

| Check | Observed result | Decision |
| --- | --- | --- |
| Four-case initial replay, grounded versus compact | Both 2/4 labels correct and 4/4 explanations complete; assistant reference review found 1/4 adequate label-plus-reason pairs in each. Mean stage time 15.01 versus 25.95 seconds. | Keep compact opt-in. |
| Previously oversized newspaper case | Complete assessment and scoring now fit; both tested formats finish in about 12 seconds. Historical initial stage failed after 182 seconds. Label remains wrong. | Reliability recovery only; historical versus replay timing is not a controlled end-to-end speedup. |
| Four-case first-review/gate comparison | Focused, questions, mapping and coherent arms each ended 0/4 correct: one harmful acceptance, two rejected correct proposals and one clipped invalid proposal. Direct image/caption control got 2/4 correct. | No semantic mode promoted. |
| Field-only completion follow-up | Same first proposal byte-for-byte on case 918; previous full retry stayed clipped, new bounded field retry completed, passed independent verification and produced a correct accepted explanation consistent with the reference. | Keep recovery fix; one mechanism regression does not establish general accuracy. |

The four gate arms used 17 generation requests each across four cases; the direct
control used four. GPU inference was serial; brief CPU-only validation and selection
work overlapped part of the replay, so wall times are descriptive, not an isolated
microbenchmark. Mean times were 58.86s focused, 62.84s questions, 59.24s mapping,
63.06s coherent and 10.24s direct. The first gate comparison predates the final
field-only completion fix; do not present those four-way counts as a rerun of the
final source. All failures remain in the records. Completion recovery took 79.14s
and six calls because independent verification could now execute, versus 36.52s
and two calls for the invalid proposal. No timeout or failed generation request
occurred in the 20-comparison replay, but four proposal outputs were unusable.

Inspect `outputs/iteration_initial_v2_20260930/qualification_summary.json`,
`outputs/iteration_gate_20260930/qualification_summary.json` and
`outputs/iteration_completion_followup_20260930/qualification_summary.json` with
their corresponding `records.jsonl`. The earlier interrupted
`outputs/iteration_initial_20260930` is retained and excluded from completed-run
counts. Qualitative reasoning assessments are assistant reviews of references and
images, not independent human ratings or a numeric embedding similarity score.

The final automated suite completed 733 tests with one pre-existing expected
failure (`test_scoring_cannot_condition_on_assessment`); it remains an unresolved
scoring-independence qualification issue. The suite is not semantic accuracy evidence.
Current defaults remain grounded initial reasoning and focused audit under the
already-authorized review5 trial configuration. New semantic variants remain opt-in.

The one-case full-pipeline combined-mode smoke run
(`outputs/iteration_combined_fullpath_20260930`) completed all stages with a complete
final explanation and an explicit UNRESOLVED/UNVERIFIED justification status. Its
correct bagel proposal was still rejected. Reinspection did not execute: the
unchanged case guard required a 165-second reserve and only 161.49 seconds remained.
The route passed scripted production-routing tests, but its live effectiveness is
unqualified. Do not loosen the budget or report this smoke test as an accuracy gain.
See the run's `qualification_summary.json` and complete trace for the exact failure.


## 2026-10-02 condition verification implementation

The default V5 indexed-source mapping path can now recover an unterminated final reason with one 160-token field-only retry. Complete bindings and unmatched roles are frozen and revalidated; partial OCR, partial bindings and other malformed records are not promoted into complete evidence.

New `condition-audit-1` and `condition-repair-1` modes are **experimental**, not defaults. They preserve the full caption, compare actual/asserted state pairs, derive redundant verdict fields from one condition table, bind final explanations to the audit, and optionally regenerate one faulty candidate before a fresh audit. Keep feedback disabled. Existing case budgets and two-hearing maximum remain.

The paired four-case check produced 1/4 final labels in both arms: the experiment removed one harmful change but lost one useful correction. The mapping recovery passed a separate live check. An oracle-assisted repair succeeded but is excluded from accuracy. Automated validation: 746 tests, one pre-existing expected scoring-independence failure. See `docs/CONDITION_ITERATION.md` and `outputs/condition_pair_20261002/qualification_report.html` before using the new options. Defaults remain grounded initial reasoning and the existing focused review5 trial.


## 2026-10-02 causal qualification follow-up

The opt-in `reason-first-audit-1` changes only explanation-field order and its ordering instruction in the existing proposer/relation contracts. It retains all fields, independent verification, source/image bindings, validators, budgets and gate. **Not promoted:** the four-case real-review comparison finished 1/4 correct in both arms, with one helpful and one harmful revision each. The candidate improved one proposal label but not its final answer or explanation; both still accepted unsupported jersey originality. Keep existing focused defaults and feedback disabled.

The 34 isolated diagnostic stage evaluations locate output-order sensitivity, literal-proof demands, internally contradictory response fields, and comparison errors that persist despite correct OCR. Their label improvements are not pipeline accuracy gains. Full automated validation: 750 tests, one pre-existing expected scoring-independence failure, no unexpected failures. See `docs/CAUSAL_QUALIFICATION.md` and `outputs/reason_first_pair_20261002/qualification_report.html` for the evidence, per-case reasoning reviews, limitations and replay command.


## Structural routing revision — 2026-10-02

The maintained tribunal passes typed visual questions directly, rejects missing/invalid requests without generic substitution, and resolves CLI/direct-runner reasoning defaults through one policy. Source identity preserves the full caption but does not certify upstream issue tags. Unqualified tags retain broad neutral visual coverage; concrete missing-fact requests remain targeted. Feedback stays disabled.

See [structural refactor results](STRUCTURAL_REFACTOR_RESULTS_20261002.html). The new canonical-assessment and authored-example candidates failed promotion: they are evaluation-only, not normal CLI modes. The restricted comparison evaluator also remains in evaluation tooling because translation was not qualified. Existing focused-audit-1 remains the default under the review5 tribunal configuration. No additional recovery budget, judge, model download or initial-arbiter change was introduced in this revision.

Bounded evaluation entry points are `evaluation.qualify_grounded_gate`, `evaluation.qualify_question_transport`, and `evaluation.qualify_comparison_translation`. They require retained source artifacts and new output directories. Full dataset runs remain user-controlled; prior checkpoints must pass the existing source/config compatibility checks.
