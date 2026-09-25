# Final-run context — 24 September 2026

This is a context handoff, not a new experiment or an implementation change. It consolidates the recent turns of `final gods run`, the closing comparisons in `gods run 2`, earlier project context, the current working tree, and saved run artifacts. Historical test results below were read, not rerun in this task.

## Current assessment

FigDebate's execution, provenance and output validation have improved substantially. The tribunal has not demonstrated a reliable net accuracy benefit. It can misinterpret the source claim, select misleading observations, repeat a semantic error across calls, reject sound corrections, or certify harmful ones. Successful JSON and source identity checks do not establish semantic truth.

The immediate unresolved question is which first incorrect transformation causes downstream failure. The last task proposed controlled stage replacement to isolate this; it did not report a completed stage-replacement study. Another broad prompt rewrite is not yet supported by the evidence.

## Working copy and standing scope

- Active repository: `figdebate_release_candidate` inside the enclosing workspace. The enclosing workspace itself is not a Git repository.
- Branch at inspection: `tribunal-stability-20260913`.
- HEAD: `ab2ff4e` — Preserve tribunal proof identity and add bounded precedent feedback.
- There are 33 modified tracked files, plus numerous untracked production modules, tests, reports and runs. HEAD is not the current implemented system. A checkout of HEAD alone would omit the compact proposer and V5 implementation currently present as untracked files.
- `research/` in the enclosing workspace contains experiments, before snapshots, scripts and rejected variants. These are not automatically production features.
- Model/data/environment paths in the current launch script point to `C:/Users/Sai Prateek/Desktop/FigDebate_sent/figdebate final/`, outside this checkout. Preserve that distinction when reproducing runs.

Standing instructions in `AGENTS.md`:

1. Keep the initial arbiter unchanged unless the user explicitly changes scope.
2. Fix shared mechanisms; never add production answer rules keyed to sample IDs, captions, image hashes or gold labels. Provenance checks and named regression fixtures are allowed.
3. Keep feedback disabled. Bounded tribunal repair is a separate mechanism and can remain enabled.
4. Use bounded targeted checks through the real review and gate. Full dataset evaluations remain user-run unless explicitly requested.
5. Separate execution reliability, helpful corrections, harms, feedback contribution and runtime. Selected development checks are not held-out accuracy.

## Current configuration

The latest completed diagnostic used:

| Setting | Value |
|---|---|
| Runner | `run_figdebate.py` |
| Dataset | `vflute_train`, diagnostic purpose |
| Execution | stagewise, batch size 10 |
| Hardware profile | `paper-8gb-review5` |
| Initial vision | Qwen3-VL-4B-Instruct |
| Language / arbiter runtime | Mistral-7B-Instruct-v0.2 |
| Tribunal | Qwen3.5-4B, V5 compact proposer |
| Judge scope | all |
| Debate / evidence | enabled / enabled |
| Candidates / semantic bridge | independent / corroborated |
| Tribunal repair / audit | bounded / baseline |
| Feedback | disabled |
| Hardware recorded | RTX 4060 Laptop GPU |

The selected profile limits individual judge generation to 120 seconds and cumulative judge work per case to 240 seconds, with a 6,144 total-token budget. These are not end-to-end sample wall-time limits: other agents, witness work, model loading and export add time. The plain CLI defaults are not identical to this selected experimental configuration; in particular, the judge is disabled by default.

Larger/different model trials exist in research, including 9B and 27B work and Gemma-family trials. They did not replace the production Qwen3.5-4B. Selected frozen-role comparisons do not isolate pure model capability or establish end-to-end superiority.

## Directory and execution map

| Location | Responsibility |
|---|---|
| `run_figdebate.py`, `config/` | CLI, split/sample selection, configuration, provenance, resume and exports |
| `dataset/` | Load samples and images; preserve evaluation reference separation |
| `models/` | Vision, language, judge and diagnostic NLI runtimes |
| `agents/` | Visual testimony, caption claim extraction and multimodal review entry points |
| `comparators/` | Compare claim requirements and available evidence; identify review needs |
| `arbiter/` | Initial binary decision; protected from this improvement scope |
| `engine/batch_runner.py` | Stagewise scheduling, model residency, checkpointed hearings and finalization |
| `engine/evidence_ledger.py`, claim modules | Evidence identities/lifecycle and source claim representation |
| `engine/simple_judge.py` | Compact independent candidate proposal |
| `engine/evidence_review_v5.py` | Visual selection, role mapping, relation judgment and challenge |
| `engine/tribunal.py`, `semantic_bridge_verifier.py`, `review_board.py` | Case-bound verification and deterministic revision acceptance |
| `engine/tribunal_repair.py`, `question_router.py` | One bounded, targeted follow-up and reuse of valid unchanged obligations |
| `engine/result_store.py`, integrity/provenance modules | Durable records, source identity, resume and export safeguards |
| `evaluation/`, `tests/` | Scoring, diagnostics and mechanism tests |
| `runs/`, `docs/` | Saved experiment artifacts and change/qualification reports |

The active flow is:

1. Select samples, save run identity and keep labels/reference explanations for evaluation.
2. Agent 1 gathers caption-blind atomic scene, object, text and relationship observations; assemble validated visual testimony.
3. Agent 2 extracts caption meaning and claim requirements. Source identity is preserved, but semantic correctness is still fallible.
4. Build evidence records and compare them to claim requirements. Generic text NLI is diagnostic, not visual proof.
5. Produce the initial arbiter decision; route uncertainty to optional targeted witnesses.
6. The compact proposer sees the original image, full caption and admissible observation catalogue. It outputs interpreted assertion, decisive reason, evidence IDs and SUPPORT/CONFLICT/UNRESOLVED. It does not receive Agent 2's structured interpretation in its proposal packet, although language information participates in later bridge construction.
7. A separate visual call selects up to four deciding observations. Software binds IDs to exact catalogue text.
8. Mapping binds source participants/scopes to those observations. This call explicitly receives no image.
9. A relation call independently assesses the full caption and image with selected evidence and mappings.
10. The challenge evaluates the argument and alternatives. Current baseline input includes the verifier decision and proposer interpretation/reason, but does not explicitly name the proposer relation separately.
11. The deterministic gate checks source/image/evidence binding, admissibility, relation agreement and audit resolution. A rejected proposal preserves the existing decision.
12. Eligible failures may receive one budgeted follow-up. Tribunal-only repairs can freeze the candidate and reassess its proof; new witness evidence can support a fresh proposal. This is independent of disabled feedback.
13. Persist final predictions, traces, timing and completion artifacts; score against references afterward.

## Performance baseline

Five saved random development batches, read from their `tribunal_quality.json` files:

| Run | Initial correct | Final correct | Helpful changes | Harmful changes | Wall time |
|---|---:|---:|---:|---:|---:|
| `random10_20260922_205440` | 7/10 | 7/10 | 0 | 0 | 49m 36s |
| `random10_20260923_183831` | 6/10 | 5/10 | 0 | 1 | 31m 52s |
| `random10_20260923_191620` | 8/10 | 7/10 | 0 | 1 | 31m 40s |
| `random10_20260923_195201` | 4/10 | 4/10 | 0 | 0 | 28m 02s |
| `random10_20260923_202138` | 3/10 | 4/10 | 1 | 0 | 32m 29s |
| Total | 28/50 | 27/50 | 1 | 2 | |

These are saved development results, not a new benchmark of the current code. The September 23 mechanical revalidation of the same saved outputs changed final correctness to 28/50, with two helpful and two harmful accepts. That replay generated no new model outputs and must not replace the original run result in reporting.

Latest complete fresh rerun: `worst10_retest_20260924_102021`, the same ten samples as the last row above:

- Initial/final: 3/10 -> 3/10; macro F1 0.2308.
- Four correct-label change proposals rejected; two harmful change proposals blocked.
- Three wrong same-label confirmations; one semantic abstention.
- Nine tribunal repair follow-ups, no accepted changes.
- Zero failed reviews and zero cases with failed verification execution in the quality report.
- Two cases with invalid verification output; terminal outcomes include two incomplete audits and two contradictory judgments.
- Initial visual diagnostics still record three failed answers in one case: one truncation and two overlong responses. Therefore, “no failed tribunal executions” must not become “no failures anywhere.”
- Wall time 1,783.9075 seconds = 29m 44s, down 8.4% from 32m 29s.
- Tribunal time 680.33 seconds = 11m 20s, a subset of wall time.

The lost accepted correction previously benefited from a formatting retry of a semantically contradictory relation. The new code correctly routes that contradiction away from formatting repair, but its semantic follow-up does not recover the useful correction. The result is an engineering improvement with a real correct-label regression on this batch. The older accepted rationale was also weak; label correctness alone does not prove sound reasoning.

Earlier history must stay separate: `gods run 2` compared an earlier training pair at 16/20 final against a different validation pair at 12/20 final. Both pairs gained two correct answers from the tribunal. The cohorts, phenomenon mix, profiles and code differed, so the raw accuracy drop was not a controlled causal comparison. September 14's separate 50-case run had provisional checkpoint results of 35/50 -> 35/50 with two corrections and two harms, alongside an export failure. Do not merge that cohort with the later random 50.

## Recent changes and disposition

- September 14–17: completion/export recovery, more explicit outcomes and runtime accounting, source snapshots, bounded image residency and streamed/durable results. V5 introduced blind evidence selection, source-bound mapping/relation proof and bounded diagnostic repair. Earlier context-overflow, timeout and export defects motivated these changes.
- September 20: fuller review context and actual image/chat token accounting; tribunal repair became separately selectable with feedback disabled. Optional process audit remained unqualified and off by default. Some older context-delivery claims describe the pre-compact proposer and should not be assumed to describe the current proposal packet.
- September 21: retained source-interval, selection repair and trace improvements. Rejected the larger semantic dialogue overhaul: selected checks showed no accuracy benefit, more harmful proposals and much more work.
- September 22: integrated the four-field compact proposer; corrected restored evidence-location handling; removed challenge character ceilings; bounded alternative-field clarification; preserved source expressions, candidate interpretation and complete repair diagnostics. Added audit-only reassessment. Expanded audit and premise-accounting variants were withheld after regressions.
- September 23: removed remaining active text clipping, aligned schemas/validators, strengthened raw-response integrity, handled explicit `None` alternatives without erasing uncertainty, removed a coarse family-label veto for otherwise valid V5 proofs, froze candidates in eligible tribunal-only repairs, separated semantic contradiction from formatting failure, and made repair budgeting account for the remaining proof.
- Recorded September 23 validation: 176 focused tests passed. This is historical software evidence, not a fresh test run here or a semantic-accuracy guarantee.
- September 24 prompt qualification: 24 stage evaluations on seven diagnostic cases, 25 generations. Explicit targets and demonstrations sometimes helped locally but also introduced false objections, worse explanations and inconsistent outputs. No proposed prompt replacement was promoted. These experiments did not constitute fresh end-to-end pipeline comparisons.

## Remaining issues, in priority order

1. **Claim meaning can change downstream.** Preserving source bytes does not preserve negation, desire versus actuality, comparison direction, speaker or scope. Relevant: claim modules, `simple_judge.py`, `evidence_review_v5.py`.
2. **Visual interpretations can masquerade as observations.** Inferred roles and repeated descriptions can contaminate later reasoning. Provenance establishes origin, not truth. Relevant: visual agents and evidence ledger.
3. **Relation reasoning remains unreliable.** Missing support can become contradiction; topic overlap can become support; figurative meaning may be treated as unavailable or require literal depiction. Relevant: V5 relation stage and interpretation instructions.
4. **Audit target and bookkeeping remain problematic.** Candidate and verifier verdicts are not explicitly separated in the baseline payload. False objections and contradictory alternative/condition fields block useful proofs. An explicit-target experiment is not yet a qualified production fix.
5. **Gate inputs are fallible semantic judgments.** Correctly enforcing a model-generated proof cannot guarantee the proof is true. Prior harmful accepts prevent simply relaxing acceptance.
6. **Selection and mapping can lose deciding context.** Coverage is self-certified; mapping has only selected observation text. The causal impact of these choices has not been isolated.
7. **Repair often repeats rather than resolves the error.** Frozen-candidate paths cannot replace a wrong candidate. Latest nine follow-ups yielded no correction; matched repair-disabled ablations are needed for causal benefit.
8. **Model errors may be correlated.** Proposal, selection, relation and challenge calls use the same judge model. Separate calls do not establish independent errors. Controlled capability and error-overlap tests remain outstanding.
9. **Runtime remains substantial and variable.** Multiple sequential judge stages, witness work and occasional retries contribute. Low GPU clocks were observed historically, but their cause is unresolved; do not claim a proven hardware diagnosis. Latest tribunal time accounts for only part of wall time.
10. **Qualification and release gaps remain.** No representative untouched evaluation of the finalized system, no established semantic non-regression guarantee, and many current files remain uncommitted. Old and new paths coexist; documentation must distinguish active configuration from legacy and research code.

## What the previous tests did not establish

Prompt experiments reused saved upstream agents and proofs, tested some stages separately, and did not pass every fresh proposal through fresh verification and real follow-up. Most few-shot variants changed instructions as well as examples; examples were text descriptions, not image demonstrations. Evidence-order sensitivity, repeatability, clean model capability and matched repair-off benefits remain insufficiently tested.

The next bounded diagnostic proposed in the last task is to substitute one independently checked input at a time: neutral visual observations, preserved caption meaning, selection/mapping, then sound versus defective candidate arguments with comparable labels. Run the remaining real verification/gate and follow-up paths. Keep these interventions evaluation-only and exclude their outputs from ordinary accuracy claims. This identifies the earliest damaging transformation before selecting another production change.

## Proposed completion criteria

These are planning criteria, not achieved guarantees or new authorization for a full run:

1. Preserve an exact current baseline and qualify each retained change through the real path, including useful corrections and harmful controls.
2. Choose a documented final configuration with bounded runtime and explicit failure outcomes; stop adding unqualified mechanisms.
3. Have the user run the agreed broader comparison on fixed manifests, separating development diagnostics from untouched evaluation.
4. Report initial versus final accuracy, helpful/harmful changes, abstentions, execution failures, repair contribution, explanations and wall time honestly, even if the tribunal shows no net gain.
5. Freeze the release source/configuration and document limitations. Project completion does not require claiming the semantic problem has been solved.

## Evidence to reopen first

- `docs/TRIBUNAL_GENERAL_FIXES_20260923.md`
- `docs/PROMPT_CHANGE_QUALIFICATION_20260924.md`
- `docs/PROMPT_RESEARCH_REVIEW_20260924.md`
- `docs/TRIBUNAL_REMAINING_20260922.md`
- `docs/COMPACT_JUDGE_INTEGRATION_20260922.md`
- `runs/worst10_retest_20260924_102021/{run_config.json,tribunal_quality.json,run_timing.json,records.jsonl}`
- `../research/tribunal_general_20260923/replay_after.json`
- `../research/prompt_qualification_20260924/`

No production source, configuration or model was changed, and no inference or tests were launched while preparing this context handoff.
