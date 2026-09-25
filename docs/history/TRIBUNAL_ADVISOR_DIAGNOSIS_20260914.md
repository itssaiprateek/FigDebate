# Focused tribunal, advisor and export diagnosis

Scope: second offline audit of `dev50_reliability_20260914_111440`. Timing optimization is excluded. Production code is unchanged. The only executable addition is a scratch-file reproduction of the existing exporter failure; no model inference was run.

## What the evidence establishes

- Final checkpoints give 31 abstentions. Twenty-nine persisted through two hearings; 23 final abstentions supplied no counter-interpretation. Eight of these abstaining cases had an incorrect initial answer; 23 retained a correct answer. Abstention count alone therefore does not measure correction opportunity.
- All 50 claim graphs, including all 31 abstentions, used `UNDECOMPOSED_SOURCE` with `UNQUALIFIED_USE_FULL_CAPTION`. This is a deliberate source-preservation fallback in `engine/claim_graph.py`, not evidence of a transport failure. It protects against paraphrase drift but leaves the tribunal to distinguish background reference, evaluation, idiom and necessary factual conditions itself. Its causal contribution is a hypothesis supported by the outputs, not an isolated intervention result.
- Examples of failed interpretation include treating 'close to the add' as spatial proximity to an advertisement, asking for a visible cow for 'pass the cow', demanding explicit economic data for an inflation image, and demanding printed emotion for evaluative captions. Other cases describe apparent opposition (startled figures versus a calm meeting) but still return UNRESOLVED. These are qualitative diagnoses of the written reasoning; not all 31 abstentions are proven avoidable.
- The two accepted harms show that the same semantic distinctions also fail inside verification. Weather: narrow current-condition text is used against a broad negative forecast evaluation. Flash's bike: the joke's effect on the reader is confused with the object's usefulness. This explains why more willingness to propose does not necessarily improve net accuracy.

## Feedback cannot currently perform the requested advisor role

`agents/multimodal_judge.py::_precedent_followup` only proceeds when the baseline relation is UNRESOLVED. It skips a binary proposal as `baseline_not_semantically_unresolved`. Thus it did not challenge the confidently wrong weather proposal.

`engine/tribunal_feedback.py::retrieve` requires the caption's predicted type to match the precedent types. Flash's bike was classified `literal`, with `speaker_scope` as a structural feature; retrieval returned no precedents. This is a brittle eligibility boundary: an upstream interpretation error suppresses the mechanism that might correct it.

The library contains synthetic methodological guidance, not validated empirical precedents. Retrieval is based on a few broad structural features. The guidance goes to a new proposer attempt, never to the independent verifier. A wrong verifier reading therefore cannot be repaired simply by adding more proposer guidance.

Of 22 attempts, 18 remained UNRESOLVED. The four binary guided attempts were:

| Case | Guided result | Reference relationship | Selected? | Precedent applies? |
|---|---|---|---|---|
| 1406 | CONFLICT | Correct confirmation | No | False |
| 2173 | CONFLICT | Wrong confirmation | No | False |
| 2240 | CONFLICT | Harmful change | No | False |
| 4430 | CONFLICT | Correct confirmation | Yes | False |

Selection checks proof validity, not whether the precedent applied. That is not inherently wrong—a fresh review may discover sound evidence despite irrelevant advice—but the result must not be attributed to successful precedent application. None of these attempts produced a corrective proposal. The issue is therefore not just an over-strict gate discarding useful feedback corrections.

## Recommended advisor design to test

Use one bounded, non-authoritative advisor assessment for relevant uncertainty or a proposed change, rather than only for abstention. Do not make eligibility depend solely on the predicted humor/metaphor/sarcasm label. Route by the unresolved operation: ambiguous reading, conflicting speaker roles, comparison direction, desired versus actual outcome, incomplete visual coverage or subjective evaluation.

The advisor should return a concrete issue, the source span and image evidence involved, competing readings, a proposed resolving observation/check, and remaining uncertainty. A generic reminder to reconsider is insufficient. Keep the complete caption immutable; structured readings are hypotheses alongside the source, not replacements for it.

Before proposing, the tribunal uses this assessment to explain its selected reading and the decisive relation. For verification, supply a neutral check question and source references rather than the advisor's recommended verdict. The verifier must establish its own relation. Following a rejected proof, target the failed component and explicitly assess the objection; do not regenerate the whole case and silently forget the disagreement. An advisor cannot create visual evidence, supply an authoritative answer or bypass unresolved critical proof errors.

Require a competing-reading analysis for a proposed label change. In particular, distinguish the reader's reaction from the depicted object's property, current weather from a multi-day forecast, intent from result, and a character's statement from established fact. The same model can still make correlated mistakes; this design needs bounded comparative tests, not a promise of independent judgment.

Validate against a frozen baseline on varied cases: corrections, harmful candidates, justified uncertainty, literal controls and representative figurative constructions. Keep these inspected dev cases as regression fixtures only. Report net corrections, harmful flips, rejected correct proposals, uncertainty resolution and advisor-specific contribution separately. A no-advisor comparison is required before claiming benefit. Do not tune production rules to these sample IDs or feed their gold explanations into memory.

## Export failure: reproduced mechanism

`runs/diagnose_export_lock.py` extracts the actual `write_predictions` function from `run_figdebate.py` and runs it on a temporary scratch CSV. A Windows file handle permits reads but not deletion/replacement. The exporter raises PermissionError with WinError 5, matching the failed run. The old CSV remains intact and the newly written temporary CSV exists. After releasing the handle, the unchanged exporter succeeds. Assertions passed; result is in `runs/export_lock_diagnostic.json`.

This establishes that a replacement-blocking handle is sufficient to reproduce the failure. It does not prove which program held the historical file. The original run files were not modified by the test.

The application defect is allowing a derived export to abort durable result finalization. The general fix should commit per-case results and progress independently of a CSV; handle transient replacement failures with bounded retries, retain an explicit export-pending/error state, and support offline regeneration after the lock is released. A persistent lock should leave an honestly reported incomplete export, never a falsely successful run or an inference rerun requirement.

## Proposed order

1. Repair and test durable result/export separation.
2. Prototype structured interpretation and advisor eligibility without changing initial-arbiter behavior or weakening the gate.
3. Target the observed verifier failure modes with bounded tests across positive, negative and uncertain cases.
4. Compare advisor-enabled and advisor-disabled behavior before another full development evaluation.

No production repair or advisor improvement is claimed by this diagnostic work.
