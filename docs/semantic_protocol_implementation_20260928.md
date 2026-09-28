# Semantic protocol implementation and qualification â€” 2026-09-28

This implements the preceding [research review](semantic_debate_research_review_20260928.md) in the existing pipeline. It adds no model, judge panel, external service, dataset or feedback loop. The working checkout is the local Desktop `FigDebate` repository. The previously paused edits were backed up before modification.

## What was learned from the implementations

The following released code was inspected, not merely its abstract:

| Source | Concrete implementation | FigDebate adaptation |
| --- | --- | --- |
| [Du et al. debate code](https://github.com/composable-models/llm_multiagent_debate/blob/9846749350eb917ae5bfaaff4c645fc705b8d3af/math/gen_math.py) | Independent initial message histories; later rounds exchange other agents' preceding responses. | Retain independent proposals and source-based verification. Do not import unrestricted self-verification or recursive retry behavior. |
| [Debate or Vote code](https://github.com/deeplearning-wisc/debate-or-vote/blob/82c929ea773d534cfb3fb0ddc4b7d14d245ab549/src/main.py) | Generate independent round-zero answers and keep per-round outcomes. | Separate initial reasoning, proposals and accepted decisions in qualification. Compare the same cases rather than counting accepts as success. Existing independent-vote control remains available. |
| [Stay Focused response generator](https://github.com/jonas-becker/problem-drift/blob/0a38cc9ed52a472eb2eceee5e40c55d4c6799eda/models/ResponseGenerator.py) | Explicit task focus; agreement is not a valid reason to change an answer. Its focus calculator measures scored changes offline. | Preserve the source assertion and track the disputed inference. Do not use offline gold-dependent focus scores during inference. |
| [CoVe method](https://arxiv.org/html/2309.11495v2) | Factored verification answers hide the draft response before revision. | Preserve existing fresh verification calls: visual selection, mapping and relation do not see the candidate. Only the final audit compares the candidate with the independent result. |
| [ColMAD Algorithm 1 / Appendix E](https://arxiv.org/html/2510.20963v2) | Independent answers followed by collaborative evidence checks and convergence logic. | Reuse one targeted repair, tied to the failed premise and available sources; record a dispute identity and stop repeated checks. Do not use agreement or confidence as certification. |

CoVe and ColMAD adaptations above use the papers' methods/prompts. No official author implementation was located for either during this review. They must not be described as reproduced released code. ColMAD v2 is a preprint. No benchmark gains from those papers transfer automatically to FigDebate.

[V-FLUTE](https://aclanthology.org/2025.naacl-long.1/) supplies the task convention: assess the original assertion, resolve contextual metaphors, and distinguish sarcastic expressed praise from the implied complaint. Sarcasm is never an automatic label. Human explanations remain evaluation-only.

## Implemented changes

- **Completion and explicit failure state:** `--reasoning-mode completion` increases the initial assessment allowance from 112 to 256 tokens, checks the required fields and unfinished clauses, and permits one bounded rewrite. It never scores an unfinished draft as if complete. If both attempts fail, the output records an incomplete assessment, unresolved relation and unverified binary fallback. Oversize inputs fail explicitly rather than silently truncating the source.
- **Aligned reading:** `--tribunal-audit-mode aligned-reading-1` adds reading kind, depicted referent and asserted property to the existing proposal and relation calls. These fields precede the verdict and are explicitly defined in the prompt. Figurative comparisons do not require a literal source object in the picture. `UNCERTAIN` cannot carry a directional relation.
- **Focused audit:** `focused-audit-1` uses the aligned reading and replaces independently generated error/status fields with one canonical objection list. Each objection names a disputed step, a reason and evidence IDs. Legacy status fields are exact software projections. Acceptance requires matching claim scope, no material objection, valid wire records and all existing evidence/provenance checks.
- **Bounded reconciliation:** a fully delivered but inconsistent audit may receive the existing specific recheck. Truncated, invalid or failed generation cannot masquerade as a semantic objection. Repeating the same failed premise and source evidence stops, even if the question is rephrased. A frozen candidate can be re-audited; new evidence or a changed candidate invalidates dependent proof.
- **Reproducibility:** official CLI/configuration and initial-stage cache identities distinguish reasoning modes. Qualification rotates variant order within each case, resets the seed and gate cache, checks original image bytes, snapshots sources and stores runtime details. No sample-specific inference rules were added.

The focused path still has five logical calls at most for an eligible first proposal: proposer, visual evidence, mapping, independent relation, audit. Existing bounded format recovery can add attempts. A schema proves record consistency, not natural-language truth. Multiple contexts of the same model are not statistically independent judges.

## Initial-reasoning comparison

Six selected development cases, three variants, 18 completed records. Exact upstream observations were frozen. No reference explanation, gold label or previous decision was supplied to the initial arbiter. Baseline and completion differ in the completion mechanism; the combined arm additionally changes interpretation prompts and structured relation handling.

| Variant | Complete assessments | Correct labels | Mean seconds | Median seconds |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0/6 | 1/6 | 21.43 | 11.77 |
| Completion only | 6/6 | 2/6 | 32.21 | 14.07 |
| Completion + grounded reading | 5/6 | 4/6 | 23.64 | 13.85 |

Completeness here requires the field and generation audit to pass. A response ending in a sentence at the token limit can still fail that audit. Completion is not semantic correctness. The larger allowance is an intentional compute increase, not an equal-compute accuracy comparison. Latency includes all initial-stage work; one long text-image case dominates the means.

All six original images and dataset references were inspected after inference. The completion-only email explanation gives an acceptable explosion-to-chaos connection. Its doctor/angel explanation still demands wings, and its wheelchair explanation remains indecisive despite a correct label. The combined arm gets the wheelchair comparison right, but its correct labels for the introvert and currency examples are paired with unresolved or inadequate explanations. Its email explanation adds an incorrect participant/instruction detail. Thus 4/6 correct labels is not 4/6 acceptable reasons. This is an assistant review, not an independent human annotation study.

Outputs: `outputs/semantic_protocol_initial_20260928`. The existing offline BERTScore tool remains unchanged. Its optional RoBERTa-large weights are not present in this checkout, so no new BERTScore or BLEURT number is claimed. Similarity would not validate the wrong-but-fluent explanations above.

## Gate qualification and release decision

Six identical saved cases, three variants, 18 completed first-review records. This runs the real proposer, pixel-based verification and acceptance gate, with frozen upstream testimony. It does not run a repair hearing. Each arm starts with the same saved initial decision, same observations and image hash, same model/profile and seed, and an empty gate cache. Order rotates by case. No failed or abstaining case is removed.

| Gate variant | Correct final labels | Helpful changes | Harmful changes | Correct / wrong / abstaining proposals | Mean / median seconds |
| --- | ---: | ---: | ---: | --- | --- |
| Baseline | 1/6 | 1 | 1 | 4 / 1 / 1 | 45.66 / 48.38 |
| Aligned reading | 2/6 | 2 | 1 | 2 / 3 / 1 | 54.86 / 58.90 |
| Focused audit | 2/6 | 2 | 1 | 2 / 3 / 1 | 54.63 / 59.25 |

All 18 proposals passed their delivery/format contracts. That does not imply all downstream checks or semantic inferences passed. Accepted label changes are 2, 3 and 3 respectively. The baseline begins with one correct initial label in this selected cohort, gains the email case and loses the wheelchair case. New variants additionally gain the angel label. They preserve the email correction but do not eliminate the wheelchair harm. The angel explanation still expresses the transferred property weakly, so the extra correct label is not strong evidence of improved joint reasoning quality.

The focused audit gives useful, specific objections for the introvert example (smiling is substituted for disappointment) and Prime example (a delay notice is substituted for expressed praise). It blocks both erroneous proposals. However, the aligned proposer has regressed from the baseline's correct-label proposals on those cases. This is why the audit's cleaner behavior is not sufficient to promote the combined change. The currency baseline has a correct dataset label with an invalid absence-of-evidence argument; its rejection is appropriate.

The schema variants share the existing model and case limits and add no logical call. They are not identical token allocations: aligned fields consume extra tokens; the focused compact audit allows 384 output tokens versus the baseline audit's 512. Report runtime as descriptive, not a controlled speedup or compute-matched semantic result. CPU unit checks overlapped part of this replay; no other GPU inference ran concurrently. Cold starts, case exits and longer explanations also affect timing.

The first diagnostic was stopped after discovering missing reading-field instructions; completed records and its source snapshot remain in `outputs/semantic_protocol_gate_20260928`, marked `STOPPED.txt`. It is not a completed evaluation. The corrected comparison is `outputs/semantic_protocol_gate_revision2_20260928`. Both completed comparisons contain `qualification_summary.json`, a post-inference `reference_reasoning_review.json`, and inputs for the optional similarity evaluator.

### Enabled and experimental behavior

**Enabled:** normal stagewise CLI runs now default to `--reasoning-mode completion`. The development launcher specifies it explicitly. Sequential execution retains its prior baseline behavior. For a controlled old-initial-reasoning arm specify `--reasoning-mode baseline` explicitly.

**Not enabled by default:** grounded initial reasoning, aligned reading, focused audit and the earlier process-audit experiment. Use `--reasoning-mode grounded` or the appropriate `--tribunal-audit-mode` only for an explicitly recorded experiment. The tribunal audit default remains `baseline`. This preserves the user's promotion rule: implement candidate mechanisms, but do not claim mixed semantic results qualify a new default.

A verified replacement clears stale initial-generation failure flags. A verified same-label confirmation may replace an explicitly failed initial explanation with its audited reason, retaining the failed original in the trace. An unverified proposal cannot clear failure state.

### Operational checks

- Final unit/integration contract suite: **720 tests run; suite passed with one existing expected failure**. No new expected failures or skipped failures were introduced.
- Completion-only restart/order check: two cases rerun after unloading/reloading the model and reversing their relative order. Labels, explanation text and completion records matched exactly **2/2**. This is a small repeatability check, not a cross-hardware guarantee. Artifacts: `outputs/semantic_protocol_completion_repeat_20260928`.
- Official-runner smoke check: **1/1 completed**, with a completion manifest and final artifact. The launcher selected completion mode without an explicit override. Vision, language, independent candidates, tribunal and final output ran with focused audit, bounded repair and feedback disabled. The case changed from CONTRADICTS to the correct ENTAILS, and the final explanation passed its completion audit. The weak transferred-property wording remains; this validates integration, not a new semantic gain. One review sufficed, so live semantic repair was not triggered; bounded repair routing/reconciliation is covered by the real-path contract tests. Artifacts: `outputs/semantic_protocol_pipeline_smoke_20260928`.

The completed evidence comprises 36 paired stage evaluations, two restart/order repeats and one full-pipeline smoke case. The stopped diagnostic is retained separately. There are no running inference jobs from this task.

The source was preserved before editing under `.cache/semantic_protocol_implementation_20260928/before`; all live qualifications have source snapshots. Feedback remained disabled in every run. No full dataset run was started.

## Limits

These are selected training/development cases, not a random cohort, held-out result or full-dataset accuracy estimate. Frozen-stage replay isolates mechanisms but cannot establish end-to-end gains. Gold labels and human explanations were never used as inference input or an acceptance rule. Feedback stays disabled. At the implementation checkpoint, no files had been deleted, committed or pushed.

## Subsequent 20-case operational check

The user reran the original 20 cases with completion reasoning and the baseline tribunal audit. All 20 completed, with 525 successful recorded generation requests and no recorded timeouts or OOM recovery. Final explanation completion passed for 19/20; runtime was 68 minutes 26 seconds. Remaining failures involve a missing dedicated evidence-ID heading despite inline citations, a choice question incorrectly routed to a yes/no contract, and an internally inconsistent audit. These remain unresolved in this iteration. This check assesses operational reliability, not accuracy or repeat-run determinism. See [the operational stability audit](operational_stability_20260928.md) for evidence and limits.
