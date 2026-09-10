# Tribunal implementation checkpoint — 2026-09-09

## Scope and source of truth

Active checkout: `C:\Users\Sai Prateek\Desktop\FigDebate_sent\FigDebate_main`.
Changes are in this checkout, not the older Claude review worktree. Existing user
changes were preserved. Dataset preparation, category selection, official splits,
gold labels, human explanations, and saved predictions were not edited.

This is an implementation checkpoint, **not a declaration that all requested
semantic changes or research qualification have been completed**.

## Implemented in the existing execution path

1. Hearing accounting reads actual `hearing_transitions`; the obsolete
   `_deterministic_repair_followup` reader was removed. Plans, executed hearings,
   repair-triggered hearings, changed content and actual judge re-reviews are
   separate. Existing accepted-change reporting is retained.
2. Verification completion requires the premise, mapping and argument calls as
   well as the two relation calls. Completed negative judgments remain completed
   execution, not runtime failures.
3. Incomplete output at a token cap is distinguished from malformed output.
   A valid response at the cap is not treated as truncated. The existing single
   format retry can use a larger context-bounded budget; no retry layer was added.
4. Factored obligations have differentiated initial budgets and bounded reasons.
   These budgets are initial engineering settings, not a proven token-sizing law.
5. Failure reports expose leaf obligations and composite-check dependencies.
   Mapping and bridge-direction failures are no longer presented solely through
   their multiple downstream symptoms.
6. The existing hearing repair route now recognizes bridge-grounding,
   bridge-direction and relation-disagreement failures. The existing proposal
   repair receives the specific defects and no longer demands an invented
   opposing argument when none is supported.
7. Judge calls record seed, determinism, parameter devices, quantization,
   input/output limits, image dimensions and cold/warm status. Timeout diagnostics
   additionally retain first-token timing, generated-token count, memory and
   partial response. Partial responses remain inadmissible diagnostic data.
8. Review timing separates inclusive review time from verification and proposal
   time. Timeout elapsed is explicitly overlapping, not another additive phase.
9. Terminal-output policy distinguishes failed/incomplete review from genuine
   semantic uncertainty; neither is presented as a qualified changed answer.
10. The existing review-replay tool accepts exact sample IDs for bounded,
    reproducible development checks. No alternative production pipeline was added.
11. The arbiter omits the `intended_meaning` field unless the claim contract is
    explicitly semantically qualified. In the completed 20-case scoring-only
    comparison, this preserved all 20 baseline labels (12/20 correct). It is an
    input-integrity repair, not an accuracy improvement. Other derived language
    fields have not thereby been proved safe.
12. The proposal prompt explicitly permits an empty counterargument when no
    evidence-supported alternative exists, while forbidding hiding a material
    alternative. This agrees with the existing empty-counter gate contract.

## Existing mechanisms retained instead of duplicated

- Token-aware dossier packing with mandatory source/graph preservation and an
  explicit index/retrieval path for evidence not fully shown.
- Bounded evidence retrieval, bounded proposal repair, and a second hearing only
  when requested; reconsideration depends on a changed deliberation signature.
- Image/caption/proposal binding of verification. Changed proof subjects cannot
  reuse prior acceptance proof.
- Acceptance of qualified corrections in either direction, and same-label
  confirmation. An absent or same-direction counterargument already does not
  automatically veto a proposal.
- Human-review export and paired evaluation tools.

Historical diagnostics, test fixtures, attribution files and source-preserving
fallbacks have not been deleted merely because they are not imported in a normal
prediction. No broad file deletion was justified by this change. Rejected
experiments stay outside production, in the research workspace.

## Experiment boundary

The matched argument experiment used two development cases and four arms each:
baseline, enum reversal, clearer wording, and both. All eight returned SUPPORT
for the bridge, including explanations that explicitly described contradiction.
This rejects these interventions as remedies for the observed typed-direction
error; it does not establish an exhaustive model capability limit.

The argument prompt/schema from this experiment has **not** replaced the
production argument audit. There is no label copying or lexical override.

## Not yet qualified or deployed

- A new argument/mapping design qualified against human reference judgments.
- Relaxation that accepts a sufficient decisive justification despite defective
  incidental narration, or dismisses a material objection. This requires reliable
  identification of what is decisive/material; it cannot be implemented safely by
  ignoring a failed Boolean check.
- An Agent 2 model replacement or removal of its existing phases.
- Removal of the prior assessment from scoring. In the new 20-case comparison it
  produced three helpful and three harmful changes (12/20); combining it with the
  intended-meaning restriction produced two helpful and three harmful changes
  (11/20). Neither combined nor assessment-removal candidate was deployed.
- Per-obligation deadline scaling. Output budget alone is not a calibrated time
  model. The global 120-second setting is unchanged.
- Full fresh end-to-end paired development evaluation, larger two-rater semantic
  qualification, calibration and final held-out research evaluation.

Three expected failures remain explicit: two truncation heuristics
and the assessment-conditioned classifier requirement. They must not be reported as
passing requirements or hidden to make the suite look complete.

## Evidence location

Detailed test logs, experimental outputs and current validation summary:
`C:\Users\Sai Prateek\OneDrive\Desktop\FigDebate_sent\FigDebate_sent\research\implementation_20260909`.

The initial integration replay reproduced timeouts on cases 4315 and 1584, before
factored verification. Case 4555 completed with SUPPORT, including one bounded
proposal repair, but still returned an unresolved-counter Boolean inconsistent
with its prose. Its inclusive review time was approximately 488.79 seconds.
This run therefore does not establish execution reliability or an accuracy
improvement. Each replay records its source checksum; subsequent instrumentation,
counter-prompt clarification and classifier repair must not be attributed
retroactively to it.

The user supplied first-reviewer judgments: case A/4555 ENTAILS and cases B/1584,
C/3351 and D/4315 CONTRADICTS. These agree with the four dataset labels. Unanswered
dimension-level ratings remain null; the reply is not blanket approval of every
generated observation, mapping, argument or counterargument audit.

## Additional candidate results

The already-installed Qwen model completed five caption-only core-extraction
probes with valid JSON. This is not a complete Agent 2 replacement evaluation:
interpretation, graph construction, independent source audit, witness behaviour,
latency under matched conditions and downstream accuracy remain unqualified.

A separate four-case argument contract used explicit ENTAILS/CONTRADICTS labels
and counterargument materiality categories. It corrected the extracted direction
on 4315, but still returned ENTAILS while describing contradiction on 1584,
treated the frog's literal objection as material, and incorrectly endorsed the
3351 argument's reversed outcome mapping. It was rejected for production use.
Human feedback on the four final labels does not cure these model-output errors.

The broad acceptance relaxation and Agent 2 replacement are therefore still
unfinished, not silently enabled. There is no claim that the remaining critical
problems are solved or that this checkpoint is research-release ready.
