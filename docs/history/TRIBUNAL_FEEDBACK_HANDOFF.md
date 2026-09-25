# Tribunal and feedback qualification

This supersedes the previous engineering handoff for tribunal/feedback work. No full dataset evaluation was run for this change. The initial arbiter, visual extraction, model weights and model revision pins remain unchanged.

## Implemented changes

1. **Original-caption binding.** The production acceptance gate now receives the same immutable dataset caption as the verifier. Previously, an agent-normalized caption could alter the subject fingerprint through whitespace changes, rejecting valid proof. Hash verification remains exact: changing the caption's substantive content still invalidates the proof. The source contract is copied, never mutated in the initial decision.
2. **Verification reserve.** Proposal generation, formatting retries and evidence retrieval share the time above a 120-second verification reserve within the existing 240-second case ceiling. Follow-up admission requires time for a proposal and that reserve. This bounds spending; it does not guarantee completion on slow hardware.
3. **Compact judge output.** The judge uses bounded structural JSON whitespace and asks for short complete deciding clauses. Source quotations and whitespace inside strings are preserved. Initial agents retain their existing decoder behavior. The same evidence, role, direction and counterargument checks remain mandatory.
4. **Runtime diagnosis.** Per-call diagnostics now record power connection, GPU clocks/power/temperature/utilization, and time spent in grammar and completion callbacks. These are bounded read-only snapshots; they do not change power plans or GPU settings.
5. **Selective precedent feedback.** `--feedback-mode precedent` first runs the ordinary tribunal review. It supplies a retrieved precedent only when that review is semantically unresolved and sufficient time remains. A guided review can replace it only with valid independently checked proof. Resolved baseline reviews are preserved. Unsuccessful guidance remains archived with its cost and failures visible.
6. **Frozen library and leakage controls.** Retrieval uses structural features rather than caption word overlap or expected answers. The default is one precedent, with a maximum of two. Each entry has applicability conditions, exclusions, an example and a counterexample. Empirical entries require training provenance and disjoint image, caption and template-group identities; missing group metadata fails before inference. No online updates occur during evaluation.
7. **Measurement and replay.** Feedback failures are reported even when the system retains the ordinary review. The new bounded replay tool reconstructs round-one hearings from checkpoints, excludes newly promoted judge evidence, checks image/caption identity, freezes the inputs, uses the production seed and runs the ordinary acceptance gate. It performs no initial inference or new witness hearing.

## What the library is

`config/tribunal_precedents.json` contains four explicitly labelled **synthetic methodological precedents**. They illustrate figurative scope, comparison roles, speaker attribution and evaluative polarity. They are not empirically validated historical cases and must not be described as a learned precedent memory in a paper. The infrastructure also accepts reviewed training cases, with provenance and overlap checks. Those cases still need an independently reviewed training library before making an empirical case-memory claim.

The final verifier receives current evidence and the proposed argument, but no retrieved library entry, precedent assessment or old case facts. Its use of the same model does not establish independent model errors.

## Qualification design

Artifacts are under `runs/tribunal_feedback_qualification/`:

- `before/` is the frozen pre-change Git source.
- `baseline3/` runs that source on three saved hearings.
- `revised3/` tests compact review generation before the caption-binding integration fix.
- `gate_binding_replay.json` rechecks the saved outputs through the corrected gate, including altered-caption rejection controls. It performs no new inference.
- `feedback3/` is the rejected unconditional-feedback experiment. It is retained for transparency and is not the final implementation.
- `final5/` exercises the final selective-feedback implementation and the production gate, including preservation controls.
- `final_unit_suite.log` records code/grammar regression checks.

The five-case live check completed with **3/5 initially correct and 4/5 finally correct**, one helpful correction, no harmful changes, and no failed review or verification executions. Both added preservation controls stayed correct. Tribunal work took 413.25 seconds, excluding model load and upstream stages. This small, selected result is not evidence of 80% population accuracy or a general 20-point gain.

On the three shared diagnostic cases, frozen baseline tribunal work took 327.20 seconds; the compact implementation before the gate fix took 198.93 seconds. Replaying those saved proofs through the corrected gate admitted one additional correct answer while rejecting the harmful proposal. The final live five-case check independently exercised that correction through the production gate.

The full code suite completed **513 tests: 512 passed, one existing expected failure, no skips**. Additional focused metric and retrieval checks passed after that suite. The expected failure is `test_scoring_cannot_condition_on_assessment`, an existing arbiter issue outside the user-authorized change scope; it remains a paper limitation. Source comparison confirms the arbiter and initial visual components are unchanged.

The unconditional-feedback experiment caused a timeout and supplied no additional correction, so it was rejected. Selective feedback completed without an execution failure but initially retrieved an irrelevant speaker precedent for a temporal claim. The matcher was corrected to require explicit attribution structure rather than any time/scope field. `retrieval_final1/` checks that refinement separately; it must not be counted as another independent evaluation sample.

That final recheck completed successfully in 88.93 seconds for the ordinary review plus feedback. It retrieved the figurative-scope precedent, explicitly assessed its applicability, and retained the unresolved answer because no verified resolution was found. **Feedback has not demonstrated an additional correct answer in this qualification.** Keep it opt-in for paired evaluation; the positive correction demonstrated here comes from the revised tribunal and caption-binding fix. A reviewed empirical precedent library and untouched evaluation remain research work.

Runtime snapshots captured a slow call at approximately 555 MHz and 14.66 W versus faster calls near 2,500 MHz and roughly 45–55 W, with AC reported connected. This is evidence of runtime variation, not proof of its cause. No hardware/power setting was modified, and no guarantee of timeout-free operation under other operating conditions is made.

These are selected development diagnostics, not a fresh five-case pipeline run or a held-out accuracy study. The replay excludes later witness hearings and second tribunal rounds. Do not pool repeated cases across variants or infer population accuracy from them.

## User-run evaluation options

Keep all settings and selected IDs identical across the two commands. For the tribunal-only arm use `--feedback-mode disabled`. For the selective-feedback arm use:

```powershell
--feedback-mode precedent --verified-feedback-file config/tribunal_precedents.json
```

Use a new output directory for every arm. Existing default feedback remains disabled, so older commands do not silently change the experiment. The run configuration records the library SHA-256 and its tribunal-only role.

## Completion criteria for the paper

Freeze the model/code/library and development-selected configuration before an untouched evaluation. Compare initial arbiter, tribunal without precedents, and the same tribunal with precedents. Report net accuracy change, correction precision, initial-error correction coverage, harmful-flip rate, operational failures and runtime, both overall and by figurative type. A gain above ten percentage points remains a research target until demonstrated on untouched data; it is not an acceptance rule or a promise for every run.
