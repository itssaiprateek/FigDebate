# Resumed implementation — 2026-09-28

The user subsequently authorized implementation based on the paper review.
The state below is a preserved historical stop checkpoint, not the current release
status. Current changes, enabled settings and validation are documented in
[the implementation report](semantic_protocol_implementation_20260928.md).
Completion-only initial reasoning is now the stagewise CLI default; the broader
semantic experiments remain opt-in after mixed live results.

---

# Paused at the user request â€” 2026-09-28

No inference process remains running. Existing results and source edits were preserved; no files were deleted, committed, or pushed.

## Safe defaults

The new initial-arbiter completion/grounding behavior and the additional tribunal grounding obligation are OFF by default. Their constructor switches remain available for explicit diagnostic replays. Existing output telemetry and final-artifact audit additions remain in the working tree. This is unfinished work, not a qualified release.

## Verification state

- The existing suite initially passed 693 tests (one expected failure).
- A later suite passed 702 tests (one expected failure).
- The focused completion/tribunal suite subsequently passed 31 tests.
- Changes after those checks, including the final safe-default checkpoint, have not received another full suite run.
- The latest compact, grammar-constrained initial assessment implementation has NOT received live inference qualification.
- Earlier free-form assessment experiments had mixed label results and did not reliably resolve semantic errors.
- The extra tribunal reading check blocked the harmful wheelchair proposal, but also blocked the previously helpful email proposal. It is not qualified for promotion.
- Baseline gate replay included an additional conditional instruction; it was subsequently removed for baseline mode. A strictly isolated baseline rerun remains outstanding. Do not claim controlled semantic gains from this preliminary replay.
- These are selected development cases with frozen upstream evidence, not a held-out or full-pipeline evaluation. Feedback was disabled. References and labels were kept out of model inputs.

## Saved runs

- `reasoning_completion_ablation_20260928`: 24 complete JSON records; invalid lines: none.
- `reasoning_completion_revision2_20260928`: 12 complete JSON records; invalid lines: none.
- `grounded_gate_20260928`: 8 complete JSON records; invalid lines: none.

## Resume carefully

Inspect the uncommitted diff and saved source snapshots first. Retain the unsuccessful results. Qualify the latest implementation on the same cases and exercise the real acceptance gate before enabling any behavior. Do not automatically resume an inference run or promote these changes merely because prior tests passed.
