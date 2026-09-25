# Tribunal repair and qualification — 17 September 2026

The user authorized general tribunal changes, conditional on improvement, with targeted tests only. Initial inference and feedback are outside this change. The exact dirty starting source is preserved in `../research/tribunal_repair_20260917/before/` (manifest `157f029858297ef4b0e287395cfb91639c39a189e035878dd5a56c8d45331bee`). Earlier uncommitted work is preserved.

## General reliability changes

- Validate integer types and numeric bounds; reject Boolean, fractional and out-of-range span positions. Keep the separate nonempty interval check.
- Share source-span recovery across the compact proposer and verifier mapping/relation arrays, including unestablished conditions. A generated replacement changes one selected field only. Revalidate the entire retained response and its proof obligations. No clamping or extra retries.
- Use the existing one-field recovery for incomplete verifier attachments. Identify the field as a location/panel/speaker, keep its broken value in the audit, and avoid presenting truncated OCR as the rewrite target.
- Recognize dangling `and`/`or` in tribunal-generated explanations. This is an explicit tribunal-only option; the visual agent retains its original completion heuristic. Exact source quotations remain outside this prose heuristic; a missing period is allowed. Character-limit boundaries are telemetry, not automatic rejection or a semantic truth test.
- Record `single_field_repair` accurately, including the narrowed token allowance. Keep the existing call/case ceilings and gate conditions.

These changes address transport and recovery. They do not establish model accuracy or guarantee that every future output will be valid.

## Rejected challenge experiment

Four frozen, first-hearing completed proof chains were tested in both response orders, using the same explicit challenge contract, image, source, proof payload, seed and token/time limits. All four eligible chains were included. Feedback was disabled. No references were sent to the model.

| Order | Format-valid results | Passed production proof audit |
|---|---:|---:|
| Error first | 3/4 | 2/4 |
| Alternative first | 2/4 | 0/4 |

The two audit passes were confirmations of already-correct initial labels, not new corrections. Their explanations still did not establish a sound positive incompatibility: one treated lack of literal evidence for a figurative claim as sufficient conflict. Both orders repeated the same false incompatibility on the harmful advertising proposal. The remaining identity/scope case failed to complete a qualified response within the bound.

The experiment therefore fails the stated promotion criterion. Neither a larger acceptance count nor structurally valid `NONE` is evidence of a better tribunal. The experimental instructions, placeholder validator and response-order changes remain outside the production pipeline. The gate has not been loosened, objections have not been dropped, and uncertainty has not been converted into agreement.

## Qualification evidence

The experiment scripts, complete generated responses, input manifests, source snapshots and logs are in `../research/tribunal_repair_20260917/`.

The first two live repair probes had a test-wrapper defect: the wrapper did not advertise `json_schema`, so the production wrapper did not forward the decoder schema. Those probes are excluded. The corrected harness asserts the replacement-only schema and the 160-token ceiling before every live repair. This is a harness correction, not a change to the model runtime.

## Rejected context-selection controller

A separate controller implemented one short information selection, batched retrieval, protected requested records and one final proposal. Five offline disclosure tests passed. It was tested against the existing controller with identical frozen upstream state, feedback off, and unchanged V5 reserves/case limits. The predeclared panel selected one initially correct and one initially wrong case by hash, excluding the latest ten and the previous compact/final live panels.

The first pair failed qualification, so the second case was not run:

| First disjoint case | Existing controller | Selection then proposal |
|---|---:|---:|
| Valid final review | Yes, ABSTAIN | No, proposal timeout |
| Final answer | Original correct answer | Original correct answer |
| Judge seconds | 113.76 | 120.75 |

The selector spent 9.73 seconds but requested no unread evidence or context. That cost consumed part of the shared proposal allowance. The final generation then timed out with approximately 119 seconds still reserved for verification. This demonstrates a failure of this proposed controller under the current allocation; it does not justify lowering the reserve to fit this particular case. The controller and helper were archived outside the application, and no profile enables them.

## Final recovery checks

| Check | Result | Live repair calls | Seconds |
|---|---|---:|---:|
| Recorded empty source interval | Repaired only the interval endpoint | 1 | 11.66 |
| Recorded truncated OCR attachment, revised repair context | Repaired only attachment to `red box` | 1 | 22.52 |
| Incomplete-location fault injected into a different valid visual record | Repaired only that attachment | 1 | 9.07 |

The initial attachment instruction alone failed: the model copied the truncated OCR again. The retained revision removes that broken attachment from the rewrite target and error echo, while retaining the original in the audit and keeping the full source evidence available. The independent injected fault checks the same mechanism on another image. These are format-recovery interventions, not three corrected task answers.

The source-span probe replays a malformed response with one repair allowance available. It does not establish recovery when an earlier semantic error has already consumed that allowance. The production path still permits at most two attempts overall.

Two additional frozen-proposal checks used the current verifier and production gate, allowing reuse only of successful records whose exact production input/cache keys matched. Both originally harmful proposals remained unaccepted and both correct initial answers were preserved:

- The emotion proposal completed the proof calls but retained an unresolved challenge. This was not a useful correction or a demonstrated sound identification of the model's error.
- The album proposal successfully repaired its visual attachment and completed mapping, then timed out during relation verification within the diagnostic's 160-second total bound. This is an incomplete proof, **not** a successful semantic rejection. The diagnostic cap was at most 160 seconds, within the normal 240-second case ceiling; it is not a measurement of a complete fresh pipeline run.

These two negative controls observed no harmful acceptance. They do not establish zero harm over the dataset or complete end-to-end reliability.

## Preservation and tests

The final regression suite ran **180 tests: 179 passed and one pre-existing expected failure** (`test_scoring_cannot_condition_on_assessment`, in the unchanged arbiter scope). The exact dirty baseline comparison confirms 169 existing source files unchanged; six existing source files changed, plus one new transport test module. Normal V4 valid-response prompts and token budgets match the starting version in both relation directions. The initial arbiter, model runtimes, profiles, time-budget policy, feedback and acceptance-gate files are byte-identical. The final scope check isolated the conjunction heuristic to tribunal callers, preserving its tested tribunal behavior and restoring the visual agent's original behavior. `git diff --check` passed.

The final verified source manifest is `dc472b28d34d92e8578a0f7ee59a71bc93d950d635a8359fa09187db784d69f5`, under `../research/tribunal_repair_20260917/final_source_verified/`. Each live experiment also retains its own source archive; the final scope-only heuristic change does not change tribunal prompts, decoding or completion decisions.

The new field repair preserves the other fields, rebinds exact source text and reruns the full response validator. Tests cover empty, reversed, fractional, Boolean, negative and oversized intervals; an invalid second repair cannot request a third attempt. A repair containing an unrelated verdict/field cannot be accepted.

## What was not promoted

No new semantic proposer rewrite or challenge policy was enabled. The normal V4 proposer already contains explicit role/scope and condition comparisons. A new V4-versus-shortened-V5 proposer-format comparison remains unqualified; it was not expanded into another prompt-tuning loop after the challenge and controller candidates failed. V5 remains the existing opt-in experiment. Feedback changes remain deferred.

The retained changes improve validation and demonstrated recovery. They **do not establish a dataset-wide accuracy gain, improved correction precision, or zero future failures**. No full development-set or 6,000-sample run was performed. A larger benchmark remains user-run, and these selected development cases must not be presented as held-out evaluation.
