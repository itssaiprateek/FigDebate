# Tribunal and visual reliability handoff — 14 September 2026

The changes are implemented in the release candidate working tree. The initial arbiter is unchanged. Production behavior uses general evidence, routing and runtime mechanisms; sample identities appear only in diagnostic selection and provenance checks. No full development-set run, Git push or hardware-setting change was performed.

## Changes

- **Question routing:** caption premises go to the language witness, visual observations to the visual witness, and semantic interpretation to the tribunal. A final visual dispatch guard blocks interpretation questions from reaching that witness. Semantic questions remain tribunal tasks and are never added as visual evidence.
- **Time budgets:** recent measured stage costs inform reserves within the existing hard case deadline. Required witness evidence takes priority over optional precedent feedback. Model timeout fragments remain unusable; budget exhaustion is reported explicitly.
- **Verification:** the model selects token intervals in the source caption; code reconstructs exact quotations and rechecks their provenance. A narrowly scoped incomplete-reason repair preserves the other proof fields and revalidates the result. Interpretation guidance distinguishes evaluation, figurative mapping and factual assertion; mapping remains a role-grounding stage.
- **Feedback:** select one relevant methodological precedent for the unresolved operation, allow at most one guided attempt per case, and log its proposal, verification rejection and cost. The frozen precedent library remains methodological guidance, not an empirically validated case memory or evidence about the current image.
- **Visual recovery:** detect repetitive OCR early and use one bounded structured transcription attempt. Valid selective text is positive evidence only, with no claim of exhaustive coverage. A model-produced, schema-valid `UNCLEAR` remains uncertainty. Ordinary visual descriptions containing the word “Contradiction” are no longer automatically treated as leaked task answers.
- **Diagnostics:** record hardware state throughout the run, including power mode, clocks, utilization, temperature, memory and supported clock-limit counters. Reports count proposals from all hearings, including rejected harmful proposals, and separate feedback cost and stopping reasons.

## Targeted validation

| Check | Observed result | Artifact under `runs/` |
|---|---|---|
| Three selected full-pipeline cases | Initial 2/3 correct → final 3/3; one correction, zero harmful final changes, zero failed tribunal reviews, zero invalid verification outputs, zero failed initial visual answers. Two semantic abstentions retained correct initial answers. Wall time 786.25 seconds. | `reliability_targeted3_20260914/` |
| Final mapping refinement, frozen first-hearing replay | The correction was accepted in the first hearing through the production gate, in 87.49 seconds. This is a replay, not an additional independent sample. | `reliability_final_mapping_20260914/` |
| Previously falsely rejected symbolic-cue question | Returned a valid observation after a bounded retry, in 9.80 seconds. | `reliability_visual2_20260914/` |
| Previously repetitive OCR crop, final recovery | Primary repetition was detected; structured recovery returned valid `UNCLEAR` in 6.69 seconds. The old attempt took 26.59 seconds and remained invalid. Text was **not** recovered. | `reliability_ocr_final_20260914/` |
| Focused regression suite | 213 tests: 212 passed, one existing expected failure. | `reliability_revision_final_unit.log` |
| Final visual refinement regression suite | 50 tests passed; overlaps the preceding suite, so the counts must not be added. | `reliability_final_visual_unit.log` |

All 129 Python files in the tested source/test directories parsed successfully, and Git whitespace checks passed.

The three-case integration run preceded the final narrow refinements: role-only mapping guidance, stricter archived source-span validation, one feedback attempt per case, additional telemetry fields and selective OCR recovery. Those refinements received the bounded replay and regression checks above. The complete final configuration has not been rerun across the full development set. The intermediate plain-text OCR retry still looped; that rejected attempt is retained in `reliability_visual2_20260914/` for audit.

## Remaining uncertainty and next evaluation

- The selected cases establish that these execution paths work, not held-out accuracy or a guaranteed gain of ten percentage points. Feedback produced no additional correct answer in the three-case check. Two semantic abstentions remain; lowering acceptance safeguards merely to increase changes is not justified by these results.
- The earlier GPU slowdown was observed, but its underlying trigger is still unproven. The user reported Performance mode and the original charger during the earlier run. The targeted integration run recorded Turbo mode throughout; no sampled active clock limits appeared and the earlier low-clock state did not recur. Hardware conditions confound any claim that code alone solved the slowdown. New telemetry is intended to distinguish the cause if it recurs.
- Recovery bounds failure cost and prevents invalid output from becoming evidence; it cannot guarantee that a model always supplies readable OCR, a correct interpretation or an answer before a deadline.
- The next user-run evaluation should use a fresh output directory and the same ten-case selection/settings for comparison. Review initial/final accuracy, corrections and harms, all-hearing proposals, failed verification, remaining abstentions, isolated feedback contribution, wall time and the hardware timeline separately. A later held-out evaluation and feedback ablation are still needed for paper claims.

The initial-arbiter scoring expected failure remains outside this revision's scope. No unresolved unexpected failure remained in the focused suites.
