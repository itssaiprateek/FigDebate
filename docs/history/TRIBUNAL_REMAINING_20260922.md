# Remaining tribunal changes — 22 September 2026

The compact proposer remains the baseline. Its prompt, schema and adaptation functions are unchanged. The initial agents, arbiter and model implementations were not modified. Feedback stays disabled. No full dataset or fresh ten-sample upstream run was performed.

## Retained changes

1. Carry the compact proposer's interpreted assertion explicitly into the final challenge. Include it in the proof's provenance binding so a changed interpretation cannot reuse a previous certificate. The independent relation verifier still does not see the proposer verdict or reasoning.
2. Preserve the complete source expression in downstream verification context. This makes the enclosing meaning available; it does not prove that a selected fragment captures every necessary qualifier.
3. Preserve the complete repair diagnostic in structured context. Long semantic diagnostics use a short routing question plus the complete attached detail, respecting the router's 320-character question limit. The existing model context, call and case budgets remain in force; no silent truncation is added.
4. Reject internally incompatible assessments of the same quoted condition, both during generation validation and during gate readback. Distinct unresolved conditions do not automatically erase established counterevidence. Remind the relation verifier that missing evidence does not prove its opposite.
5. Replace the expanded audit instruction experiment with a shorter tested instruction. It asks for actual invalid candidate inferences, reconciles the proposed interpretation with the source, separates same-direction alternatives from defeated opposing alternatives, and permits empty error lists.
6. Permit one bounded audit-only reassessment when an objection or internally inconsistent audit is the remaining blocker. Preserve the original compact candidate, bind it to the current image and caption, and reuse successful source-bound prerequisites. Genuine objections continue to block the normal gate. No objection is deleted by code.
7. Distinguish a complete but inconsistent clarification patch from invalid JSON. Reconstruct the patch against its recorded original, require exact field and schema agreement, and reject tampering. This affects eligibility for reassessment, not eligibility for accepting a label.

## Changes tested and withheld

- Expanded audit instructions: a three-case live check lost the previously accepted correction. That prompt was replaced rather than weakening the gate.
- Explicit premise accounting: implemented and tested as a separate candidate. It added required source conditions, stable premise IDs, per-condition citations and evidence-basis statuses. The candidate blocked the accepted correction after its expanded response reached a character cap. It was removed from the production path. Its code and traces remain in `research/tribunal_remaining_20260922/coverage_candidate` for inspection.
- Independent-reading experiment: not enabled. The existing blind relation stage remains; a further model call has not demonstrated enough benefit to justify adding it.

## Verification

The retained code passes 93 focused tests covering production entry/gate paths, exact source and image binding, complete repair context, routing, condition consistency, bounded scheduling, source-bound candidate reuse, successful prerequisite caching, tampered clarification patches, retained genuine objections, runtime errors and context overflow.

Four live development regressions used frozen upstream evidence, the production compact proposer, production verification and the production gate. The final check's source snapshot stayed unchanged during those four cases. Gold labels were used only for scoring after inference.

| Case | Result | Calls | Seconds |
|---|---|---:|---:|
| 4151 | Existing helpful correction preserved and accepted | 5 | 66.76 |
| 620 | Correct proposal remained blocked by a false audit objection; bounded audit-only repair reused the candidate | 7 | 79.19 |
| 2183 | Missing evidence remained unresolved rather than becoming a certified contradiction | 7 | 71.23 |
| 1300 | Correct initial answer preserved; audit identified a wrong role binding but remained internally inconsistent | 7 | 99.82 |

These four cases produced one helpful accept, no harmful or wrong accepts, and no runtime failures. One complete but inconsistent audit remained; this is not a claim of error-free semantic output. The clarification-patch scheduling issue found in the control was subsequently fixed and checked separately, without repeating the four-case run.

The additional live control follow-up successfully reused the original candidate and completed prerequisites. The audit still did not qualify for acceptance, and the correct initial answer remained unchanged. The follow-up validates execution of the repaired path, not a semantic cure.

## Limits and remaining work

There is no demonstrated increase in the number of helpful accepts over the previous run. Case 620 still exhibits confusion between evidence contradicting the caption and an error in the CONFLICT decision. Case 2183 still lacks a grounded figurative inference. Evidence selection and role binding can still omit or misuse decisive information. The new context and consistency checks expose these problems; they do not establish that the model's interpretation is true.

The default path retains only the smaller changes that preserved the accepted correction. These selected development checks do not establish dataset-wide accuracy, per-type stability or a guaranteed absence of harmful changes. No sample IDs, captions, image hashes or gold answers were added as production decision rules. Image/source hashes are used only for provenance and safe reuse.

Detailed traces, rejected candidate code, source snapshots, test logs and final analysis are in `research/tribunal_remaining_20260922` at the workspace root. The existing `run_compact10.ps1` remains the user-run command, with feedback disabled.
