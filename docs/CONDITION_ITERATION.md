# Condition verification iteration — 2 October 2026

## Scope and predeclared checks

Experimental modes: `condition-audit-1` and `condition-repair-1`. Existing focused defaults remain unchanged. Feedback remains disabled. No model replacement or arbiter scoring redesign.

The dirty starting source is archived under `.cache/condition_iteration_20261002_before` (manifest b026316288e3ca91af4ecbe12660d65713557a9a3942228fc21b13e3f4901915). Earlier uncommitted work is preserved.

C0: evaluation-only substitutions for checked visual selections, claim-role mappings, or condition assessments, with exact source hashes and reviewer provenance. Oracle results are diagnostics and never count toward performance.
C1: full original caption is immutable C1 in each new-mode stage. Interpretation remains a hypothesis. The comparator uses the same asserted-claim convention only in the new modes.
C2: one authoritative condition table; overall relation, unresolved list and legacy condition fields are software projections. Runtime decomposition remains unqualified, so C1 covers the complete caption. This does not prove that a model actually checked every clause.
C3: explicit paired actual and asserted states, with participant/speaker/time scope; one pair for a single subject, two for a comparison. Mapping retains these roles without adding a call.
C4: the condition connection links visible basis to a licensed interpretation and attributed property. No literal-object requirement or blanket symbolic association.
C5: the repair variant regenerates a candidate once for a concrete candidate objection; verifier disputes retain the frozen candidate. Repairs retain original C1, case budgets, novelty checks and maximum rounds. Incidental unsupported records without a material missing-observation request no longer trigger witness repairs.
C6: mapping-only recovery of an unterminated final reason when all prefix fields are complete and schema-valid. One field retry, 160 tokens, all bindings preserved and revalidated. Does not complete OCR or partial bindings.
C7: final artifacts distinguish the binary label, verified label and exact audited explanation, including source/image and explanation hashes. Existing gate and provenance safeguards remain.
C8: capability/model comparison is conditional on residual evidence; no new model or training in this iteration.

## Bounded live qualification (declared before inference)

Four paired first-review cases: vflute_train_1595 (preserve a valid idiom correction), vflute_train_4547 (prevent qualifier-dropping harm), vflute_train_3351 (actual outcomes versus desires), vflute_train_4255 (mapping completion and comparison). Both arms use the same frozen pre-review ledger, image bytes, caption, model, seed, budgets, and actual acceptance gate. Arms: focused versus conditions. Rotating arm order; no best-of selection, retries beyond existing bounds, gold in prompts, or full dataset inference.

The mapping reliability fix is common to both arms; this is a semantic comparison, not an estimate of its isolated gain. A separate bounded mapping-only check will qualify that repair. Comparator changes are not exercised by frozen-input replay; a later user-run end-to-end check is required.

Report operational validity, proposal correctness, final corrections and harms, explanation adequacy against references, and runtime separately. These selected development cases cannot establish general accuracy. Promotion requires improved acceptable corrections with no added harms; tiny/noisy outcomes leave semantic modes opt-in. Log all outcomes including regressions.

Human-reference reasoning review: mark supported, partly supported, or unsupported; compare decisive property, roles/polarity and evidence rather than wording overlap. Label correctness alone is insufficient.

## Qualification result

746 tests completed: 745 passed, one pre-existing expected scoring-independence failure. Four paired live first reviews: focused 1/4 final labels, one correction and one harm; conditions 1/4, no corrections and no harms. All returned call outputs were valid; neither arm had a failed model call. Mean times were 58.66s and 54.21s; earlier abstention explains part of the latter. Image/reference reasoning review found one adequate final answer in the control and none in the experimental arm. Semantic modes are NOT promoted.

The default mapping-only completion fix recovered the saved failure with frozen bindings in 6.91 seconds and also triggered successfully in an unassisted control review. Its benefit is completion, not proven semantic accuracy.

The one-case oracle intervention demonstrated that a correct condition assessment leads to a useful critique; the repair variant then generated and accepted a correctly reasoned contradiction. This used the supplied assessment and is excluded from accuracy. Candidate scope objections, as well as inference objections, now permit one new candidate; verifier-only disputes retain the frozen candidate. No proof of improved unassisted repair effectiveness is claimed.

See `outputs/condition_pair_20261002/qualification_report.html` and `qualification_summary.json` for per-case reasoning and exact traces. The oracle repair ran a shared in-memory 240s case budget; production also restores source-bound proof cache across model reloads. The comparator change has unit coverage only; replay held upstream outputs fixed.

To evaluate the experimental path in the official stagewise runner, use `--tribunal-audit-mode condition-audit-1` or `--tribunal-audit-mode condition-repair-1 --tribunal-repair-mode bounded`, retaining `--feedback-mode disabled`. Neither is the default. For bounded replay, the existing `evaluation.qualify_grounded_gate` accepts variants `conditions` / `condition_repair`, optional `--bounded-semantic-repair`, and a separate `--oracle-file`. Oracle entries require source hash, stage, full schema-compatible value, reviewer and basis. Witness repair routes are logged as requiring upstream execution, never simulated.
