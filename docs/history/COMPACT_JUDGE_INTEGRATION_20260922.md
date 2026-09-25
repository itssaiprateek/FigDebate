# Compact judge integration and audit reliability

## Ordered changes

1. Integrate the tested compact proposer into `TribunalMediatorAgent._review` for
   the V5 (`paper-8gb-review5`) pipeline. The implementation lives in
   `engine/simple_judge.py`; production does not import research scripts or
   replace methods at runtime. It sees the full source caption, original image,
   and complete admissible literal observation catalogue. The initial answer,
   gold label, and human reference explanation are not supplied.
2. Preserve evidence locations when a normalized visual verification is restored
   from a checkpoint/cache. Translation from `image_location` to `attachment`
   is idempotent. The result must still equal its original raw response, match
   the current catalogue, and pass case/model/prompt/schema binding checks.
3. Remove the V5 challenge's 240/480-character string ceilings. These could
   force closed JSON strings containing unfinished sentences even with tokens
   remaining. Keep the four-entry error-list bounds and concise sentence
   instructions. Increase the baseline challenge output allowance from 384 to
   512 tokens; the existing context ceiling, case time budget, and one retry
   allowance remain in force. Other stage budgets are unchanged.
4. Validate alternative/status consistency inside the challenge generation
   boundary. One bounded clarification can reassess only the alternative,
   alternative relation/status, deciding IDs, and explanation. Existing error
   lists and process checks remain unchanged. Every replacement is revalidated.
   Software never converts `UNRESOLVED` to `NONE`, and cannot erase an objection.
5. Report exhausted alternative-contract clarification as `INCOMPLETE_AUDIT`,
   with semantics not assessed. Retain separate reporting for runtime failures,
   truncation, genuine ambiguity, and inconsistent judgments. All continue to
   preserve the initial answer unless the complete evidence gate passes.
6. Provide `run_compact10.ps1`, which calls the official `run_figdebate.py`
   directly for the same ten IDs, fresh output directory, and disabled feedback.

## Acceptance policy

A transport defect does not establish that the proposed label is wrong. It also
does not establish that the label is correct. Only a completed, coherent audit
can authorize a correction. Material unresolved alternatives, unsupported
inferences, source/scope mismatches, unresolved necessary conditions, and
unverified image evidence remain blocking conditions.

The compact proposer retains its tested 384-token/four-field contract and is
followed by the same independent visual selection, role mapping, relation
assessment, challenge, and deterministic gate. These model calls share one
model; they are not statistically independent evidence of truth.

## Verification scope

Focused scripted tests exercise both correction directions through the real
production entry point and gate, full source delivery, invalid/duplicate IDs,
context overflow, explicit timeout reporting, restored cache locations, raw
record tampering, placeholder clarification, preserved objections, and genuine
ambiguity. They test mechanisms, not model accuracy.

Live qualification uses three frozen development cases with the production
tribunal and gate. Upstream outputs are archived; this is not a fresh full
pipeline run or a held-out evaluation. Results are recorded under
`research/integrated_simple_judge_20260922/` in the parent workspace. The full
ten-sample pipeline run is left to the user. No selected regression can prove
dataset-wide accuracy improvement or guarantee that hardware never times out.
