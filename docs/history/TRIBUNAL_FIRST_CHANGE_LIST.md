# Tribunal-first change list

Status: proposed work only. No implementation or model test is authorized by this document itself.

The experimental advisor candidate has since been removed from the live system at the user's request. This tribunal-first plan remains proposed; its acceptance and abstention changes have not been implemented by that removal.

## Baseline and scope

Use the tribunal behavior recorded in `runs/dev50_reliability_20260914_111440`, before the recent structured-interpretation/advisor candidate. The recovered records in `runs/dev50_recovered_completion` restore the same decisions; they are not a new experiment. Do not treat the failed four-case candidate as this baseline.

The original run finished with 35/50 correct initial decisions and 35/50 correct final decisions. Nine final change proposals comprised five correct-label and four harmful proposals. The gate accepted two of each and rejected three correct-label proposals. Of 31 final semantic abstentions, eight preserved wrong initial decisions and 23 preserved correct ones. There was also one failed review.

Keep the initial arbiter, models and upstream inference fixed. Exclude the new advisor and the large interpretation schema. Retain separately tested persistence/export repairs. Do not change sample-specific production behavior, inject gold into inference, weaken evidence checks to hit an acceptance count, or promise that only correct cases will abstain.

The historical run used legacy precedent feedback. For causal comparisons, preserve that condition on both sides or disable it on both sides and label the resulting baseline separately. A feedback-disabled replay must not be presented as the exact historical experiment. Freeze saved inputs, source/configuration identities and per-case budgets before comparison; a Git commit alone may not reproduce the dirty source used for the original run.

## T1 — Give the verifier its own evidence selection

**Observed problem:** verification checks the proposer's selected observations and restricts its citations to them. A true narrow quotation can pass while missing the deciding scene, panel or time range. Independent prompts therefore still inherit an important selection decision from the proposer. Whether this is the principal cause needs a controlled comparison.

**Change:** modify the existing visual-verification stage to inspect the full image and the immutable caption with a bounded index of grounded observations. Let it identify the deciding observations independently, including relevant counterevidence, without seeing the initial or proposed verdict. Keep at most four deciding records in the final proof. When a deciding fact is absent from the ledger, request one specific visual observation through the existing witness route; do not fabricate a new grounded record from interpretation.

Bind the resulting proof to the actual selected observations, image, source and configuration. Changing the evidence selection invalidates dependent verification. Preserve observation provenance and source-span checks.

**Boundary:** `engine/evidence_verification.py`, `engine/independent_review.py`, evidence/proof binding and existing witness routing.

**Qualification:** omitted relevant evidence can enter the verified case; unsupported or unrelated evidence cannot. Check broad claims against complete relevant scope, and retain genuine corrections. Compare against identical frozen proposals, so proposer changes cannot explain the result.

## T2 — Check the same subject, property and scope before deciding direction

**Observed problem:** harmful acceptances confused an object's property with the reader's reaction, and treated an unprinted subjective evaluation as contradiction. Correct proposals were blocked when sarcasm was reversed or a desired outcome was treated as an actual outcome.

**Change:** use the existing mapping and condition-check fields to make the comparison explicit: the exact caption assertion, its subject/property/scope, and the observed state of that same subject. Preserve expressed assertion separately from intended irony when they differ. Do not replace the source with a freely rewritten caption or revive the unqualified claim decomposer.

For SUPPORT, the deciding evidence must establish the necessary claim conditions. For CONFLICT, it must establish a positively incompatible state on the same subject and scope. Missing sentiment words, a missing literal object for an idiom, or an intention without an observed result is insufficient by itself. UNRESOLVED remains available when the comparison cannot be established.

Keep this within the existing short mapping/condition responses; do not add the failed eight-field interpretation object. Structured consistency can reject malformed reasoning but cannot guarantee that the model's semantic judgment is true.

**Boundary:** mapping/relation prompts, contracts and validation in `engine/evidence_verification.py`; source-preservation checks in `engine/tribunal_protocol.py`.

**Qualification:** check both directions of each mechanism, literal controls and relevant humor/metaphor/sarcasm examples. Improvement must not depend on matching a caption, keyword, ID or gold label in production.

## T3 — Make a critic rejection specific, and repair only the disputed component

**Observed problem:** the challenge stage can veto a sound proposal using its own faulty interpretation. Conversely, a clean challenge can repeat the proposer's mistake. A correct-label proposal can also have unsupported reasoning and must not be accepted merely because its label matches evaluation gold.

**Change:** the existing challenge must identify the disputed inference and its deciding evidence using its current error, evidence-ID and reason fields. Classify the failure operationally as evidence, role/scope, relation, source/format, or unresolved alternative. Preserve a specific rejection reason in the final audit.

Use one already-budgeted repair opportunity on the failed component. Retain only valid, unchanged upstream components; recheck every dependent component affected by a repair. Resolve a proposer/verifier disagreement by checking the disputed premise, not by counting votes or trusting confidence. A critic's unsupported objection is not automatically ignored: the repaired proof must explicitly resolve it before acceptance.

A proposal with a correct label but an invented premise remains rejected until the premise and inference are actually supported. No global gate loosening, automatic acceptance of disagreement, or special treatment of inspected examples.

**Boundary:** challenge and repair logic in `engine/evidence_verification.py`, acceptance/rejection bookkeeping in `engine/tribunal.py`, bounded generation in `agents/multimodal_judge.py`.

**Qualification:** recover sound blocked proposals, retain rejection of unsound proposals, preserve existing valid corrections, and reject harmful changes. Exercise the complete production gate, not only the judge's text output.

## T4 — Make abstention identify a resolvable requirement

**Observed problem:** 29 of the 31 final abstentions persisted through two hearings; 23 supplied no counter-interpretation. Some requested irrelevant literal evidence for figurative language. Eight abstentions retained wrong answers, but the record does not prove all eight were avoidable.

**Change:** require the existing abstention/follow-up response to identify the exact deciding requirement and distinguish:

- a missing observable fact;
- unresolved caption or cross-modal interpretation;
- conflicting grounded evidence;
- genuinely insufficient information;
- execution or contract failure, reported separately from semantic abstention.

When observations already exist, ask one specific relation question that compares the plausible readings against those observations. When visual information is missing, ask the visual witness for that fact. Caption-only interpretation goes to the language witness; cross-modal inference stays with the tribunal. Do not demand an unrelated literal event or printout of an evaluation.

The second hearing must add a relevant observation, corrected binding or a specific new resolving check. Do not spend it repeating a generic request to reconsider. Preserve justified uncertainty and the maximum of two hearings.

**Boundary:** proposal/follow-up contracts in `engine/tribunal_protocol.py`, routing in `engine/review_routing.py`, follow-up planning in `engine/tribunal.py` and stage transitions in `engine/batch_runner.py`.

**Qualification:** count wrong-to-correct resolutions separately from correct-to-wrong changes among previously abstaining cases. A lower raw abstention count is not a success metric. Inspect whether the missing requirement was actually resolved rather than merely receiving a different label.

## T5 — Keep the repaired tribunal within its output and time budget

**Observed problem:** the old run had a 1,024-token proposal followed by a nearly complete retry that exhausted its remaining time. The later candidate added enough output that none of four proposals completed at the observed low throughput.

**Change:** copy source text and identifiers deterministically where possible, reuse existing concise fields and remove redundant explanations. Repair one invalid component within the shared case deadline; never restart the entire proposal merely because one field failed. Reserve time for the dependent verification work. Do not add an advisor, unrestricted retries, more mandatory calls or a larger default deadline to hide an oversized contract.

An incomplete response cannot authorize a revision. Record budget exhaustion separately from semantic uncertainty, preserve the valid initial answer and committed run result, and report the failure honestly. The system cannot guarantee completion at arbitrary hardware throughput.

**Qualification:** test representative complete responses against the installed tokenizer, then a small real-model check at measured throughput. Fault-inject truncation, invalid source references and deadline exhaustion. Budget feasibility is a prerequisite to interpreting semantic results.

## Implementation and decision order

1. **Freeze the old tribunal baseline and diagnostics.** Keep the original model outputs and historical reference scores. References are evaluator-only.
2. **Implement T1–T3 first, with only necessary T5 support.** Replay frozen proposals through the baseline and revised gate. This isolates acceptance behavior before changing proposal discovery.
3. **Implement T4 after gate qualification.** Then run the complete proposer-to-gate path on a small fixed mixed panel, including harmful candidates, sound blocked proposals, successful corrections and both correct and wrong initial answers that previously abstained. Run in bounded batches of two to four cases. Include general mechanism variants; diagnostic examples are not production rules or an independent accuracy estimate.
4. **Record one comparison and decide.** Require no additional harmful acceptances on the targeted panel, preservation of existing sound corrections, fewer sound proposals blocked, and at least some correctly resolved prior uncertainty with grounded reasoning. Track executions and budgets separately. These are engineering checks, not evidence of held-out safety or conference-level performance.
5. **Stop if the mechanism does not qualify.** If a compact, completed check still confuses the same roles or outcomes despite adequate evidence, investigate one scoped judge/verifier model replacement or training study. Do not keep adding prompts or loosen the gate to manufacture acceptance.

Only after these checks should the user run a full development evaluation. Report correct proposal recall, harmful proposal rate, accepted-change precision, recovered valid proposals, wrong-initial abstentions, execution failures and net accuracy change. Final paper claims require disjoint evaluation and uncertainty estimates.

The immediate aim is more **well-supported correct acceptances** and fewer **avoidable abstentions**, with fewer harmful changes. On the historical final proposals, an oracle gate would have gained five correct answers and reached 40/50; that is a diagnostic ceiling for those proposals, not a prediction or a justification to admit an unsound explanation.
