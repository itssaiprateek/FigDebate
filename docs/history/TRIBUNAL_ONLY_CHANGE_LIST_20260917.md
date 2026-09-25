# Tribunal-only diagnosis and change list

Status: proposed changes only. Production code is unchanged. Feedback design and second-hearing improvements are deferred. The investigation used saved runs, first-hearing checkpoints and CPU-only contract/decoder checks; no new model inference was run.

## Conclusion

The Review 5 changes did not qualify. They combined a smaller practical proposal/retrieval allowance, a shorter proposal representation, independent evidence selection and a revised challenge contract. These changes exposed existing format/retrieval problems and failed to improve semantic reasoning. Treating the result as merely an overly strict gate would lead to harmful acceptances.

The matched historical panel declined from 9/10 to 8/10, with the same initial 8/10 labels and the same image/caption/reference identities. This is an observed regression on ten development cases. It does not isolate the causal contribution of each changed component or establish a dataset-wide accuracy difference.

## Tribunal performance before feedback

The ten first-hearing checkpoints show:

- Six final valid binary proposals: three harmful changes and three correct-label confirmations.
- Two abstentions and two invalid/failed proposals.
- Four proof chains reached the challenge. All four were blocked solely at the challenge audit: three correct-label confirmations and one harmful change.
- Five execution/contract/context failures, four rejected proposals and one genuine semantic-uncertainty terminal outcome.
- Zero verified conclusions and zero accepted corrections before feedback began.

An important refinement to the previous summary: **zero final corrective proposals does not mean no earlier correct-label drafts existed.** On 1028, two drafts said CONTRADICTS before retrieval replaced them. They took 31.82 and 27.85 seconds. A third full generation timed out at about 60 seconds. Neither draft had an independently verified proof, and both leaned on missing evidence about the subscription upgrade. Retaining their label alone would not establish a sound correction.

## Where the problems lie

| Location | Confirmed behavior | Relationship to the new changes |
|---|---|---|
| `engine/case_budget.py:10` | The proof reserve became a fixed 120 seconds; the historical 1028 review used 75 seconds. New 1028 stopped with approximately 120 seconds still reserved and no proof run. | New policy. Order-independent budgets are desirable, but this allocation was not qualified against the cost of repeated proposals. |
| `agents/multimodal_judge.py:785` | Every context retrieval runs a whole new proposal and replaces the prior one. After two cycles, a further context request causes `CONTEXT_NOT_DISCLOSED`. | Existing controller, interacting badly with new proposal cost and budget allocation. |
| `engine/compact_proposal.py:10` | Source spans replace copied text; repeated observation and role/scope generation were removed. The role/scope field is now a software placeholder until later independent mapping. | New representation. Its contribution to wrong reasoning is a hypothesis requiring a frozen-input comparison, not an established cause. |
| `engine/evidence_review_v5.py:24` and `:219` | Alternative-first challenge responses repeatedly emit the placeholder `UNRESOLVED`, sometimes copying the relation explanation or listing `"None"` as an error. | New order/instructions and additional acceptance requirements; observed failure is confirmed, but ordering versus prompt/model capability is not isolated. |
| `engine/evidence_verification.py:219` | Source-span errors in verifier outputs return a general invalid-output result. The new narrow span repair exists in the proposer, not throughout verification. | Incomplete repair coverage. The underlying source-span mechanism predates the new proposer. |
| `engine/output_contracts.py:36` | The shared shape validator has no integer type/range branch: it accepts a string, Boolean, negative number, out-of-range number and fractional number against an integer schema. | Pre-existing defect confirmed by offline tests. The source binder still rejects invalid spans, so this does not mean invalid source quotes are accepted. |
| `engine/output_contracts.py:73` | The unfinished-clause heuristic misses endings such as `or` and partial words. Character caps can close long generated fields mid-thought. | Pre-existing limitation exposed in the new run. A format-valid field is not proof of a complete or sound argument. |

### Reasoning errors are independent of formatting

The model repeatedly treats absent support as contradiction or changes what is being compared. Examples in the saved traces include an included album supposedly disproving dislike of that album, and an image saying only advertisements can be watched supposedly contradicting a complaint about too many advertisements. The blind relation verifier repeated the latter error.

The prompt already says missing evidence is not contradiction. The validator already requires a condition labelled CONFLICT. The model can still label an invalid comparison CONFLICT and satisfy that structural test. Adding the same instruction again or requiring another agreeing Boolean would not solve this.

### The decoder does not force abstention

A CPU check using the actual local tokenizer and production grammar accepted both a well-formed `NONE` challenge and the unhelpful `UNRESOLVED` placeholder. Therefore a hard-coded grammar restriction is not forcing every challenge to abstain. The contract permits vague outputs, and the model is selecting them.

An explicitly unsafe, in-memory counterfactual replaced the challenge outcome with `NONE` and cleared error lists while leaving other proof components untouched. All four first-hearing completed proofs then passed the proof audit, **including harmful proposal 2893**. These are deliberately modified diagnostic objects, not model results, accepted changes or a proposed fix. They establish why removing the challenge restriction would be unsafe.

## Ordered change list

### 1. Fix transport validation and one-field recovery first

Implement strict integer type/range validation, rejecting Booleans as integers. Keep the separate cross-field condition `0 <= start < end <= token_count`; integer checks alone cannot detect an empty interval. Share the same source-span error and single-field repair mechanism across proposer, mapping and relation responses.

For an invalid generated span, request that span only within the existing single format-retry allowance. Revalidate the complete retained response and re-run affected proof obligations. Never silently expand or clamp a quote, invent a source condition, accept a partial response or grant additional retries after the allowance is consumed.

Give attachment fields an explicit role: a short region/speaker/panel description, not a repeated OCR paragraph. Preserve exact source fragments separately from generated explanations. Detect clear unfinished generated clauses and expose length-boundary diagnostics; do not classify every source fragment or missing period as an error. A short incomplete field gets its existing bounded repair, not a full new argument.

**Check:** replay recorded malformed outputs plus synthetic empty, reversed, fractional, Boolean, out-of-range and valid intervals. Confirm repairs cannot alter a verdict or unrelated evidence silently. Exercise the production generation wrapper with scripted outputs and then a bounded live contract check. No accuracy claim follows from parser success.

### 2. Finish context disclosure before regenerating a final proposal

Replace the repeated “full proposal → retrieve → full proposal → retrieve → full proposal” sequence with a bounded information-selection step followed by one final proposal. Use existing context and evidence IDs, retrieve requested records together, and keep the selected records protected in the final packet. Required caption/hearing information must not be displaced by optional candidate commentary.

Distinguish selecting information from making a decision. The information-selection response should contain only requested IDs and the specific missing requirement, not another complete verdict and explanation. Do not add an unrestricted retrieval loop or enlarge every packet. If essential information still cannot fit or does not exist, stop with that precise reason.

Keep prior drafts in the audit. A later failure must not masquerade as “the judge never found an alternative,” but **an unverified earlier draft cannot become the final answer**. Only an already verified conclusion can survive as a committed review result; changed evidence invalidates affected proof components.

Retain an order-independent, predeclared budget and the existing case/call ceilings. Recalibrate the proposal/retrieval/proof allocation on a separate representative cost panel; do not choose a reserve to make 1028 pass or restore dependence on earlier samples. The completed first-hearing proofs here took about 39–53 seconds, but four examples are not enough to set a safe universal reserve. Reduce redundant generation before redistributing time.

**Check:** multiple context needs are disclosed without repeated full proposals; no unread record is cited; source evidence remains intact; case order cannot change eligibility; failure accounting identifies the exact unfinished phase. Demonstrate completion under the actual local budgets.

### 3. Restore a concise, explicit comparison before the verdict

Use the existing V4 proposer as the control on frozen upstream inputs. Test the shortened V5 proposer separately; do not assume removal of role/scope work is harmless because a later verifier exists. Retain independent evidence selection only if its separate comparison preserves useful proofs and improves coverage.

The tribunal must identify the source assertion, the matched subject/property/scope and the actual observed state before choosing the relation. Keep this within the existing compact condition/mapping fields; do not revive the large interpretation schema or add another agent. Conventional figurative correspondence must be distinguished from literal identity, speaker reaction from an object's property, and actual outcomes from intended ones.

Require a real positive incompatibility for CONFLICT and established necessary conditions for SUPPORT. Detect mechanically contradictory global and condition-level fields and request a bounded correction; do not silently relabel the answer. Do not use English keyword rules or another self-reported “verified” flag as a substitute for semantic validity.

**Check:** both relation directions, matching versus swapped roles, absence of evidence versus contrary evidence, evaluative reactions, literal controls and licensed figurative readings. Include correct initial answers and sound corrective proposals; assessment labels stay evaluator-only. If the current model fails these distinctions with clean inputs, stop expanding prompts and run a separately scoped verifier/model comparison with the same contract.

### 4. Make the challenge return a concrete audit outcome

Keep one challenge call. Clearly define the existing fields and their allowed combinations in a concise response contract visible to the model. Test the previous error-first order against the new alternative-first order on identical completed proofs; do not declare either order superior without that comparison.

A genuine unresolved result must name a concrete material alternative or a specific undecidable condition. An objection must identify the disputed inference/attachment and deciding source evidence. No material objection is a legitimate outcome, with empty error lists, no fabricated alternative, cited deciding evidence and a concrete reason. `"UNRESOLVED"` as the entire alternative and `["None"]` as an error list are contract defects, not meaningful reasoning.

Repair an inconsistent challenge once within its existing allowance. Never convert vague uncertainty into `NONE`, drop objections automatically, or increase acceptance merely by normalizing text. A challenge that cannot establish its conclusion must remain unqualified. The source/visual/mapping/relation checks remain mandatory.

**Check:** a complete, grounded counterargument can block a harmful proposal; a justified no-objection result can let a sound proposal proceed; genuine uncertainty remains allowed; placeholders cannot authorize a decision. Include the harmful frozen proofs as well as useful ones, and evaluate explanations rather than only labels.

### 5. Qualify the tribunal alone before feedback work

Keep the current normal profiles on V4 and the candidate opt-in. Freeze upstream states and disable both legacy precedent retries and integrated feedback for this comparison. Keep infrastructure fixes—source archives, immutable records, bounded batching and honest failure categories—separate from semantic experiments.

Apply the above changes one group at a time, starting with deterministic/offline checks. Compare each semantic component against its control on the same predeclared small, mixed development panel, with identical budgets and no gold in model inputs. Follow successful engineering checks with disjoint image/template groups rather than repeatedly tuning these ten cases. Larger evaluation remains user-run.

Promotion requires completed justified decisions, preserved sound corrections, additional sound corrections on the chosen panel, no added harmful acceptances on that panel, and reproducible handling under reordered cases. These selected-case requirements do not guarantee dataset-wide accuracy or zero future errors. If the first-hearing tribunal still has no useful marginal effect, do not add feedback calls to conceal that failure.

## Scope and stopping point

No initial-arbiter changes, sample-specific prompts, caption/ID lookup rules, new memory/advisor subsystem, forced binary answers, unconditional larger timeouts or lower gate thresholds are proposed. Feedback remains deferred. This is a finite tribunal repair and qualification plan, not a promise that every implementation item will improve model accuracy.

Supporting artifacts: `research/tribunal_only_plan_20260917/diagnose.py` and `diagnosis.json` in the enclosing workspace. The earlier complete run review remains in `research/dev10_review5_20260917/REPORT.md`; its proposal counts refer to final retained reviews, while this investigation also inspects discarded retrieval drafts.
