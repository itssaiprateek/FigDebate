# Bounded change list for project completion

Status: proposed implementation and qualification plan, grounded in the saved development run, source audit and primary research. No production changes were made for this plan. Timing optimization is deferred. The initial arbiter remains unchanged. All production changes must apply across examples and phenomena; sample IDs and gold answers are permitted only in diagnostic fixtures and post-hoc scoring.

## 1. The objective we can actually test

The tribunal cannot know that the initial answer is correct without an oracle. The gate is also a fallible decision procedure, not a correctness oracle. The release objective is therefore: discover and correct more initial errors, preserve correct decisions, and decline revisions when available evidence does not support a safe change. Some initially wrong cases may remain unresolved. A schema-valid proof is not a mathematical guarantee of semantic truth.

Baseline from the 50 latest tribunal checkpoints: 35 initial and 35 final correct; 2 corrections and 2 harms; 31 semantic abstentions; 5 correct and 4 harmful final change proposals; 3 correct proposals rejected; 22 feedback attempts with no accepted correction. Final export failed. These checkpoints are provisional until recovered into a complete report with provenance.

Proposed development release criteria, to freeze before implementation:

- At least 40/50 correct with the same frozen upstream decisions (at least +10 percentage points over the current 35/50), with at most one harmful accepted change. This requires at least five corrections with zero harms, or six with one harm. These are engineering targets, not results or statistical guarantees.
- Correct final change proposals should substantially outnumber harmful proposals; use at least 80% proposal precision as the development target. Report the numerator and denominator, not just a percentage.
- Do not impose an arbitrary overall abstention quota. Track abstention among initially wrong cases separately from abstention among correct cases, together with final correction recall and harms.
- Advisor-on must improve corrections or prevent harms over advisor-off on matched cases without worsening net outcome. A matched confirmation or an irrelevant precedent is not evidence of advisor value.
- All selected cases have a durable terminal result, recoverable progress, and complete export accounting. Execution failures remain explicitly labelled; a fallback answer must not disguise a failed model operation.
- Full evaluation, ablations, explanation assessment and held-out validation remain necessary after implementation qualifies. The inspected dev50 is a regression/development set, not untouched final evidence.

These are proposed operational targets. If they are not met, the implementation is not qualified; do not redefine success after seeing results.

## 2. Root problems and strength of evidence

| Area | Established finding | What remains a hypothesis |
|---|---|---|
| Interpretation | 29 of 31 abstentions persisted after two hearings; 23 final abstentions offered no competing reading. Outputs repeatedly demand literal evidence for evaluation, idiom and analogy. | How much a structured reading stage will recover on the existing 4B judge. |
| Representation | All 50 graphs use the intentional full-source fallback. The old decomposer was not qualified. | Re-enabling that decomposer would help; it must not be done without qualification. |
| Verification | Proposer, fresh relation call and challenge repeated the weather and Flash-bike errors. Useful proposals also failed due to wrong interpretations inside verification. | A different model or an extra call alone would fix correlated errors. |
| Advisor | Only unresolved proposals are eligible; predicted 'literal' excluded Flash's bike; verifier never receives advice. All four binary guided results rejected precedent applicability. | A relevant advisor using verified observations will improve decisions. |
| Completion | A replacement-blocking file handle reproduces WinError 5 in the unchanged exporter; CSV export precedes progress update. | Which application held the historical handle. |

## 3. Research that informs the design

- **V-FLUTE / Understanding Figurative Meaning through Explainable Visual Entailment:** task meaning must be preserved. In the sarcasm construction, the image contradicts the expressed/literal claim; rewriting it into the intended criticism can change the question being evaluated. Their analysis separates hallucination, unsound reasoning and incomplete figurative reasoning. The implication for this project is to preserve both source assertion and possible intended stance, with explicit separation. Do not use source-dataset identity as a label rule. [Paper](https://arxiv.org/html/2405.01474v2)
- **VISCO / LookBack (CVPR 2025):** model-generated critiques can hurt correction, while specific human critiques help. LookBack converts reasoning information into questions and checks it against the image before producing critique. FigDebate already performs atomic visual checks, so merely adding more OCR is not the contribution. The proposed adaptation is checking decisive coverage and semantic attachments before accepting criticism. Its results are not a guarantee for this judge or dataset. [Paper](https://arxiv.org/html/2412.02172v2), [Hugging Face](https://huggingface.co/papers/2412.02172)
- **Large Language Models Cannot Self-Correct Reasoning Yet (ICLR 2024):** intrinsic correction without reliable external feedback can degrade reasoning. This supports treating repeated same-model agreement as fallible, not proof. It does not establish that all self-correction is impossible. [Paper](https://arxiv.org/abs/2310.01798)
- **CRITIC (ICLR 2024):** grounded, tool-interactive critique improves correction on its studied tasks. The transferable principle is that advice should cause a concrete evidence check, not another unsupported opinion. External tools in those tasks are not equivalent to a perfect visual or figurative-language oracle. [Paper](https://arxiv.org/abs/2305.11738), [Hugging Face](https://huggingface.co/papers/2305.11738)
- **MAPPER (FigLang 2024):** separates description from task reasoning and uses a fine-tuned multimodal thinker. This is a relevant same-task alternative if the existing judge fails a bounded capability check; its result is not evidence that a prompt-only replacement will reproduce it. [Paper](https://aclanthology.org/2024.figlang-1.12/)
- **Learn then Test:** risk control requires held-out calibration and a defined loss, rather than an arbitrary confidence threshold. For FigDebate, use harm-aware validation of a finite set of acceptance policies. Formal guarantees would require an appropriate statistical implementation and assumptions; 50 repeatedly inspected examples are not sufficient certification. [Paper](https://arxiv.org/abs/2110.01052)

## 4. The concrete changes

### C1 — Freeze the task contract and add a small interpretation record

**Problem:** the tribunal jumps from raw caption and observations into global SUPPORT/CONFLICT/UNRESOLVED checking. It sometimes reasons about the intended sarcastic message while the verifier reasons about the expressed assertion, or demands literal proof of an idiom.

**Change:** keep the original caption and full-source obligation authoritative. Add a bounded record alongside it, not a replacement paraphrase or a revived unqualified graph. It identifies source spans, asserted meaning, possible intended stance, relevant subjects, actual versus desired outcomes, necessary qualifiers, and one or at most two plausible readings. Each reading must name a deciding observation or a genuinely missing prerequisite. Unsupported readings stay hypotheses.

The label must answer the original image-caption relation. Conventional idiom interpretation is allowed; silently reversing a sarcastic source claim is not. Background information cannot be erased simply because it is inconvenient to verify. A decision must explain why an unresolved detail is or is not material to the relation.

**Implementation boundary:** `engine/tribunal_protocol.py`, `engine/tribunal_interpretation.py`, `engine/case_dossier.py`, and judge proposal handling. Preserve `engine/claim_graph.py` source fallback and the initial arbiter.

**Check:** source/qualifier preservation; literal and figurative controls; correct direction under role/outcome changes; original-image task consistency. A fallible model declaration that a paraphrase is equivalent is insufficient.

### C2 — Make the advisor identify errors and resolving checks

**Problem:** current feedback is generic retrieval followed by another proposer attempt. It cannot inspect a confident wrong binary proposal and can be excluded by a mistaken type label.

**Change:** replace that retry with one bounded advisor assessment, eligible for a proposed revision or a specific unresolved operation. Eligibility is based on task structure and evidence, not solely humor/metaphor/sarcasm classification. Use the existing language model as a text critic over source-anchored caption and verified observations in the first feasibility experiment; it must not assert unseen visual facts. The original image remains available to the tribunal and visual witness. Different model weights do not establish independent errors.

Return only: issue type, source/evidence references, the competing reading, a precise resolving check and uncertainty. The advisor is allowed to find no issue. It must not invent an error to justify its existence. An irrelevant precedent is explicitly excluded. Any earlier sound decision remains available during repair.

For the proposer, advice explains the disputed relationship. For the verifier, pass a neutral check question and references, not an expected verdict or the advisor's authority. Record whether the requested check was actually answered and whether it changed a decision or prevented a harm.

Keep the synthetic library clearly labelled. Do not build a gold-informed memory from this dev run. Empirical memory, if ever needed, must use separately reviewed training examples disjoint by image, caption and template/group; it is not part of this first bounded revision.

**Implementation boundary:** `engine/tribunal_feedback.py`, `agents/multimodal_judge.py`, `engine/review_routing.py`, and packet/audit fields.

**Check:** advisor reaches confident wrong proposals and mislabeled literal cases; irrelevant advice cannot be counted as a success; accurate decisions can survive criticism. Compare advisor-on and advisor-off with identical cases and starting state.

### C3 — Verify decisive evidence and relation consistency

**Problem:** proving that a selected quotation is visible does not prove that it covers the claim or entails the chosen direction. Free-text challenges can introduce their own mistakes.

**Change:** adapt the existing visual check to verify coverage of the decisive object, panel, speaker or time range. A whole-forecast claim cannot be supported by checking only the current-condition heading. Keep all evidence registered against the original image. Ask the blind relation verifier to state the subject and actual outcome before its direction. These fields distinguish, for example, a reaction in a reader from usefulness of the depicted object.

The critic challenges a specific inference, not the mere existence of a false source claim. Each critical objection needs evidence and a proposed correction or check. If the critic invents an observation or confuses an intention with an outcome, repair that component once; do not discard a correct proposal solely because the critic produced an unsupported objection. If it cannot be resolved, retain the initial answer with an explicit unresolved reason.

The deterministic gate checks case identity, source coverage, bindings, proof completeness and unresolved objections. Model-provided semantic judgements remain fallible. Stop describing a corroborated bridge as certification of a factual opposite when its only basis is agreement among calls. Use calibrated acceptance rules only after the complete pipeline is frozen and separate calibration data are available; never threshold the current unelicited `confidence=0.0` field.

**Implementation boundary:** `engine/evidence_verification.py`, `engine/independent_review.py`, `engine/semantic_bridge.py`, `engine/tribunal.py`.

**Check:** preserve genuine corrections, block both observed harmful mechanisms and varied counterparts, and recover correct proposals rejected by an erroneous component. Gold is used only by the evaluator.

### C4 — Bound hearings and recover only the failed component

**Problem:** a second hearing often repeats the first uncertainty; token exhaustion triggers another whole response that can exhaust the remaining deadline.

**Change:** keep a maximum of two hearings. A second hearing requires a new observation, corrected attachment, or explicit new competing-reading check. Ask one resolving question; reject requests to obtain metadata or historical comparisons from pixels alone. Record the missing requirement when it cannot be obtained.

Split a large proposal into bounded source-bound components only where the 1,024-token failure demonstrates need. Repair a malformed field/component once within a shared deadline. Never splice arbitrary truncated JSON into an accepted decision, and never silently truncate source evidence or qualifiers. Preserve successful components only when their provenance is unchanged. A persistent failure becomes a recorded failed stage with a valid final fallback and honest completion accounting.

**Check:** repeating the same uncertainty earns no extra hearing; token exhaustion followed by insufficient retry time ends safely; malformed nested explanations, stale proof reuse, timeout and unknown evidence cannot authorize a revision. No timing optimization is included.

### C5 — Make result finalization independent of spreadsheet availability

**Problem:** replacing a locked CSV aborts finalization after model work is complete.

**Change:** durable per-case final records are authoritative. Persist and reconcile progress from committed records. Export the CSV as a derived artifact with bounded replacement retries. A persistent lock produces `inference_complete_export_pending` or equivalent honest status and preserves retryable data. A separate offline export command finalizes reports without loading models. Write a completion manifest only after selected IDs, unique records and checksums agree; track sample-stage failures separately from export status.

Restore this failed run into a new recovery directory, checking checkpoint identities and comparing reconstructed records with the 34 durable originals. Do not rewrite the original failure history. Distinguish recovered decisions from post-run inference.

**Implementation boundary:** result persistence/export helpers, `run_figdebate.py`, final checkpoint writing in `engine/batch_runner.py`, and an offline recovery utility.

**Check:** locked CSV (already reproduced with WinError 5), lock release, persistent lock, interruption after record commit, duplicate resume, partial final JSONL line, missing/mismatched checkpoint, and offline export with model loading prohibited. Do not claim arbitrary disk loss or permanent access denial can be made impossible.

## 5. One qualification cycle with a decision point

1. **Offline foundation:** implement C5 and source/routing/provenance invariants. Recover the 50-case report. No inference is needed.
2. **Fixed capability probe before the complete advisor integration:** on a small preregistered mixed set, compare existing critique, a human-written specific critique, and the proposed advisor. Include correct, wrong and uncertain starting decisions; include examples outside the inspected dev cases. Human/oracle-informed guidance is diagnostic only. The prior 4547 intervention already showed that an intended reading is not enough for every case; do not optimize around making that one case flip.
3. **Implement C1–C4 as one versioned candidate:** fixed models, stage limits, interpretation contract and evaluation metrics. Replace redundant work rather than piling another unrestricted debate on top. Run targeted live checks through the real acceptance path plus fault-injection tests.
4. **One advisor ablation and one qualification comparison:** use matched frozen upstream inputs to isolate tribunal effects, then an end-to-end user-run dev50. Freeze results and report all harms, not just improvements.
5. **Decision:** if the candidate meets the predefined development and reliability criteria, freeze architecture and move to paper evaluation. If specific valid critique still fails to produce correct decisions, stop prompt tuning: the judge capability is inadequate. If valid critique works but the advisor cannot produce it, the critic is inadequate. In either case choose one scoped model/critic training or replacement study, or close the project with a narrower/negative claim. Do not start another unlimited prompt-and-rerun loop or pretend the target was reached.

No fixed number of days can be honestly guaranteed before the capability probe. Passing it provides a credible boundary between engineering completion and the remaining evaluation work. A few-day implementation window is an estimate to make after that result, not a research performance promise.

## 6. Paper-ready endpoint

Once the architecture is frozen and engineering checks pass, remaining work includes disjoint calibration/evaluation, baselines, advisor and tribunal ablations, per-phenomenon results, paired correction/harm statistics with uncertainty, explanation quality assessment, release provenance and reproducibility instructions. Improvement on 50 repeatedly inspected cases alone does not establish conference readiness. Claims must match the final measured contribution; if advisor benefit is not established, it must not be advertised as a useful mechanism.

This list is the bounded proposal, not a claim that its changes have already improved accuracy.
