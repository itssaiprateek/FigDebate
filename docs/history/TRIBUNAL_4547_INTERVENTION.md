# Single-case interpretation intervention

Source: `runs/dev10_reliability_20260914_102054`, case `vflute_train_4547`. Diagnostic only; this case has informed development and is not held-out evidence.

The original two hearings abstained because the image did not establish the third-kit identification or design originality. The initial ENTAILS answer remained, whereas the dataset label is CONTRADICTS.

On visual inspection the image contains a single purple football jersey with geometric pattern, yellow accents, sponsor and club marks. It supplies no comparison establishing novelty. The dataset explanation interprets the caption's praise as sarcastic disappointment. This interpretation is additional annotation-informed guidance, not a new observation.

## Intervention and result

The replay used the frozen first-hearing state, original image/caption, seed 42, and production proposer/acceptance path. An explicitly oracle-informed explanation of sarcastic disappointment was placed in the tribunal-only semantic task packet. It supplied no output label, added no visual evidence, and did not change the verifier or acceptance conditions. Inspection of the saved judge packet confirms delivery.

Result: UNRESOLVED, final ENTAILS, acceptance reason `tribunal_not_resolved`. Execution succeeded, with valid output. Proposal time was 42.51 seconds; total replay including model loading was 70.31 seconds. Independent verification was not invoked because there was no binary proposal.

The model's final reason was: “Caption claims specific kit number and design quality; image lacks supporting text or visual proof.” It separately marked both the kit identity and originality clauses unresolved. The intervention therefore did not resolve the literal condition-checking behavior. This experiment does not establish whether a correctly grounded binary proposal would pass the verifier.

Artifacts: `runs/diagnostic_4547_interpretation/manifest.json`, `records.jsonl`, `complete.json`; supplied guidance: `runs/4547_diagnostic_guidance.txt`. The only source edit adds an opt-in, single-case diagnostic-guidance argument to `evaluation/qualify_tribunal_pair.py`. Production code is unchanged.

## General improvement to investigate

Introduce an explicit interpretation stage before condition checking: separate expressed assertion, possible intended stance, background reference, and decisive image-caption relationship. Preserve all caption content while distinguishing what each clause contributes. Compare literal and figurative readings using the same evidence, then select a reading with an explicit reason and uncertainty. Do not automatically reverse sarcasm or declare a jersey unoriginal from appearance alone.

Follow-up planning should name the missing decisive information and ask only a source capable of providing it. A pixel-only witness cannot retrieve image metadata or historical design comparisons. Repeating the same missing condition should terminate with a specific evidence limitation, not another identical question.

Validate this mechanism with varied evaluation, reaction analogy, idiom and literal-fact cases, including cases where superficially plausible sarcasm would cause a harmful flip. Measure interpretation quality separately from verification and final correction. Do not implement a rule for this caption or relax the gate to reproduce its gold label. The single image leaves a real evidence limitation; a task-level policy for contextual interpretations must be defined and evaluated across cases before changing acceptance requirements.
