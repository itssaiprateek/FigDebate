# Causal review of the September 13 precedent run

Reviewed September 14, 2026. Analysis only: no production changes, hardware-setting changes, or new model runs.

Source: `runs/dev10_precedent_20260913_221611/records.jsonl`, its configuration and quality outputs, and the current production review/verification paths. Comparison: `runs/dev10_candidate_20260913_201557`. Images, caption hashes, selection order, model revisions, seeds and initial predictions match. Runtime environment flags differ as recorded in the configurations.

Standing order: improvements must generalize across the target dataset and humor, metaphor and sarcasm. Sample IDs below identify diagnostic evidence only. They must never become inference conditions. Keep the initial arbiter unchanged. Gold/reference explanations are used here after prediction for evaluation, never as inference inputs or feedback memories.

## 1. Timeouts: established mechanism and remaining uncertainty

All three failed final reviews occurred in tribunal round two, after a new witness stage and judge reload. Their first-round generations completed. The code releases each stage's runtime and clears unused memory between stages; the recorded parameter devices remain CUDA. This does not prove physical memory residency or exclude driver-level paging.

| Case | First round generation | Second round | Effective second-round deadline | GPU at timeout |
|---|---|---|---|---|
| 2240 | 269 tokens / 21.71 s; retry 263 / 22.10 s | 120 tokens / 53.98 s | 53.90 s | 480 MHz, 14.57 W, 48 C |
| 1943 | 288 tokens / 22.66 s | 286 tokens / 96.70 s | 96.44 s | 525 MHz, 14.73 W, 47 C |
| 3332 | 286 tokens / 29.53 s | 268 tokens / 89.51 s | 89.44 s | 465 MHz, 14.31 W, 49 C |

First-round completion snapshots were approximately 2,565 MHz and 53–58 W. Approximate decode rates after first token fell from 13–15 tokens/s to 2.6–3.5 tokens/s. JSON grammar callbacks consumed only about 2–4 seconds of the failed calls. These facts support real throughput degradation, rather than a purely oversized JSON answer.

Software contribution: `engine/case_budget.py` and `agents/multimodal_judge.py` reserve 120 seconds for verification inside the cumulative 240-second case budget. Round two therefore had approximately 54/96/89 seconds to propose a decision, not a fresh 120 seconds. About 119 seconds remained when each review failed. This reservation is deliberate protection for proof work, but currently cannot adapt to slow proposal generation.

Timeout memory snapshots reported 2.61–3.03 GiB free and about 3.12 GiB allocated. Prefill cleanup succeeded. Earlier prefill sometimes temporarily exhausted reported free memory. No explicit OOM occurred; persistent out-of-memory is not established as the cause. Endpoint temperature and utilization cannot exclude earlier thermal limiting, CPU stalls, or driver memory migration.

The user confirms Performance mode and the original charger during the run. All saved AC snapshots report plugged in. Do not attribute the run to Silent mode or unplugging.

ASUS documents that Armoury Crate Scenario Profiles can automatically change modes based on the foreground application. Check those profiles as a possible source of transitions, without assuming they caused this run's failure. For sustained inference, the built-in Turbo mode with the included charger is an available controlled comparison; log it as a different hardware condition. Do not manually force clocks or power limits as an unverified fix.

Read-only checks on September 14 identified an ASUS ROG Strix G614JV. The active power scheme was initially Silent and subsequently Turbo; no agent command changed it. An initial idle snapshot showed a 55 W current ceiling against a 100 W default. Clock-event counters were nonzero for software power/thermal limiting, but they are cumulative and were not sampled around the run. None of these current observations dates the limiting to the failed calls.

To determine the remaining cause, the next user-run bounded check must log clock-event reasons and counter deltas, active power scheme, effective power limit, CPU/GPU utilization, dedicated/shared memory and per-call generation timing continuously across judge -> witnesses -> judge. First reproduce in the same declared Performance configuration; compare a different OEM mode separately if needed. A standalone first-hearing replay does not cover this failure path. Do not extend every timeout or lower image resolution as an assumed fix.

## 2. Did rejected feedback hide a correction?

| Case | Initial / gold | Guided proposal | Why not selected | Accuracy effect if simply substituted |
|---|---|---|---|---|
| 4547 | ENTAILS / CONTRADICTS | ABSTAIN | No verified binary resolution | No correction |
| 566 | ENTAILS / ENTAILS | ENTAILS | Independent relation verification invalid | No gain; already correct |
| 1264 | CONTRADICTS / ENTAILS | ABSTAIN | No verified binary resolution | No correction |

No saved guided proposal would improve final accuracy. The useful correction on 1028 came from the ordinary tribunal review, with guided feedback not attempted.

566 is still an informative failed confirmation. The guided reviewer recognized an evaluative reaction to a visible offer. Its blind verifier instead treated the author's criticism as requiring proof of real-world intention/action. First verification attempt ended its explanation mid-clause; the repair attempt fabricated a non-source quote: “the person offering free beer to a woman without shirts”. Both verifier attempts proposed CONFLICT, despite explaining missing evidence rather than a positive opposite. Fixing quotation alone therefore does not establish a valid confirmation; semantic verification also needs attention. Approximately 99 seconds were spent on this guided path, including verification.

The current feedback library contains synthetic methodological principles, not empirically validated precedent cases. Its selection and use are operational, but no incremental accuracy benefit is established.

## 3. Abstentions and hidden stopping reasons

There are four final semantic abstentions, plus three failed final reviews. These are separate outcomes; final answers fall back to the initial decisions.

- 4547: reviewer demands identity/originality evidence for a jersey and does not resolve the possible evaluative/sarcastic reading. Guided feedback declines applicability. Follow-up is skipped at 156.68 seconds remaining because the threshold is 165.
- 1264: reviewer demands literal text for “flopped”, “pandemic” and emotional intensity rather than sufficiently evaluating the reaction-meme mapping. Guided feedback again abstains. Follow-up is skipped at 162.41 seconds remaining. This is an interpretation hypothesis to validate, not permission to discard necessary factual qualifiers.
- 566: reviewer demands proof of intention for an evaluative caption. Feedback makes a plausible correct proposal, but fails verification. Remaining time is 112.29 seconds, preventing a further witness hearing.
- 2893: the requested question asks whether visible text implies ad overload. `followup_plan` routes COUNTER_INTERPRETATION to the visual witness. That witness correctly blocks semantic inference. The pipeline stops instead of routing an appropriate semantic task. The full existing OCR already includes “only ads can be watched”; the failed optional crop is not total OCR loss.

The batch runner records `no_actionable_followup` for both lack of budget and rejected witness scope. The detailed nested audit retains the real cause, but the headline status obscures it.

Full-hearing audit also finds a harmful first-round proposal on 2240, blocked by unsupported visual evidence before round-two timeout. The final-review metric reports zero harmful proposals because it summarizes only the final failed review. Preserve this distinction in future reporting; zero accepted harm remains true.

## 4. Initial visual failures

- 2893 cropped OCR loops over fragments of a repeated logo. Primary generation reaches 180 tokens; retry reaches 360 tokens, still repeating. Total wasted work is 26.59 seconds. Doubling the limit does not repair repetition. Other OCR observations remain available.
- 1264 symbolic cues: the first answer ignores the requested four-item limit and truncates at 96 tokens. The concise retry completes in 42 tokens but contains “Contradiction: relaxed posture vs. intense reactions.” `PROHIBITED_DECISION_PATTERN` rejects any occurrence of “contradiction”, causing a false-positive decision-leak error. Passing validation would still not certify that every described visual detail is true.

## 5. Proposed general changes, in priority order

1. Add continuous bounded runtime telemetry and distinct error codes for GPU/call slowdown, proposal-budget exhaustion, invalid output, witness-scope rejection and insufficient follow-up budget. Record all hearing proposals and feedback overhead, not just terminal verdicts. Validate the complete model-handoff path before claiming a timeout fix.
2. Replace the rigid follow-up admission threshold with a bounded scheduler that accounts for remaining proposal, required witness and proof work. Route necessary new evidence before optional feedback when both cannot fit. Keep a hard total budget, never admit incomplete proof, and return an explicit resource failure if required work cannot finish. Estimate costs from runtime measurements, never gold labels or sample identity.
3. Route observation questions to visual witnesses, caption interpretation to caption witnesses, and cross-modal semantic interpretation to a tribunal step supplied with cited observations. Preserve rejected-question audits and prevent fabricated witness answers. Separate recoverable information gaps from irreducible uncertainty.
4. Bind verifier quotations using source span references whose exact text is supplied by code. Preserve full raw caption, negation, roles and necessary qualifiers. Reject invalid spans. Do not fuzzy-match invented phrases into evidence. Use one short, field-specific repair for a mechanical output defect and then revalidate the entire dependent proof; a changed semantic claim requires renewed semantic verification.
5. Address incomplete free-text and looping OCR with short bounded fields, explicit completion checks and repetition-aware recovery. Retry a repeated OCR stream with a changed region/bounded text task instead of merely doubling tokens. Retain source regions and genuine repeated text at distinct locations; never silently deduplicate or treat cut-off text as complete evidence. Unrecoverable output becomes typed unavailability, not usable evidence.
6. Replace keyword-only visual answer leak detection with detection of actual task-decision statements. Allow ordinary descriptions of contrast while still blocking assertions of the dataset verdict, gold answer or final prediction. Test both false positives and real leaks. This changes the visual adapter only, not initial arbiter logic.
7. Align tribunal and blind verifier on the task's treatment of author evaluation, quoted speech, conventional idioms, sarcasm and reaction-meme analogies. Require the visual trigger and justified mapping while distinguishing them from assertions about invisible intentions or real-world outcomes. Missing evidence must remain UNRESOLVED, not automatically CONFLICT. Necessary qualifications must not be dropped to reduce abstention.
8. Improve precedent retrieval by diagnosed reasoning difficulty and applicability, then use only independently reviewed training precedents with counterexamples and enforced image/caption/group separation. Log which interpretation a precedent changed and which current evidence supports it. Keep verifier blind to precedent guidance. Compare feedback on/off with matched additional compute so benefits cannot be explained solely by another attempt.

## 6. Bounded acceptance checks before another dataset run

Use regression artifacts for routing, budgeting, source binding, incomplete output, genuine versus false leakage and OCR repetition. Include previously unseen wording and opposite-label controls across all three figurative types. Failures must remain visible and invalid evidence must never reach the gate.

Any user-authorized live diagnostic should include an ordinary hearing, witness model handoff and second tribunal review with continuous telemetry. Check the actual failure path, not only a standalone judge replay. Follow with a small frozen evidence comparison of tribunal/feedback behavior; report corrections, harms, unresolved judgments, execution errors and time separately. No universal accuracy gain or zero raw model-error guarantee is justified.

Operational target: no malformed output admitted to downstream evidence, one bounded mechanical repair, explicit recoverable failure if still invalid. Semantic target: fewer avoidable abstentions at a controlled harmful-change rate, not a hard quota compelling guesses.

Official hardware references:
- NVIDIA clock-event reasons and counters: https://docs.nvidia.com/deploy/nvidia-smi/
- ASUS operating modes: https://rog.asus.com/articles/guides/armoury-crate-performance-modes-explained-silent-vs-performance-vs-turbo-vs-windows/
- Microsoft power-mode controls: https://support.microsoft.com/en-au/windows/change-the-power-mode-for-your-windows-pc-c2aff038-22c9-f46d-5ca0-78696fdf2de8
