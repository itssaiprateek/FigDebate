# Completion changes and qualification — 14 September 2026

**Later update: the user requested removal of the experimental candidate. Its live advisor, expanded generation path, profile and test-run option have now been removed. The availability statements below describe the earlier implementation state. Historical results and read-only audit support are retained. Reliability fixes and the normal tribunal remain.**

The persistence and export changes passed focused checks. The proposed semantic tribunal/advisor candidate was implemented but **failed live execution qualification**. It is disabled in normal profiles. This is not a paper-ready release and does not establish increased correction accuracy.

The initial arbiter was left unchanged. Production changes contain no rules keyed to dataset sample IDs, captions, image hashes or expected labels. Case identities in diagnostic fixtures and checkpoint integrity checks serve provenance only. No full dataset inference was run during this implementation.

## Enabled changes

- Each finished sample is committed independently under `final_records/` before progress is updated. The stage runner also saves a `final_result` checkpoint, including its final decision and accounting.
- CSV and report export happens after result commits. Replacing a locked prediction file has three bounded attempts. Persistent export failure records `inference_complete_export_pending`, preserves results and returns a nonzero command status. Derived timing/report failures receive the same treatment.
- Resume reconciles committed files with the JSONL export. An incomplete final JSONL fragment can be recovered; conflicting duplicates, completed corrupt records and checksum mismatches are rejected. Completion requires the exact selected ID set and a manifest containing record hashes.
- Export retry preserves an existing timing report rather than replacing the inference duration with an almost-zero export-only duration. Resumed inference timing is explicitly scoped to the current invocation.
- Questions requesting unavailable metadata or historical evidence are blocked from pixel witnesses. Caption interpretation remains a language task. A duplicate semantic question does not justify another hearing; only one new question is carried forward. This checks textual duplication, not equivalence of every possible paraphrase.
- Incomplete nested critic explanations are rejected. When advisor checks are present, they are included in verification identity so a proof cannot be reused with different advice.

The normal `paper-8gb`, `8gb`, `12gb`, `16gb` and automatic profiles retain the previous semantic tribunal/feedback path. Its previously diagnosed accuracy limitations remain open.

## Implemented, but disabled after failed qualification

The explicit `completion-candidate-8gb` profile retains the candidate for diagnosis. Its upstream model settings match `paper-8gb`.

1. **Interpretation contract:** preserve the immutable source caption; separate its assertion, intended stance, participants, actual and desired outcomes, competing interpretation and distinguishing fact.
2. **Advisor:** run a bounded Mistral critique on the caption and existing grounded observations while Mistral is already loaded. It receives no gold label, reference explanation or prior decision. Advice goes to the proposer; only a neutral resolving check goes to verification. Advice is never image evidence or gate authority. Under this candidate, the legacy `precedent` option selects this implementation; the precedent library is marked inactive.
3. **Verification:** require explicit evidence coverage, subject/property and outcome checks. Missing support cannot certify contradiction. Critic objections must identify supplied evidence; unsupported objections are not silently converted into a clean proof.
4. **Component repair:** one bounded repair may retain only fully decoded, schema-valid components and regenerate missing components. The whole result must pass validation. Timeout fragments never become accepted decisions. This mechanism does not guarantee recovery when the remaining time cannot accommodate a valid response and its verification.
5. **Reporting:** separately record advisor attempts, valid/failed advice, time and issue types. Reports explicitly state that accuracy contribution requires a matched ablation.

These are implemented mechanisms, not established semantic improvements. The candidate is deliberately not enabled by ordinary run commands.

## Focused checks and results

### Software checks

`runs/completion_final_unit.log`: **221 tests ran; 220 passed, one existing expected failure, no skips.** Local tokenizer integration checks were included without loading model weights. The expected failure concerns initial-arbiter scoring conditioned on its assessment; it predates this implementation and remains outside the authorized arbiter scope.

Checks cover committed-result recovery, corruption and duplicate rejection, persistent export locks, successful export retry, exact completion IDs, actual finalization-path failures at multiple exporters, component repair, evidence/proof integrity, routing, profile isolation and tribunal integration.

`runs/export_lock_diagnostic.json`: the actual prediction exporter was tested against a Windows file held without delete sharing. It reproduced **WinError 5**, preserved the original file, and succeeded after release. This does not promise that permanent disk/access failure is preventable.

### Bounded live checks

The first probe used two saved hearings with specific human-written semantic critiques. Those critiques were oracle-informed diagnostics, not accuracy evidence. The old judge still accepted the harmful bike interpretation and still blocked the correct outcome interpretation. Advice alone did not resolve either case. Results: `runs/completion_capability_pair/`.

The candidate then ran four saved first hearings through the real proposer and production acceptance path, with automatic advisor generation. Initial arbiter and new witness inference were excluded.

| Diagnostic case | Advisor | Tribunal | Final outcome |
|---|---|---|---|
| 2850, subject/property ambiguity | Valid | Timed out | Original label retained |
| 1841, whole-image coverage | Valid | Timed out | Original label retained |
| 3351, actual versus intended outcome | Timed out | Timed out | Original wrong label retained |
| 1028, known correction opportunity | Timed out | Timed out | Original wrong label retained |

The four-case process took **811.55 seconds**, including loading models and generating advice. Each tribunal call stopped after approximately 121 seconds. Observed decode speed was **2.62–2.81 tokens/second**; the larger contract did not complete within the per-call deadline. Two advisor outputs were incomplete at their 60-second limit. Partial outputs were rejected and no changes were accepted.

No harm was introduced in these four failed reviews, but that is **safe failure, not a successful harm-reduction result**. There was no correction benefit. The revised verifier could not be semantically qualified because proposal generation failed first. Turbo was active and only one model worker was observed; these facts do not establish the cause of the throughput reduction. No power settings were changed.

Results and compact diagnostics: `runs/completion_candidate_advisor4/qualification.json`. The original process used `paper-8gb` while its structured-interpretation flag was true. After failure, that flag was moved to the explicit candidate profile; the original run manifest was not rewritten.

No advisor-on/off ablation or further inference was launched after this failed qualification. Continuing would not support a meaningful accuracy attribution.

## Recovery of the failed 50-case run

`runs/dev50_recovered_completion/` contains all 50 results and restored reports. Recovery retained the 34 durable original records and reconstructed the remaining 16 from validated completed tribunal checkpoints. Input identities, checkpoint fingerprints, payload hashes and decision agreement were checked. The source run was not rewritten and **no model inference was performed**.

Recovered initial/final accuracy remains **35/50 (70%)**, with two helpful and two harmful accepted changes. This is recovery of the old experiment, not a new test result. Reconstructed cases cannot recover post-stage timing/accounting that was never saved; that limitation is recorded in the recovery manifest.

Offline recovery entry point: `evaluation/recover_run.py --source-run <existing-run> --output <new-directory>`, using the project's Python environment and dataset paths. It creates a separate directory and does not instantiate inference models.

## Completion decision

The known locked-export failure is addressed and recovery is demonstrated. The semantic candidate has **not** met its release criteria. Do not claim that the tribunal now abstains only on correct cases, that the gate reliably distinguishes correct proposals, that advisor benefit is proven, or that an 80% accuracy / 10-point gain target has been reached.

A full dev run is not the next qualification step for this candidate. First establish that a compact review contract can finish at a supported, measured throughput, then test whether the judge can correctly use a supplied valid critique through the real gate. If that capability still fails, follow the bounded plan's decision point: one scoped critic/judge replacement or training study, or a narrower paper claim. Further sample-specific prompt edits are not an acceptable route to completion.
