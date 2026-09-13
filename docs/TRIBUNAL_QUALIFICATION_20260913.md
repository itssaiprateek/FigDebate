# Tribunal qualification, September 13

Status: implementation candidate; live qualification in progress. The September 10 baseline remains unchanged in the original checkout.

The candidate adds a compact per-node observation/mapping proposal, V4 visual/binding/independent-decision/challenge verification, defect-specific hearings, context compaction, cumulative judge budgets, bounded visual completion repair, portable asset roots, and separate judge/gate metrics. Exact caption, image and evidence provenance bind the verification. No case IDs, benchmark answers, or dataset-specific keywords enter inference rules.

The V4 verification uses four calls in the complete path, stopping earlier on failed prerequisites. The final call finds a fresh alternative and checks the argument for unsupported details. These calls share a model: statistical error independence is not claimed. The old order-swap protocol remains readable for baseline tests and archived results; it is not the default runtime profile.

An explanation-only repair is permitted once when image observations and role bindings pass and the independent decision agrees, but the argument has a disputed unsupported step. It cannot change the verdict, observations or citations. Its changed explanation is challenged again. Passed checks are reusable only when their image, source, evidence provenance, prompt, schema, model/runtime and output budget are unchanged. Failed semantic checks are not repeatedly sampled until agreement. Cumulative case time survives judge model unload/reload within one run; process restart is a separate execution budget.

## Observed validation so far

- Complete regression suite: 470 tests, passing with one explicitly retained expected failure in assessment-conditioned initial scoring. No skipped tests with the installed asset roots. Log: `runs/qualification_20260913/unit_tests_v6.log`.
- Both historical context-overflow request shapes fit the original 3,072-token budget after compaction: 2240 uses 3,035 tokens; 4392 uses 3,050. Exact source captions and graphs remain unchanged; all evidence/context stays discoverable. These checks use the installed tokenizer without GPU inference. Results: `runs/qualification_20260913/saved_context_results.json`.
- Four-case judge-only replay `replay4_v4_v5`: initial 1/4 correct, final 3/4, two helpful corrections, zero harmful changes, three correct change proposals and one abstention. No execution or verification-format failures. Wall time 390.8 seconds, including judge load; this is not full-pipeline runtime. Cases 4555 and 4315 are corrected, 3332's correct initial answer is preserved, and 3351 remains a correctly proposed but rejected correction because the critic misreads the visible comparison.
- Two earlier broad-retry prototypes were intentionally stopped after observed regressions; their completed records and termination manifests are retained. In `replay4_v4_v4`, argument rereview reversed two initially correct proposals and used 237.5 and 240.2 seconds. The narrowed repair uses 116.4 and 131.5 seconds for the same cases. These are small diagnostic comparisons, not throughput or generalization estimates.
- The four additional cases (`replay_remaining4_v5`) add two correct proposals and two harmful proposals; all four are rejected. Across the eight distinct replay cases, five initially wrong answers are all proposed correctly, two are accepted, no harmful change is accepted, and final correctness rises from 3/8 to 5/8. Proposal precision is 5/7, versus the baseline subset's 4/5; increased error discovery does not establish improved proposal precision. One verification output truncated while listing four observations, despite valid top-level review JSON. The visual-verification allowance now scales as 192 + 160 tokens per requested observation (maximum four), keeping the original call and cumulative time limits. A targeted rerun is required to qualify this fix.
- Full-pipeline qualification remains pending at this checkpoint. Do not combine repeated cases across prototypes into an inflated sample count. Earlier replay token-accounting records are incomplete because the replay harness did not set the per-case accounting scope; generation diagnostics and wall timings remain available. This harness defect is fixed for subsequent runs. Full-pipeline accounting already sets its scope.

## Validation ladder

1. Contract/provenance, completion, context and budget regressions; then real tokenizer integration.
2. Eight live judge-only replays from the saved dev50 states. These isolate judging and cannot validate changes to upstream visual extraction. Prior tribunal-generated evidence and descendants are removed. The replay is explicitly diagnostic and reuses the latest saved upstream hearing state, which may reflect the old tribunal's questions.
3. Twelve full-pipeline diagnostic cases covering helpful/harmful changes, missed corrections, correctly rejected harms, context failures and timeouts. Cases and criteria are locked before inference.
4. The same 50 cases, seed, pinned models and hardware profile as the baseline. Compare per-case initial/final judgments, operational/verification failures, proposal precision, missed initial errors, accepted helpful/harmful changes and wall time.
5. Freeze a candidate before 100 additional examples outside the dev50 image groups. This is a stability check. The official final evaluation set stays untouched until the paper protocol and ablations are frozen.

The objective is to find and correctly revise a majority of the initially wrong answers while minimizing harmful changes. On the old 50-case fixture that would require at least nine helpful corrections out of 16 initial errors; it is an aspiration to test, not a guaranteed outcome or a quota for changing predictions. No sample-specific exceptions or repeated tuning on the final evaluation set are permitted. If a batch does not qualify, report it and repair the identified mechanism before expansion.

Runtime target: at least 30% reduction from the old 273-minute dev50 wall time, without lowering correction quality. A 240-second cumulative judge budget is initially configured per case, excluding time in other model stages. Budget exhaustion remains an explicit failure, never a fabricated semantic abstention or a completed proof. This budget is subject to measured qualification rather than being evidence of performance by itself.

## Reproduction and assets

`FIGDEBATE_MODEL_ROOT` can point to an external prepared models directory containing `vision/` and `judge/`. Pinned model IDs/revisions still apply. `FIGDEBATE_DATA_ROOT` can point to the dataset package containing `data/processed/` and `data/provenance/`. Missing files and invalid provenance still fail at actual load time; importing pure validation code no longer requires the entire dataset.

The live qualification uses the already validated Python 3.11.9 environment and installed model assets. This does not mean a fresh candidate virtual environment has been installed. The repository's environment setup remains necessary for a clean deployment.

`tribunal_quality.json` distinguishes proposal precision from accepted-change precision and records wrong confirmations, rejected correct proposals, abstentions, failed reviews and verification short-circuits. Its dedicated tribunal time is a subset of wall time; legacy mediator/judge columns must not be added to it.

## Known limitations to keep visible

- Synthetic tests verify mechanics, not real model semantics.
- Source-anchored role/condition records remain fallible model outputs, not formal semantic proofs.
- The initial arbiter still conditions its scoring on an earlier assessment. Its existing expected-failure diagnostic is retained and this behavior is not described as independent scoring. It is outside the first tribunal qualification batch because a previous isolated removal regressed initial decisions; it needs a separate matched experiment.
- A wider-context or larger-model migration is not bundled into this candidate. Qualify model changes separately if protocol improvements are insufficient.
- No zero-error or paper-readiness claim is justified before live and independent evaluation results.
