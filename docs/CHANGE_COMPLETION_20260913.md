# Completed engineering changes — 13 September 2026

The engineering repair batch is complete. Code checks and focused failure checks are finished. The user will run the full development-set evaluation; no further batch evaluation is scheduled or running.

Implementation commit: `21324b42bd693076ec886085071146c5c5f39727`, branch `tribunal-stability-20260913` in `figdebate_release_candidate`. The original `figdebate final` checkout remains clean at September 10 commit `b4fc4b02f37f15f5cda8a7e86866d21d7dafa726`. No changes were pushed to GitHub.

This closes the engineering change list. It does **not** establish that the judge now corrects most errors, that harmful changes have decreased on the full set, or that the project is paper-ready. Those are evaluation outcomes, not properties a code test can prove.

## Applied changes

| Area | Final behavior |
|---|---|
| Judge proposal | Compact source-anchored conditions, visible states, roles, scope and deciding citations replace repetitive broad judgments. Exact captions and claim graphs remain available. |
| Verification | Visual grounding, participant matching, a fresh direction decision and an opposing-interpretation audit are separate obligations. Opposite states do not automatically invalidate a participant match. All mandatory checks must pass before revision. |
| Evidence integrity | Only admissible literal evidence can support decisive observations. Each requested visual evidence ID gets exactly one verification slot; the model can mark it unsupported. Provenance and source/image binding are checked again during acceptance. |
| Repair | One focused explanation repair per review; at most two tribunal rounds. Passed checks are reused only with unchanged dependencies. Failed semantic checks are not repeatedly sampled until agreement. Invalid semantic questions are recorded before dispatch to a literal visual witness. |
| Context | Compact records and discoverable indexes preserve the exact mandatory source. Both saved context-overflow cases fit the original text budget in tokenizer checks. Irreducible overflow remains an explicit failure. |
| Runtime | Failed prerequisites stop dependent work. A cumulative 240-second judge budget limits each case, with remaining-budget checks before follow-ups. Conservative 8 GB profiles release unused prefill allocations once while retaining live weights and cache. This memory experiment did not prove a speed increase. |
| Structured output | Bounded structural whitespace prevents the observed formatting loop. Generated explanations can finish within a 480-character allowance, while source spans retain the 240-character limit. Incomplete-field repairs identify the actual failing field and its ending. |
| Visual completion | Ordinary prompts are restored to the baseline wording. One bounded retry gets an explicit word target for prose; OCR remains a separate copying task. Explicit EOS termination is distinguished from exhausting the token allowance. Incomplete answers are not accepted merely because they have punctuation. |
| Retry reporting | The final status and error describe the final attempt. Primary and retry validation outcomes and raw responses remain separately auditable. Optional failed initial answers are now counted even when the outer visual schema is valid. |
| Source quotation | A unique multiword source span can be bound after an uppercase initial letter was added by the model. All remaining characters must match exactly. Single-word names, internal case changes, changed words or negation, and ambiguous spans are not repaired this way. Original text and source offsets are recorded; roles, observations and judgments remain unchanged. The audit reconstructs the binding from the raw answer. |
| Measurement and assets | Helpful/harmful changes, proposal quality, wrong confirmations, abstentions, operational/verification failures and latency are reported separately. Model/data roots can point to existing prepared assets. Model weights and pinned revisions are unchanged. |

## Verification completed

- **493 code regression tests: 492 passed, one known expected failure, no skips.** Final log: `runs/qualification_20260913/unit_tests_release_handoff.log`.
- **Visual completion, case 1264:** a live initial relations request reproduced primary truncation, then completed successfully with the revised retry. Both attempt outcomes were retained. This checks completion, not the truth of every observation.
- **Incomplete observation, case 2240:** the focused live repair returned a complete observation and a valid proposal contract. It was not passed off as an accepted or semantically correct revision.
- **Quotation, case 3351:** replaying the saved real mapping response through the final source binder changed invalid to valid source quotation while preserving every nonquotation field. No new inference or prediction change was involved. The raw quote and exact source span are retained.
- Whitespace/diff checks passed. The original checkout was verified clean and still points to the September 10 baseline.

Focused artifacts: `focused_final_repairs.json`, `final_saved_quote_binding_handoff.json` and `unit_tests_release_handoff.log`, all under `runs/qualification_20260913/`.

The expected failure is `test_scoring_cannot_condition_on_assessment`: initial scoring still conditions on its earlier assessment. This was explicitly excluded from the first tribunal repair batch after an earlier isolated removal regressed initial accuracy. It remains a known research limitation, not a passing test.

## Experiments closed without adoption

The evidence-first judge view and broad argument rereviews were rejected after observed semantic regressions. A source-quotation enumeration experiment produced valid strings but mixed the two products' roles; it was also rejected. The final source binder preserves the original role assignments instead of regenerating them. Negative outputs and the reasoning for these decisions remain in the qualification record.

The first 12-case full-pipeline diagnostic completed before the user's instruction to reserve full testing. Its mixed results and regressions are documented in `TRIBUNAL_QUALIFICATION_20260913.md`. The second 12-case run was stopped at the user's request after eight first-round reviews; it is explicitly incomplete in `runs/full12_candidate_v9_20260913/termination.json`. Do not report its partial results as a completed 12-case score.

## Full evaluation left to the user

Run the **candidate folder** to evaluate these changes. The original `figdebate final` folder still contains the baseline. This command uses the existing validated environment and model assets, with the same 50-case split, seed and hardware profile. It does not use the 12-case selection file.

```powershell
Set-Location -LiteralPath 'C:\Users\Sai Prateek\OneDrive\Desktop\FigDebate_sent\FigDebate_sent\figdebate_release_candidate'
$candidatePython = 'C:\Users\Sai Prateek\Desktop\FigDebate_sent\FigDebate_main\.venv\Scripts\python.exe'
$env:FIGDEBATE_MODEL_ROOT = 'C:\Users\Sai Prateek\Desktop\FigDebate_sent\figdebate final\models'
$env:FIGDEBATE_DATA_ROOT = 'C:\Users\Sai Prateek\Desktop\FigDebate_sent\figdebate final\dataset'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONHASHSEED = '42'
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = (Get-Location).Path.Replace('\', '/')
$candidateRun = 'runs/dev50_final_candidate_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& $candidatePython -u run_figdebate.py `
  --dataset-split vflute_train_dev50 --num-samples all `
  --selection-seed 42 --seed 42 --execution-mode stagewise `
  --debate-mode enabled --evidence-mode enabled `
  --judge-mode tribunal --judge-scope all `
  --semantic-bridge-mode corroborated --candidate-mode independent `
  --control-mode none --feedback-mode disabled `
  --hardware-profile paper-8gb --run-purpose diagnostic `
  --run-dir $candidateRun
```

Compare all 50 initial and final decisions, helpful/harmful changes, proposal precision and error discovery, optional visual failures, failed reviews, valid abstentions, and wall time against the saved baseline. A gain achieved solely by abstaining more is insufficient. Review new harmful changes individually. No full-set accuracy, harm reduction or runtime gain is established by the focused checks above.

If that run qualifies, freeze the candidate before an additional image-disjoint stability set and the paper's held-out evaluation and ablations. The compact direct-judge control and the 17-question saved-failure replay are prepared but were not run following the user's instruction. A clean standalone environment and final research evaluation remain release work; the current validation reused the installed environment and assets.
