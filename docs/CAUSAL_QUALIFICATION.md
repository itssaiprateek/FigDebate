# Causal qualification â€” preregistration, 2 October 2026

Goal: improve labels AND sound explanations without additional harmful revisions or acceptance based on wrong evidence. Feedback disabled. Current initial arbiter and default focused tribunal remain frozen during diagnosis.

Diagnostic cohort: 1595 (idiom), 3351 (reversed outcomes), 4255 (reaction/comparison), 3255 (wordplay), 4315 (correct-answer preservation), 4547 (ambiguous originality guard). Five cases have primary semantic targets; jersey originality is an uncertainty guard, not a forced binary success target. Original dataset labels remain in the report and never enter prompts. A flawed reference sentence about inflation was noticed during selection; that case is excluded rather than silently treating its explanation as factual authority.

First arm: 6 cases x 4 isolated relation calls = 24 calls. Factors: original pre-review observation catalogue versus concise pixel-checked observations; existing aligned relation schema versus minimal reason/citations/relation schema. All arms share identical task instructions, source caption, full image, model, seed, 384 output-token budget and runtime profile. No proposer label, initial label, dataset label or reference is visible. No historical mapping is supplied, so all arms independently interpret the same sources. This isolates relation reasoning, not whole-pipeline performance. The checked package changes evidence organization and length as well as quality; improvement cannot alone be attributed to OCR correction. Decisive facts are already present in the original bottle and bubble-wrap catalogues.

Preregistered order: rotate four arms across cases; seed each arm identically. Record every output, validity, latency and prompt hash. Existing schema retains its field order; any schema effect may include generation order and field burden. If such an effect appears, perform a bounded matched-schema order-only check before claiming a cause.

Controls: repeat identical calls on two representative cases; evaluate a paired original versus opposite factual claim and a source-preserving versus source-paraphrasing interface only if needed to locate residual failures. Checked judgments and observations are oracle diagnostics, never deployable accuracy gains. Previously completed one-condition substitution and real bounded repair are retained as evidence, not repeatedly rerun.

After diagnosis: make one general mechanism change only. Validate through the real proposal, independent verification, audit and acceptance gate on unassisted original inputs. Compare against the unchanged focused control on selected cases, and reserve 1943, 2075, 3332, 4201 as an additional development check not used for choosing the change. These are development traces, not held-out test data. No production rule may depend on case identities or expected answers.

Promotion requires more acceptable corrections, no new harmful changes, no incorrect-evidence acceptance, preserved figurative reasoning and completion, within unchanged budgets. If evidence fails this standard, retain the implementation opt-in and report that no qualified accuracy gain was established. Never select best-of repeated generations.

## Controlled change selected before pipeline qualification

The field-order control on 3351 changed minimal label-first SUPPORT into minimal reason-first CONFLICT with the same information and field set. Existing-schema reason-first also produced the correct comparison but retained an uncertainty-field inconsistency. Therefore test `reason-first-audit-1`: move the existing decisive explanation to the first generated field in proposer and independent relation stages. Preserve every field, citation/source/image binding, uncertainty check, condition check, independent audit, and gate. Remove only the conflicting instruction to generate the reading first. All budgets remain unchanged. No minimal-contract or weaker-gate variant is promoted.

Pilot first-review qualification: current focused versus reason-first on 1595, 3351, 4255 and 4547. If there is an acceptable correction signal without additional harmful changes, run the predeclared reserve cohort. Track losses of baseline-correct answers separately from harmful changes relative to the initial answer. A correct label with a faulty justification does not count as a gain.

## Completed results

All 34 diagnostic stage evaluations completed (35 generation attempts): 24 factorial arms, six order controls, four exact-input repeats. One initial token-limited response recovered with the existing bounded retry (the retry had a 768-token allowance; 384 is the initial-call allowance). Four records were internally inconsistent rather than transport failures: an UNCERTAIN reading paired with a directional conclusion. Identical repeats had identical prompt hashes and byte-identical raw outputs. The order-only schema pairs have identical properties, required-field sets and all other constraints.

Five clear cases, plus the jersey uncertainty guard:

| Evidence / contract | Correct labels /5 | Valid correct labels /5 | Valid label plus clearly adequate explanation /5 | Jersey guard |
| --- | --- | --- | --- | --- |
| Original / existing | 0 | 0 | 0 | UNRESOLVED |
| Original / minimal | 4 | 4 | 2 | Unsupported SUPPORT |
| Checked / existing | 2 | 0 | 0 | UNRESOLVED |
| Checked / minimal | 4 | 4 | 3 | CONFLICT from missing evidence |

The label-only increase is not a deployable accuracy improvement. The minimal contract loses the ambiguity safeguard and some correct labels have partial explanations. Explanation ratings are conservative assistant reviews against image evidence and dataset reasoning, not independent human annotations or text-similarity scores. Reference explanations are fallible too: the jersey reference asserts unoriginality without supplying comparison evidence.

### Controlled real-review pilot

Eight real first-review evaluations completed on the four fixed cases. Both arms use the same source images, frozen observations, initial decisions, pinned model, seed, budgets and gate. They exercise proposal, visual selection, source mapping, independent relation, eligible audit, and actual acceptance. An inconsistent relation stops before audit as in production. They do not rerun the upstream initial agents, debate or a second repair hearing.

| Outcome /4 | Current focused | Reason-first candidate |
| --- | --- | --- |
| Correct proposal labels | 1 | 2 |
| Correct final labels | 1 | 1 |
| Clear-case final label + adequate explanation /3 | 1 | 1 |
| Helpful accepted changes | 1 | 1 |
| Harmful accepted changes against dataset labels | 1 | 1 |
| Wrong clear-case same-label verifications | 1 | 1 |
| Generation requests | 20 | 21 |
| Token-limited attempts, recovered | 1 | 1 |
| Failed generation requests | 0 | 0 |
| Mean review seconds | 101.36 | 105.89 |

No added correct final answer or acceptable explanation. The jersey revision is counted against the recorded dataset label for comparability, and separately fails the evidence-based uncertainty guard in both arms. Realized compute differs despite matched budgets; runtime is descriptive. GPU jobs ran sequentially, and the full CPU regression suite ran after the GPU experiment.

- 1595: both arms correctly connect distress to disappointment/sadness. The full tribunal succeeds where the isolated stage fails, demonstrating context dependence rather than an established hard model capacity limit. Reason-first relation metadata still wrongly calls the reading LITERAL; the final explanation is adequate.
- 3351: correct OCR and entity/outcome attachments survive visual selection. Both proposers list the actual reversed durations and still call them SUPPORT. Independent relation and audit repeat the mistake. This local failure is a semantic comparison/intent-versus-outcome error, not missing OCR or a purely Python-gate failure.
- 4255: reason-first changes the proposal to the correct direction, but the interpreted assertion is replaced by the image's negative evaluation. Independent relation marks the same condition CONFLICT and unresolved. The gate correctly rejects the inconsistency. The wrong initial answer and its explanation omitting the appeal comparison remain final.
- 4547: both arms accept originality from colors, pattern and branding without comparison evidence. Default and candidate fail this guard; no relaxation of the gate is justified.

**Decision:** do not promote `reason-first-audit-1`. It remains opt-in for reproducible diagnosis; grounded initial reasoning and focused review5 defaults remain unchanged. Do not promote the minimal contract. No sample-keyed production rules, budget increases, removed validators or feedback were introduced. The additional four-case reserve cohort was not run because the predeclared pilot promotion criterion failed. Optional opposite-claim/source-paraphrase diagnostics were not needed to select this bounded change and are not claimed as completed. A model/precision or training comparison would be needed before claiming an intrinsic capacity limitation.

Automated regression validation: 750 tests, 749 passed, one pre-existing expected failure (`test_scoring_cannot_condition_on_assessment`); no unexpected failures. `git diff --check` passed. Software tests are not semantic-performance evidence.

Artifacts:
- `outputs/reason_first_pair_20261002/qualification_report.html`
- `outputs/reason_first_pair_20261002/qualification_summary.json`
- `outputs/reason_first_pair_20261002/causal_diagnostics_summary.json`
- `outputs/reason_first_pair_20261002/reasoning_review.json`
- Raw records/manifests/source snapshots in `outputs/causal_factorial_20261002`, `outputs/causal_order_20261002`, `outputs/causal_repeat_20261002` and `outputs/reason_first_pair_20261002`.
- Before-change snapshot: `.cache/causal_iteration_before/source_snapshot.zip`.

Reproduce the real-review pilot with a NEW output directory:

```powershell
$env:PYTHONHASHSEED = '42'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:TOKENIZERS_PARALLELISM = 'false'
.\.venv\Scripts\python.exe -m evaluation.qualify_grounded_gate --source-runs outputs/dev50_20261002_123143 --sample-ids vflute_train_1595 vflute_train_3351 vflute_train_4255 vflute_train_4547 --variants focused reason_first --output outputs/reason_first_pair_repeat
```

The replay needs the retained source run and local upstream dataset cache; these evaluation artifacts are not assumed to be present in a fresh clone. The normal pipeline option is `--tribunal-audit-mode reason-first-audit-1`, explicitly experimental, with feedback disabled. This report does not recommend enabling it for production performance.
