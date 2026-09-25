# Full development run diagnosis

Run: `runs/dev50_reliability_20260914_111440`. This is an offline diagnosis of saved outputs, checkpoints, code and hardware telemetry. No model rerun or production change was made. Dataset labels are used for post-hoc scoring only.

## Result and scope

The exporter failed, so these are provisional latest-tribunal-checkpoint results, not a successfully finalized run. All 50 first hearings and 40 second hearings exist. The 34 durable final records are consistent with the earlier finding that progress stopped at 33: appending record 34 precedes the failing CSV replacement, while progress is updated afterward.

Initial 35/50 correct; final tribunal decisions 35/50 correct. Corrections: 1028 and 4536. Harms: 1841 and 2850. Of 15 initial errors, only two were corrected (13.3%). Of four accepted changes, two were correct (50%).

The original ten cases retain the same initial predictions and the same one-correction pattern. Their 80% to 90% result was not representative of the remaining forty, which went from 27/40 to 26/40. This does not establish that batch size itself causes semantic deterioration; a broader selection exposed failures absent from the small check. Runtime history does affect adaptive budgets, so controlled matched replays would be needed to isolate batch-history effects.

Final reviews: 18 binary, 31 semantic abstentions, one failed. Nine final change proposals comprised five correct and four harmful proposals. The gate accepted two correct and two harmful proposals, rejecting three correct and two harmful ones. Across all hearings and guided histories there were eight corrective and five harmful proposals; these counts include repeated attempts on the same cases and are not independent samples.

## Why the tribunal has no net benefit

There are both discovery and verification problems. Eight initial errors remained semantic abstentions, two received wrong same-label confirmations, three correct change proposals were rejected, and two were corrected. Lowering a global acceptance threshold would not discriminate these cases from the harmful proposals.

### Harmful acceptance: weather (1841)

Visual inspection shows the full multi-day forecast, including several rain and thunderstorm icons. The caption evaluates the Pittsburgh forecast as terrible. The proposer focused on the current condition, 'Mostly Cloudy', and treated the absence of explicit negative sentiment as contradiction. The relation verifier repeated that subjective negativity was not present in the image text and returned CONFLICT. The argument checker reported no errors. Thus a verified quotation was mistaken for verification of an invalid semantic inference, and the broader forecast evidence was not used decisively.

### Harmful acceptance: Flash's bike (2850)

The image says, 'If you ever feel useless just think about Flash's bike'. The caption calls the bike non-useful. The proposer and relation verifier interpreted the bike's ability to comfort the reader as evidence that the bike itself is useful. That confuses the joke's effect on the reader with the depicted object's usefulness. The argument checker repeated the same reversal and reported no errors. OCR was available; this was not a missing-text problem.

The verifier is separately prompted but uses the same model family/runtime; the proof explicitly records `model_error_independence_established: false`. Separate calls therefore do not guarantee independent semantic errors. The normative guard in `engine/tribunal.py` exempts a corroborated bridge from rejection, so an erroneous corroboration can pass that safeguard too.

### Correct proposals rejected

- **4111, damaged packaging:** proposer returned CONFLICT, but the relation verifier returned SUPPORT by treating damage as validating the caption's sarcastic 'care'. The argument check noticed problems, so the result was rejected, without resolving to the correct direction.
- **3351, liked/disliked product duration:** proposer and relation verifier returned CONFLICT. The argument checker then misread intention ('trying to ration so it lasts') as an actual duration and disputed the argument. This is another role/outcome distinction failure. Some error explanations are unfinished despite the output being marked format-valid; schema validity is not semantic completeness.
- **2199, police radar:** the last visual verification rejected one cited record as an invalid observation format. Its earlier hearing reached relation/argument checking. The proposal matched the dataset label, but its rationale also jumped from a broken radar to assumptions about driver admissions. This should be repaired with sound reasoning, not automatically admitted because it matches gold.

## Feedback

22 guided attempts cost 933.89 seconds (15m34s). Eighteen remained UNRESOLVED. Four produced CONFLICT: 1406, 2173, 2240 and 4430. Only 4430 was selected, and it confirmed an already-correct initial answer. None contributed an accepted correction.

Guidance mainly repeats methodological instructions; it is not an empirical memory of validated previous cases. The same model still supplies the interpretation. Feedback therefore often repeats the original failure, or proposes a direction without resolving its evidence obligations. Generic audit root-failure lists on an unresolved proposal mean verification was not executed; they must not be presented as many independent runtime failures. Feedback and verification costs are subsets of tribunal time, not additional wall time.

## Execution failures

- **657:** initial proposal used all 1,024 tokens in 89.26 seconds. The retry generated 897 tokens and hit its remaining 68.70-second deadline. Throughput was approximately 13.6 tokens/second. This is a token-exhaustion/full-retry budget failure, not demonstrated GPU slowdown during that call. Invalid partial output was discarded.
- **Initial visual answers:** 5 of 518 were invalid across three cases: 3560 (truncated OCR), 379 (decision-leak flag on relations), and 1312 (truncated full/right OCR and overlong left-region OCR). These labels identify validation failures; the flagged answer on 379 still needs a content audit before concluding it truly leaked a label.
- No failed independent-verification executions or invalid verification outputs were recorded by the existing quality checker. This does not certify every generated explanation as complete or correct.
- Two follow-ups were skipped for insufficient remaining budget.

## Export failure

`record_result` appends and flushes the authoritative JSONL record, then rewrites `predictions.csv`, then updates progress. `write_predictions` uses `os.replace` with no retry or recoverable handling. Windows returned WinError 5 replacing the CSV. A file handle held by Excel, synchronization software or another process is plausible, but the historical owner is not identifiable from the saved log. The failure does not prove OneDrive caused it.

This design allows a derived CSV export failure to abort finalization despite completed inference. The appropriate general repair is durable per-case results, bounded retries for transient replacement failures, explicit export-pending status, and an offline export/recovery path. Do not rerun model inference merely to recreate a spreadsheet. Preserve the original failed directory and recovery provenance.

## Runtime

Hardware timeline: about 15,400 seconds (4h16m40s). Approximately 10,461 seconds precede the first tribunal phase. Tribunal phases total about 4,833 seconds, with approximately 106 seconds of follow-up witnesses. Phase sampling is approximate.

Stored stage-time sums provide a more useful breakdown:

| Stage | Recorded seconds | Approximate minutes |
|---|---:|---:|
| Initial visual agent | 3,991 | 66.5 |
| Caption/language agent | 2,580 | 43.0 |
| Initial arbiter | 898 | 15.0 |
| Independent candidate generation | 2,059 | 34.3 |
| Debate | 569 | 9.5 |
| Tribunal/mediator | 4,827 | 80.5 |

These are component timings, not an exact reconciliation of wall time; loading, serialization and other overhead remain. Some diagnostic summary fields are zero because the synthetic checkpoint records omit final-report timing fields, not because tribunal execution was free.

The same first ten cases took 1,110.6 seconds in visual processing versus 310.8 in the ten-case run, and 1,321.7 seconds in language processing versus 283.0. For case 1028, identical input/output token counts show one language call increasing from 3.62 to 19.42 seconds and one visual call from 5.49 to 24.99 seconds. This establishes a real throughput regression, not merely more cases or longer answers.

Turbo was recorded throughout. The upstream phase included 22 sampled busy-GPU readings below 1,000 MHz; examples include 525 MHz at 98% utilization and about 14 W. No active software power/thermal or hardware power-brake flags were sampled. System available memory reached 16 MB, and was below 500 MB for approximately 51 sampled seconds across the run; process RSS peaked near 10.62 GB. Memory pressure and low-clock episodes are evidenced contributors to investigate, but do not by themselves explain all four hours. The current telemetry lacks per-call upstream phase labels, disk/page-fault measurements and historical competing-process memory usage, so the triggering cause remains unproven. Busy tribunal samples did not show the same low-clock state.

## Prioritized general changes to validate next

1. Separate interpretation, participants, comparison direction, intended action and observed outcome before proposing or verifying a relation. Require the challenge stage to evaluate the strongest competing reading, including a reading that preserves the initial answer, without revealing gold. Test evaluation, irony, analogy and literal controls together.
2. Validate coverage of decisive panels/time ranges rather than accepting a narrow true quotation as sufficient proof of a broad claim. Distinguish missing evidence from positive contradiction in the actual structured proof, not only prompt wording.
3. Repair a failed proof component while preserving valid observations; do not repeatedly regenerate an entire unchanged hearing. Evaluate correct-proposal recovery and harmful-proposal rejection separately.
4. Treat feedback as experimental until it adds measured value. Test an ablation and validate any future precedent memory on disjoint data; never add these gold explanations to production memory.
5. Make exports recoverable; shorten or decompose token-exhausted proposal generation and enforce a shared attempt budget. Add targeted file-lock and token-exhaustion regression checks before another expensive run.
6. Add upstream stage labels and host-memory/page-fault diagnostics, then compare a few fixed calls under matched hardware conditions. Explore bounded in-memory batches only with explicit control for changed budget history and extra model-load costs.

Evidence tables: `runs/dev50_detailed_diagnosis.json`, generated by the offline script `runs/diagnose_dev50.py`. No claimed improvement follows from this diagnosis alone.
