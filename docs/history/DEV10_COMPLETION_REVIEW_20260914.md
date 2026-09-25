# Ten-case completion check — 14 September 2026

Run: `runs/dev10_completion_check_20260914_180200`. Offline review only; no code changes or inference rerun.

The run used `paper-8gb`, structured interpretation disabled, and legacy `precedent` feedback enabled. The experimental advisor made zero calls. The later tribunal-first change list was not implemented in this run.

## Results

| Measure | Result |
|---|---:|
| Initial correct | 8/10 |
| Final correct | 9/10 |
| Helpful accepted changes | 1 |
| Harmful accepted changes | 0 |
| Final correct change proposals | 1 |
| Final harmful change proposals | 1, rejected |
| Semantic abstentions | 7 |
| Failed tribunal reviews | 0 |
| Invalid initial visual answers | 0/108 |
| Wall time | 3,110.92 seconds, approximately 51m51s |

All ten committed records agree with their completion-manifest hashes. Progress is complete and reports were exported. Recorded judge verification executions and output contracts show no failures. Recursive inspection of saved trace diagnostics found no TIMEOUT, TOKEN_LIMIT or REPETITION termination events. This is evidence of successful execution on this run, not a guarantee against future failures or semantic mistakes.

## Decisions and remaining weaknesses

- **1028:** the tribunal corrected the initial ENTAILS decision to CONTRADICTS. Its deciding observation compared the displayed 1.88 Mbps with the claimed 10 Mbps package.
- **2240:** the tribunal proposed a harmful CONTRADICTS change, treating inclusion of an album with tickets as contradicting dread of receiving it. The gate rejected it during visual grounding, citing mismatched/truncated OCR records. It did not reach and demonstrate correction of the underlying semantic error. Thus this is a blocked harm, not proof that semantic acceptance is fixed.
- **4547:** the remaining wrong answer. The tribunal abstained after two hearings, stating that a third-kit release was not established by the image. This is still a correction opportunity to diagnose, not grounds to force a label change.
- Six other abstentions preserved already-correct answers. All seven final abstentions followed two hearings.
- Legacy feedback attempted three guided reviews, all unresolved and unselected, costing 94.66 seconds. No correction is attributable to this feedback.

## Matched comparison

The earlier `dev10_reliability_20260914_102054` used the exact same ten cases. Every initial and final label matches this run. Both achieved 8/10 to 9/10 with one accepted correction and no accepted harm.

Abstentions fell from eight to seven, but this was accompanied by a harmful final proposal on 2240, not another correct final answer. Final proposal precision is one correct out of two proposed changes; accepted-change precision is one out of one. These tiny denominators do not establish general reliability.

Wall time increased from 1,977.49 to 3,110.92 seconds, about 57%. Recorded initial visual-agent time rose from 310.84 to 1,078.14 seconds, explaining approximately 767 of the 1,133 added seconds. Tribunal time was approximately unchanged: 822.73 versus 816.54 seconds. Language, initial-arbiter and debate times were also similar. These timings locate much of the increase upstream; they do not identify its hardware or software cause. Timing categories should not be added indiscriminately because some overlap.

## Decision

The enabled version completed this ten-case run without the failures seen in the experimental candidate. Its final accuracy has not improved over the earlier matching run. The tribunal-first plan remains appropriate: qualify evidence selection and rejection/repair behavior, then address avoidable abstentions. Do not interpret this 90% result on repeatedly inspected development cases as paper-ready performance or as evidence that the new plan has already worked.
