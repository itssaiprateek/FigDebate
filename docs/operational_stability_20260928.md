# Operational stability audit: stability20_20260928_203632

Scope: completion and output reliability only. No accuracy judgment or new inference run.

## Run integrity

- All 20 requested cases completed in the original order; image and caption hashes match the two source ten-case runs.
- All 20 record hashes match the completion manifest. No duplicated/missing records or missing final labels/reason strings.
- Inference wall time: 4,106.1362 seconds (68 minutes 26 seconds), including model loads.
- Completion reasoning mode, baseline tribunal audit, bounded repair, batch size 10, feedback disabled. No upstream stage reuse.
- All 525 recorded generation requests returned successfully. No recorded generation exceptions, timeout, OOM recovery or input-truncation flags were found. This does not make every returned response valid.

## Explanation completion

19/20 initial assessments and final explanations pass their recorded completion checks. Four initial assessments needed the second attempt: 4459, 1177, 4204 and 1049. The first three recovered. Case 1049 did not.

For vflute_train_1049, both drafts ended normally before the 256-token ceiling (157 and 132 tokens). Both included [VF001] in Visual Evidence but omitted the separate Evidence IDs field. The field parser therefore rejected both drafts. The final record contains an explicit unresolved fallback instead of an explanation. This is a field-contract failure, not token exhaustion. The generic fallback wording about the bounded generation budget does not describe the precise underlying cause; the attempt diagnostics do.

## Other remaining problems

1. The same 1049 case asked the visual witness, "Is the phone screen fully visible or partially obscured?" The router classified this choice question as yes/no, required YES/NO/UNCLEAR, and rejected the model's "Fully visible" and "FULLY VISIBLE" answers. This left one invalid final visual-witness response (19/20 were valid).
2. Case 1397 ended with an inconsistent audit record after its bounded clarification: it supplied an alternative and CONFLICT while declaring alternative_status NONE. This is a complete-but-inconsistent record, not a timeout or missing JSON. The gate withheld verification. Two other final relation judgments were tagged inconsistent; those are semantic-consistency outcomes, not execution failures.
3. Twenty generation events reported hitting their output ceiling: eight visual-grounding, nine visual-witness and three tribunal calls. A token-limit hit is not automatically an unusable answer, but this leaves a residual truncation risk upstream even though final prose is mostly complete.

Five cases reported deliberate image resolution reduction under the hardware operating limits (693, 1800, 1264, 2690, 1049). No OOM recovery was recorded. Do not conflate that planned policy with a memory crash.

## Assessment

Run-level completion passed. Strict output reliability did not pass fully: one final explanation fell back, one visual answer was rejected by a mismatched question contract, and an audit ended inconsistent. The completion change is substantially better than the earlier 14/20 visibly cut-off final explanations, but this run does not establish zero missing fields or repeat-run determinism.

Priority fixes to evaluate next: preserve and validate inline evidence IDs when the dedicated heading is absent; distinguish choice questions from yes/no questions; keep audit fields consistent without weakening the acceptance gate. Review the recorded token-limit cases individually before changing their budgets. No production code was changed during this audit.
