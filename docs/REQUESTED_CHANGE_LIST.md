Here is the complete proposed change list, in implementation order. **We will keep your three-category dataset: humor, metaphor and sarcasm.** No changes have been implemented yet.

## 1. Lock the dataset and evaluation protocol

1. Make the three-category filter explicit and reproducible in the preparation script.
2. Pin the upstream dataset version.
3. Preserve original sample identifiers, source-dataset information and image hashes.
4. Verify that rebuilding produces the same samples, labels and split counts.
5. Avoid unnecessary lossy image re-encoding, especially where readable text matters.
6. Detect duplicate and near-duplicate images.
7. Keep related image–caption pairs together when creating internal training/calibration splits.
8. Keep the existing development samples for debugging; use untouched held-out samples for final evaluation.
9. Keep gold labels and reference explanations outside every inference prompt.
10. Document the benchmark’s label interpretation, particularly sarcastic captions versus their non-sarcastic paraphrases.
11. Describe results as evaluating the **three-phenomenon V-FLUTE subset**.

Do not silently remove examples from established evaluation splits. Document overlaps and provide a separately identified leakage-controlled evaluation where necessary.

## 2. Fix judge execution and failure reporting

12. Separate these outcomes:

   - Successful binary judgment.
   - Genuine insufficient-evidence abstention.
   - Generation/runtime failure.
   - Malformed output.
   - Valid proposal rejected by the acceptance gate.

13. Stop counting failed generations as semantic abstentions.
14. Preserve failed-attempt diagnostics, memory information and elapsed time.
15. Record the actual reason each judge attempt failed.
16. Clearly identify when the final answer comes from the unchanged baseline because the judge failed.
17. Calculate each rate with the appropriate denominator; no successful reviews means semantic abstention rate is unavailable.
18. Add live judge qualification tests before launching an evaluation batch.

## 3. Repair the judge’s input

19. Remove the leaked initial prediction from nested dossier fields.
20. Remove previous-answer proxies, such as decision-derived recommendations, from independent judging.
21. Replace broad copying of internal records with an explicit list of permitted input fields.
22. Remove duplicated claim contracts, observations and hearing content.
23. Enforce a total token budget—not merely a limit on evidence-item count.
24. Reserve sufficient space for the judge’s structured answer.
25. Keep the complete history on disk while sending a compact, relevant case packet.
26. Allow omitted evidence to be retrieved before the judge relies on it.
27. Record exactly which evidence text the judge actually received.

## 4. Rebuild Agent 2’s claim contract

28. Store the original caption separately and immutably.
29. Separate **what the caption expresses** from **its possible figurative interpretation**.
30. Split the oversized contract into a small mandatory claim structure and optional interpretation fields.
31. Represent subject, predicate, object, negation, quantities, comparison direction and scope explicitly.
32. Attach caption spans to extracted entities and claims.
33. Allow genuinely inapplicable fields to be empty rather than forcing invented values.
34. Replace entity-word matching with checks that preserve semantic roles.
35. Replace broad opposite-word heuristics with relation-specific checks.
36. Prevent sarcastic reinterpretation from silently changing the proposition being evaluated.
37. Increase or adapt generation budgets based on measured output requirements.
38. Qualify schema-constrained generation where supported.
39. Repair only defective fields instead of repeatedly regenerating the entire contract.
40. Measure formatting validity and semantic correctness separately.
41. Qualify the extraction model on held-out cases before deciding whether to replace or fine-tune it.

**Required regression cases:** swapped roles, negation, numbers, comparisons, pronouns, panel/time scope and sarcastic/non-sarcastic caption pairs.

## 5. Improve visual evidence and targeted verification

42. Preserve the distinction between directly observed facts and tentative symbolic interpretations.
43. Attach region or panel references to decisive observations where possible.
44. Bind OCR text to the correct object, region or panel.
45. Record uncertainty instead of treating every generated observation as established fact.
46. Ask targeted follow-up questions about the exact missing relationship.
47. Reinspect relevant crops when the original image pass is insufficient.
48. Check the decisive observation against the image—not merely against Agent 1’s description.
49. Preserve original observations and explicitly mark corrections or supersession.

## 6. Replace circular semantic-bridge verification

50. Stop treating word overlap as proof that an observation supports a premise.
51. Stop treating the proposing judge’s confidence as independent corroboration.
52. Independently check:

   - Whether the observation is accurate.
   - Whether the entities and scope are correctly bound.
   - Whether the observation supports or contradicts the caption claim.
   - Whether the figurative interpretation is justified.

53. Hide the proposer’s preferred answer during independent verification.
54. Evaluate a different model family for verification where practical.
55. Replace the current artificial position-reversal check with genuinely repeated scoring under swapped presentation order.
56. Prevent derived evidence from certifying its own parent proposal.
57. Measure false corroborations and false rejections on annotated examples.

## 7. Repair evidence lifecycle and acceptance gates

58. Exclude superseded, rejected and otherwise inadmissible evidence from decision strength.
59. Check the validity of an evidence item’s ancestors, not just the item itself.
60. Reverify derived conclusions when their supporting observations change.
61. Keep shadow-mode proposals completely separate from operational evidence.
62. Prevent duplicate derivatives of one observation from being counted as independent corroboration.
63. Separate structural/provenance checks from semantic judgments.
64. Record the precise reason a correction was accepted or rejected.
65. Do not weaken thresholds merely to obtain more revisions.
66. Test whether useful corrections are being blocked alongside whether harmful corrections are being accepted.

## 8. Make the tribunal perform meaningful deliberation

67. Generate independent candidate interpretations before exposing them to peer answers.
68. Replace deterministic “advocate” summaries with actual candidate reasoning where needed.
69. Identify the strongest competing explanation and the decisive disagreement.
70. Do not force disagreement when both candidates reasonably agree.
71. Direct witness questions toward evidence that could resolve the disagreement.
72. Allow a bounded second round when a specific follow-up can supply useful information.
73. Stop when no new evidence appears, the supported resolution stabilizes, or the budget is exhausted.
74. Preserve unresolved disagreement rather than forcing consensus.
75. Have the judge compare the competing cases, not simply endorse the existing answer.

These architectural changes are **research-supported proposals**, not guaranteed accuracy improvements.

## 9. Repair confidence and uncertainty

76. Separate model preference, observation reliability, verifier confidence and final calibrated probability.
77. Stop presenting hard-coded reliability weights as measured probabilities.
78. Calibrate confidence and acceptance policies using disjoint development data.
79. Freeze thresholds before final testing.
80. Report calibration, class-specific recall and harmful revision rates.
81. Keep genuine uncertainty distinct from infrastructure failure.
82. Preserve the rule that missing support alone does not establish contradiction.

## 10. Make the explanation trail faithful

83. Generate the final explanation from the accepted decision record.
84. Include:

   - The original claim.
   - The decisive visual evidence.
   - Relevant entity and scope bindings.
   - The necessary figurative interpretation.
   - The strongest alternative.
   - Why that alternative was rejected.
   - Verification results and any fallback status.

85. Check that citations support the statements attached to them.
86. Prevent explanations from relying on superseded or unseen evidence.
87. Explain rejected revisions as well as accepted revisions.
88. Evaluate explanations with blinded human review.
89. Add evidence-removal and controlled counterfactual tests.
90. Keep readable justifications concise while preserving the detailed audit record.

## 11. Unify execution and reproducibility

91. Make the API, sequential runner and stagewise runner use the same reasoning logic.
92. Forward all supported settings correctly; reject unsupported combinations explicitly.
93. Fix the sequential debate-disabled configuration mismatch.
94. Record complete relevant source, dependency, prompt, schema and dataset identities.
95. Include uncommitted source changes in experiment identification.
96. Reject duplicate sample identifiers during comparisons.
97. Verify identical images, captions and intended configuration differences—not just matching IDs and labels.
98. Use stage-specific cache fingerprints so unchanged upstream outputs can be shared fairly.
99. Invalidate dependent checkpoints when relevant inputs or code change.

## 12. Prove whether the tribunal adds value

100. Compare the improved system against:

   - The same base pipeline without hearing or judge.
   - Hearing without judge.
   - Judge without hearing.
   - Matched-budget independent voting/self-consistency.
   - Conventional bounded debate.
   - The complete proposed tribunal.

101. Remove individual components to measure their contribution.
102. Report corrections **and harms**, not simply the number of changed predictions.
103. Separate the full tribunal gain from the judge’s incremental gain.
104. Report accuracy, macro-F1, per-category results, coverage, calibration and explanation quality.
105. Use paired, image-group-aware uncertainty estimates.
106. Run prespecified seeds for stochastic configurations.
107. Select an adequately sized held-out evaluation; stop treating ten-case gains as general performance evidence.
108. Record unsuccessful variants and avoid selecting configurations using final-test results.
109. Define the desired practical improvement before testing.

## 13. Keep hardware practical and the paper defensible

110. Use sequential model loading, compact contexts and targeted crops.
111. Qualify consumer-scale quantized models before considering larger replacements.
112. Measure peak memory, runtime, model calls and token consumption.
113. Add regression tests for every confirmed defect and replay tests using real model outputs.
114. Retain saved papers, citations and exact adaptation records.
115. Attribute existing debate methods; position novelty around FigDebate’s **figurative claim handling, targeted evidence disputes, independent verification and faithful revision traces**.
116. Make paper claims match the measured implementation—including limitations and negative results.

**First implementation milestone:** dataset/protocol reproducibility, correct failure reporting, bounded label-blind judge inputs, and evidence lifecycle fixes. Then repair Agent 2 and bridge verification before expanding the debate architecture.
