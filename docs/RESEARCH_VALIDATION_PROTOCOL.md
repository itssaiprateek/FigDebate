# Validation protocol — not a results claim

The target remains explainable image–caption entailment on the **humor,
metaphor and sarcasm subset of V-FLUTE**. Existing samples, labels and official-style
split files are retained. The heavily inspected dev50 is a diagnostic fixture,
not a fresh generalization benchmark.

## Fixed interpretation and contamination controls

- Evaluate the original expressed caption. Represent pragmatic/figurative
  interpretations separately; neither sarcasm metadata nor missing support is
  a shortcut to CONTRADICTS.
- Gold and reference explanations are evaluation-only. They must not enter
  witness, candidate, judge, verification or repair prompts.
- Audit encoded-image and decoded-pixel hashes, source metadata and candidate
  near duplicates. Near-duplicate candidates require inspection, not automatic
  removal. Report unchanged established-split results and a separately named
  leakage-controlled supplement.
- Keep related images together. Exclude previously inspected dev50 image groups
  from a fresh calibration sample. Freeze the sample manifest before comparing
  variants. Upstream reconstruction is pending the user's Hugging Face access;
  no historical revision equivalence is currently claimed.

## Qualification order

1. Regression tests for execution taxonomy, role/polarity/scope preservation,
   label blindness, unknown/unseen citations, lifecycle ancestry and shadow isolation.
2. Live development-only model/contract tests, including ordinary, OCR-heavy,
   comparative and symbolic cases. Report failures and truncation, not just successes.
3. Paired component ablations on development. Keep baseline extraction outputs
   fixed wherever applicable. A candidate argument is not additional verified evidence.
4. Fit any calibration on disjoint image groups; serialize the calibration
   data hash, source/config identity and parameters. Never fit on final test.
5. Freeze the complete method and analysis choices; run held-out comparisons.
6. Two blinded human raters independently assess claim preservation, observed
   fact accuracy, entity/scope binding, relation correctness, citations and
   alternative handling. Resolve disagreements after independent scoring and
   report inter-rater agreement. Do not fabricate these ratings.

## Required comparisons

Use the improved base, not the historical broken base: base only; hearing without
judge; judge without hearing; independent voting/self-consistency at a declared
budget; conventional bounded debate; full tribunal; and component removals.
Report actual calls, input/output tokens, latency and peak memory. If budgets
differ, label the comparison unmatched; do not call it compute-matched.

The planning target is a five-percentage-point absolute accuracy improvement
with a paired image-group confidence interval above zero, plus benefit over
the strongest qualified simpler control. This is a **prespecified research
target, not a promised outcome or a rule for relaxing evidence gates**.
Freeze the comparison family and apply a declared multiple-comparison correction.
Use at least three fixed inference seeds for stochastic variants, with one fixed
sample manifest; repeated seeds are not independent images.

## Reporting and release gate

Primary accuracy includes every requested case, including delivered fallbacks
and unsuccessful outputs. Also report macro-F1, class/phenomenon recall,
corrections and harms, correction precision, useful rejected proposals,
execution success, schema validity, semantic abstention among eligible reviews,
coverage/selective risk, calibration and human explanation ratings.

For a complete run, every requested case must have a record. The evaluator
counts every prediction-file row, scoring nonbinary/missing answers as incorrect
and retaining binary answers even when their evidence contract is invalid.
Evidence-contract coverage and subset accuracy are separate diagnostics. A
partially completed file is explicitly labelled incomplete; its score is not
the requested-population accuracy. The confusion matrix includes a no-answer
column so failed outputs cannot silently disappear.

No eligible judge reviews means semantic abstention is undefined, not 100%.
Report baseline-to-final and pre-judge-to-final gains separately.
Use paired image-cluster bootstrap intervals and cluster sign-flip tests;
row-level McNemar is supplementary where caption siblings are dependent.

Do not claim publication readiness or a substantial gain until these tests
are completed. Retain unsuccessful variants and limitations. The contribution
under investigation is the combination of expressed-claim preservation,
targeted cross-modal disputes, independent bridge checks, lifecycle-aware
revision and faithful accepted-decision records—not invention of debate itself.

The ten-paper evidence archive and bibliography remain in the task workspace
under research/tribunal_audit_20260905. Saved sources document attribution,
not blanket permission to reproduce text, figures, datasets or code.
