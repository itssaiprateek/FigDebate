# Using the official V-FLUTE splits

This is the current workflow for dataset selection and before/after experiments.
Run commands from `C:/Users/Sai Prateek/Desktop/FigDebate_sent/FigDebate_main`
using its existing Python environment. No re-merging or relabeling is required.

## What stays unchanged

The official split membership is preserved. Your humor/metaphor/sarcasm filter
leaves 3,992 training, 573 validation and 569 test cases. The 50 development
cases remain a saved subset of training, not a fourth independent partition.
The original full five-phenomenon release has 6,027 cases; we are deliberately
not restoring idioms or similes.

Use the published expert-verified labels and explanations as the default gold.
A complete new annotation of every case is not a prerequisite for development.
Keep targeted questions about ambiguous examples in a separate issue register.
Do not change gold because a model disagrees. Reference verification and human
judgment of the model's explanation are different checks.

## Choose any reproducible cohort

New cohorts default to a seeded random order. Explicit ID lists default to
their listed order. A saved manifest automatically supplies its original
selection seed and strategy, independently of the inference seed.

```powershell
# Ten random development cases; add your normal model/tribunal flags.
python run_reproducible.py --dataset-split vflute_train_dev50 --num-samples 10 --selection-seed 42 --judge-mode tribunal --run-dir outputs/dev_before

# After a code change: exactly the same ten cases, same order and inputs.
python run_reproducible.py --dataset-split vflute_train_dev50 --sample-manifest outputs/dev_before/sample_manifest.json --num-samples 10 --judge-mode tribunal --run-dir outputs/dev_after

# Randomize ONLY within those ten previously selected cases; choose five.
python run_reproducible.py --dataset-split vflute_train_dev50 --sample-ids-file outputs/dev_before/selected_ids.json --selection-strategy random --selection-seed 99 --num-samples 5 --judge-mode tribunal --run-dir outputs/dev_five

# A filtered official training cohort; no internal repartition is required.
python run_reproducible.py --dataset-split vflute_train --phenomena metaphor --sources irfl vismet --num-samples 20 --selection-seed 17 --judge-mode tribunal --run-dir outputs/train_metaphor_20

# Inspect chosen test cases diagnostically, without pretending they are untouched.
python run_reproducible.py --dataset-split vflute_test --phenomena sarcasm --num-samples 10 --selection-seed 17 --run-purpose diagnostic --judge-mode tribunal --run-dir outputs/test_sarcasm_diagnostic

# Full selected three-phenomenon validation split.
python run_reproducible.py --dataset-split vflute_val --num-samples all --run-purpose evaluation --judge-mode tribunal --run-dir outputs/validation_full
```

You can instead use `--sample-ids ID1 ID2 ...`. Every ID must belong to the
chosen split. `--num-samples all` means all rows surviving the supplied filters
and ID list, not all 6,027 original examples. Oversized counts, empty filters,
unknown IDs and duplicate IDs fail explicitly rather than silently changing
the requested experiment. Do not combine replay manifests with new filters.

`--selection-strategy stratified` remains available for balanced diagnostic
samples; it does not estimate natural prevalence. `prefix` keeps source/list
order. A different selection seed produces a new random ordering, but very
small pools can legitimately produce the same permutation by chance.

Add `--selection-only` to prepare and validate samples without model inference.
It saves the manifest, selected IDs and evaluation references. To run that plan,
replay its manifest into a NEW run directory and specify the intended count.
Use `--num-samples all` only if you want the entire eligible manifest pool.

Each run saves:

- `sample_manifest.json`: full ordered eligible pool, hashes and selection settings.
- `selected_ids.json`: only the IDs requested for this run.
- `evaluation_references.json`: selected gold explanations, bound to manifest hashes.
- `run_config.json`: declared purpose, split and actual requested count.

An already-used output directory cannot be overwritten accidentally. For an
interrupted inference run, `--resume` reuses its saved manifest if none is
specified. Keep count, purpose and all inference settings identical. Code
changes are a NEW run, not a resume.

## Evaluation and references

The tribunal answers from the image/caption and its evidence. Gold explanations
are held in the evaluation sidecar, not injected into prompts. After prediction,
the evaluator verifies reference identities and writes reference-versus-generated
explanation diagnostics. Altered references or cross-split manifests fail.
Missing outputs retain their gold references and count as unanswered errors;
an absent generated explanation with an available reference scores zero.

The current automatic explanation measures are lexical diagnostics, NOT the
paper's BERTScore/BLEURT reproduction and NOT automatic semantic verification.
Do not accept or reject a tribunal update based on lexical overlap. Use the
aligned references and actual images to review reasoning; `faithfulness_verified`
remains null until independently assessed. No new evaluator models or GPU
dependencies were added in this workflow change.

```powershell
python -m evaluation.evaluate_predictions --input outputs/dev_after/predictions.csv --output-dir outputs/dev_after_metrics

# Explicit before/after code comparison; dataset/model identities must still match.
python -m evaluation.compare_runs --control outputs/dev_before/predictions.csv --treatment outputs/dev_after/predictions.csv --code-change-description "Describe the actual tribunal repair here" --output-dir outputs/dev_comparison
```

For an unchanged-code ablation, omit the code description and declare only
changed settings, e.g. `--allow-setting-change judge_mode`. Feedback ablations
can explicitly declare `feedback_mode` and `verified_feedback_sha256` as the
treatment. All undeclared changes still fail. A code comparison evaluates the
whole declared revision; it does not prove which individual edit caused a gain.
Historical runs missing the new provenance/configuration may require
`--allow-legacy`; those remain diagnostic comparisons.

## Development versus final reporting

Default `--run-purpose diagnostic` permits inspection/replay on any split.
`evaluation` is allowed on validation/test with disabled or frozen verified
feedback, never online gold-informed collection/calibration. The flag does NOT
certify that samples are untouched. Once test outcomes guide prompt/code/memory
changes, disclose that exposure and do not use those cases as untouched final
evidence. Replaying development cases is useful, but also check independent
cases before accepting a repair.

Keep the official benchmark intact and report known source/label imbalance and
image overlaps. In particular, the New Yorker source has only ENTAILS examples.
Use per-source/per-phenomenon breakdowns; source correctness does not prove an
absence of shortcuts. Preserve separate sensitivity analyses rather than
silently deleting or rebalancing official cases.

## Feedback isolation

Gold-informed feedback collection/calibration is restricted to diagnostic
training or validation runs. The offline feedback builder now requires complete
outputs, declared development provenance, matching source identities and
hash-bound references. It refuses test runs, missing provenance, selection-only
plans and overwriting an existing feedback file.

In `verified` mode, memory stays frozen across every batch. The previous
gold-based reliability update between batches was removed because it could
change which memories later cases retrieved. Corrections and harms are still
measured after the run. Gold explanation text is not copied into runtime memory
by the builder; its existing evidence-backed procedural-candidate logic remains.
Review proposed feedback for genuinely generalizable repairs before deploying it.

These controls let tribunal and feedback development resume without rebuilding
the dataset. They establish reproducibility and separation, not a guarantee
that every interpretation or every future research claim is correct.
