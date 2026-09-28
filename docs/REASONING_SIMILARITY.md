# Compare explanations with the human reference

Run this **after inference**. It never supplies human explanations to agents or
changes the acceptance gate. Use `records.jsonl` for initial, each tribunal
proposal, and final explanations. A CSV exposes only the stages it exports.

Install the optional `bert-score==0.3.13` dependency in the evaluation environment.
Provide a local `FacebookAI/roberta-large` model at a pinned revision. The tested
revision is `722cf37b1afa9454edce342e7895e588b6ff1d59`. The evaluator does not
download a model. The primary inference environment need not import BERTScore.

```powershell
python -m evaluation.reasoning_similarity --input runs/your_run/records.jsonl --model-path "C:/path/to/roberta-large" --model-revision 722cf37b1afa9454edce342e7895e588b6ff1d59
```

Outputs are `reasoning_similarity.json` and `reasoning_similarity.csv`, with
per-case BERTScore precision, recall and F1, stage means, and paired final-minus-
initial F1. The protocol uses layer 17, CPU, no IDF weighting, no baseline
rescaling, and the slow tokenizer. Versions, revision and input hash are recorded.

Empty answers receive zero and remain in the denominator. Missing references
are unavailable, not zero. Text exceeding the encoder limit is explicitly
reported as unscored rather than silently truncated. These counts appear beside
the means; comparisons must use the same cohort and settings.

BERTScore measures contextual similarity. A negated or role-reversed answer can
still score highly. Report label accuracy, helpful/harmful changes, completeness,
image faithfulness and human-reference reasoning coverage separately. Review
participants, polarity, modality, comparisons and the figurative connection.
The metric must never be used alone as a semantic acceptance criterion.

The historical `research/adoption_20260924/score_run.py` launcher is not shipped
in the current checkout. Use the public module command above. Optional scoring
weights must be available locally; pipeline setup does not imply that the
optional explanation metric has been installed or run.

The 2026-09-28 initial-stage qualification exports one
`similarity_input_<variant>.jsonl` per variant for this evaluator. Keep the same
metric configuration across variants. Its separate `reference_reasoning_review.json`
records an assistant's post-inference comparison with the images and references;
it is not BERTScore and is not an independent human study.
