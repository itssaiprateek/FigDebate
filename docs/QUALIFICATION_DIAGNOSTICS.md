# Offline qualification diagnostics

The evaluation module `evaluation.qualification_report` reports missing or possibly unfinished explanations, exact copies between stages, reviewed semantic errors, and helpful/harmful replacements. It never changes a prediction, explanation, acceptance threshold, or feedback state.

From the release directory, after inference:

```powershell
python -m evaluation.qualification_report PATH_TO_RECORDS.jsonl --references PATH_TO_REFERENCE_MAP.json --output PATH_TO_REPORT.json
```

The reference map is keyed by case ID with a `label` field. Keep this file outside inference inputs. The existing `evaluation.reasoning_similarity` evaluator separately compares delivered explanations against dataset explanations. Its RoBERTa BERTScore measures similarity; it is not proof of image faithfulness, complete reasoning, or the paper's different ExplanationScore configuration.

An optional `--semantic-reviews` file supplies an ID-to-stage map, for example
`{"case-id": {"proposal_1": {"semantic_sound": false, "reason": "The inference reverses the source chronology."}}}`.
Use `visual_agent`, `language_agent`, `initial_arbiter`, `proposal_1` (and subsequent
round numbers), or `final` for stage names. Record who reviewed the examples and
whether they saw references in the accompanying evaluation report. Reference
similarity and label agreement never fill in these judgments automatically.

Completion flags are inspection signals. A generation reaching its token cap may still finish a sentence. Missing punctuation alone is not labeled truncation. The first recorded bad explanation is not necessarily the causal origin of a semantic mistake. Semantic error origins require explicit stage reviews; unreviewed stages remain unknown.

Proposal counts are review attempts, including frozen candidates re-audited in later rounds; `reused_proposal_reviews` exposes that repetition. Missing final predictions remain in the referenced-case accuracy denominator. Missing reference labels are reported separately and do not become incorrect predictions. The all-case correct fraction is a lower bound when references are missing.

`calibration_eligibility` checks prerequisites for a future calibration experiment: a qualified verifier, disjoint image groups, and a frozen criterion. It does not fit a threshold or authorize production changes. Never relax source identity or proof validity to raise acceptance counts.

The September 25 semantic candidates and their frozen baseline are outside the release under `research/implementation_20260925`. They require measured qualification before adoption. Initial arbiter behavior, OCR retention, and disabled feedback remain unchanged.
