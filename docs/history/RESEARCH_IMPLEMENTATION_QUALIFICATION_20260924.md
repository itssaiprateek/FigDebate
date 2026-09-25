# FigDebate research implementation and qualification

Closed after the user requested ending the run. All planned development variants
finished; the complete-pipeline qualification was interrupted. No experimental
inference change was promoted. The independently tested reference-similarity
evaluator remains available. Experimental source copies, completed results and
the partial candidate checkpoint are retained in `../../research/adoption_20260924/`.

## Scope and fairness

The starting point was the existing working tree, including earlier uncommitted
work. Its source hashes and Git status were frozen before changes. The comparison
does not silently substitute the last commit for the actual starting system.

Each tribunal variant receives the same six saved first-hearing inputs, complete
evidence catalogue, images, seed and acceptance rules. It runs actual proposal,
verification, acceptance and scheduled follow-up paths. Feedback is disabled.
Gold decisions and human explanations are only used after inference. Separate
upstream trials rerun the actual visual or initial-arbiter stage. These selected
development examples diagnose mechanisms; their accuracy is not a held-out estimate.

The software has a 120-second judge-call limit and a cumulative 240-second judge
budget per case. Extra regional/verification calls and larger output allowances
are disclosed resource changes. Identical budgets do not mean identical compute.
Runtime varies with laptop GPU clock states; single-run timings are descriptive.
The ordered-grammar trial lost one second hearing to the remaining-budget check
while identical requests slowed down. Its result is not an equal-opportunity
measure of semantic capability under stable hardware conditions.

Two preflight pilots were excluded for harness defects: an empty evidence catalogue
and a wrapper signature that accidentally disabled constrained generation. Both
were stopped, fixed and retained. The counted control and all arms use the same
corrected harness. Completed neutral and harmful trials remain in the report.
See `methods_and_exclusions.md`, `protocol.json`, manifests and raw call logs in
the research directory for reproducibility.

## Experimental implementations and development outcomes

All label counts below concern the same six selected examples. The control is
2/6 correct. A correct changed label is not sufficient for promotion.

| Adaptation | Implemented mechanism | Observed outcome | Adoption decision |
|---|---|---|---|
| Multimodal coreference | Image access in the actual role-mapping call; immutable source spans retained | 3/6 correct, one helpful label change, no harmful label changes. Changed explanation still describes a coach and huddle while missing therapist/football wordplay. | Do not promote a label-only gain. |
| Regional verification | One selected observation gets scope classification, a normalized region where appropriate, independent crop description, full-image confirmation and provenance binding | Original arm 2/6; catches a false container-color description, but three GLOBAL outputs violate the empty-box contract. | Experimental; conditional-grammar follow-up assessed separately below. |
| Regional grammar follow-up | Enforce LOCAL versus non-local box cardinality during decoding | 2/6; all three box-format failures removed, one color error caught, unsupported coach role still accepted. Four second hearings skipped for insufficient remaining budget under slow hardware. | Do not promote; no delivered reasoning benefit and unresolved semantic/resource concerns. |
| Decode-time dependencies | Actual llguidance masks restrict compatible audit status, relation and alternative fields; uncertainty remains expressible | Original arm 2/6. A placeholder alternative and dependency loss during field repair expose integration problems. | No demonstrated answer improvement. |
| Ordered grammar follow-up | Generate status before dependent fields; preserve the same branches during repair | 2/6; no accepted corrections. One follow-up skipped during hardware slowdown. | No promotion; timing-confounded negative evidence is not proof the idea cannot help. |
| Audit isolation | Explicit candidate and independently reconstructed assessment targets; existing fresh source/image relation pass retained | 2/6; no accepted correction, with additional unresolved/incomplete audits. | No demonstrated benefit. |
| Factored verification repair | Generate a granular question; answer from source/pixels without draft; rebuild candidate and invalidate cached proof only when answered | 2/6; five UNCLEAR checks and one invalid broad question. No live successful rebuilt candidate. | Do not promote. |
| Longer, shorter-format initial explanation | 224-token allowance and concise prompt | 2/6; one helpful and one harmful label change. Removes cap hits, but meanings remain wrong. | Reject; initial arbiter stays unchanged. |
| Budget-only initial explanation follow-up | Original prompt with 224 tokens | 1/6; zero helpful, one harmful change. Removes cap hits and preserves old response prefixes. | Reject; initial arbiter stays unchanged. |
| Visual answer retention | Raise the OCR word ceiling from 180 to 360; allow YES/NO plus five factual words. Generation budgets stay unchanged. | Two previously discarded completed crop readings retained; zero newly invalid answers. Six-case mean completion 0.95455 to 0.98485; delivered OCR words 83 to 457. | Promising delivery fix, unpromoted because complete-pipeline qualification was interrupted. |

The regional experiment is a small R-CoV-inspired adaptation, not a reproduction
of the paper's multiple samples and external assistant. A box is a location
hypothesis, not proof of an abstract interpretation. The repair experiment follows
CoVe's factored information flow using the same model and original sources; it is
not independent-model or external-database verification. Existing source-spans,
decode-time masking and fresh message construction were not replaced simply to
match the terminology in the supplied research summary.

The regional grammar follow-up removes the box-format mismatch but exposes a
remaining semantic failure: a selected observation says the person is a coach.
The region description sees a seated person with a notepad; the final regional
check marks the observation supported without establishing the profession. This
is precisely the difference between locating visible attributes and verifying
an inferred role. One successfully rejected color error does not establish that
this mechanism fixes abstract-inference hallucination.

The frozen repair implementation also has a reporting defect: malformed
verification output is classified as a runtime failure. An added adversarial test
fails and is retained. Valid uncertainty remains correctly represented. A pending
reporting correction exists separately; it was not silently inserted into the
already-running arm. A scripted ANSWERED-path check confirms fresh proposal
generation, cache invalidation and all four proof obligations. This proves control
flow, not model ability to supply a useful answer.
The pending reporting correction also passes a post-processing replay of all six
saved verification records and injected timeout checks. The original frozen live
run remains unchanged; this reporting-only replay is not another model experiment.

The regional follow-up made 41 judge calls versus the control's 34, while only
one of six cases received a second hearing (control: six). Added verification
work and slower hardware both contributed; the observed 1,141.6 versus 456.6
seconds cannot be attributed solely to the algorithm. Raw matched-request
telemetry and follow-up budget decisions are preserved. All six delivered
explanations remain identical to control.

All development results, including unsuccessful variants, are in
[the compact result record](../../../research/adoption_20260924/compact_results.json).
The full hypotheses, adaptations and primary papers are in
[the research review](RESEARCH_ADOPTION_REVIEW_20260924.md).

## What the upstream tests establish

The document's two crop transcriptions contain 230 and 188 words and end normally.
Their raw model outputs are identical between control and retention candidate.
The old 180-word validator discards both; the candidate keeps them. The whole-page
302-word response actually hits the 360-token limit without EOS and remains
rejected in both arms. The existing line limit still bounds delivered crop text.
Retention is therefore not evidence of complete document transcription.

Other five examples retain identical public visual evidence. Two short presence
answers stop being unnecessarily retried because the validator now matches the
prompt's allowance of YES/NO plus five words. Generation budgets, truncation checks
and repetition checks are unchanged in this candidate.

Five baseline initial explanations reach 112 tokens, but one ends naturally at
exactly that count. Four demonstrably continue when the allowance is increased.
Reporting all five as proven truncations would be wrong. The simple incomplete
clause detector flags only two; it is not a comprehensive completeness metric.
Larger budgets improve delivery but harm one decision, so delivery and semantic
quality must remain separate measures.

## Reasoning compared with the human dataset explanation

`evaluation/reasoning_similarity.py` supplies a reusable, offline BERTScore report
for initial, proposed and delivered final explanations. It exports precision,
recall and F1 per case and stage, paired final-minus-initial changes, empty-answer
counts and explicit exclusions for missing references or overlength input.
References never enter inference through this evaluator. Empty delivered answers
score zero; missing references are unavailable, not zero-quality answers.

Configuration: BERTScore 0.3.13, RoBERTa-large commit
`722cf37b1afa9454edce342e7895e588b6ff1d59`, layer 17, slow tokenizer, no IDF or
baseline rescaling, CPU inference. Version, source and configuration hashes are
written with each report. See [usage](../REASONING_SIMILARITY.md).

On development final reasons, control mean F1 is 0.84850 and mapping is 0.85018.
The mapping explanation still omits the decisive wordplay. The harmful budget-only
variant also scores slightly higher, 0.84953. Similarity alone would select a
misleading winner. A sanity check scores identical text 1.0000, a negation 0.9544
and unrelated text 0.8750. These values are not accuracy percentages.

The frozen reasoning rubric separately assesses participant roles, polarity and
modality, image faithfulness, the reference's figurative mechanism and complete
conclusions. Masked A/B assessment by the same assistant is documented as such;
it is not independent human annotation. Reference explanations can be ambiguous,
and a justified alternative should not be rejected merely for different wording.

## Complete-pipeline qualification and stop status

The user requested ending the run during the dense-document candidate trial.
The queue and its active inference process were stopped; the GPU returned to idle.
Completed outputs and checkpoints were preserved. No unseen case was evaluated,
and no incomplete candidate run is counted as a completed comparison.

The selected candidate contains only the two visual-retention limits. Its source
and selection were frozen before validation. It passed 677 software tests with
the same four skips and one expected failure as control.

| Complete-pipeline check | Status and observations |
|---|---|
| Dense-document baseline | Completed from the image with no checkpoint reuse; 31 generation requests, 3,688 output tokens, 945.95 seconds. Initial/final ENTAILS against dataset CONTRADICTS. Valid final output and judge context; review blocked at relation and follow-up skipped for remaining budget. |
| Dense-document candidate | Interrupted during language processing after visual extraction completed. Visual completion increased from 0.7273 to 0.9091 and delivered OCR from 0 to 374 words. Both completed crops retained; truncated whole-page output rejected. No candidate final decision exists. |
| Six preselected validation cases, baseline and candidate | Neither run started. No held-out comparison or general accuracy gain is established. |

The six-case cohort was fixed before candidate outcomes: two humor, two metaphor
and two sarcasm cases with no known record, exact-image or caption overlap in
inspected previous runs. This does not guarantee no manual exposure. The separate
dense-document case is a development regression, not held-out evidence.

**Production decision:** retain the existing inference pipeline. Do not promote
the OCR candidate until downstream effects and all six validation pairs have been
checked. The semantic variants also remain experimental. The reference-similarity
evaluator is a separate tested addition and does not affect model decisions.
See [stop record](../../../research/adoption_20260924/user_stop.json) and
[candidate status](../../../research/adoption_20260924/selected_candidate.json).

## Remaining system problems

Errors originate at multiple stages: wrong initial scene/entity descriptions;
language interpretation that changes intended versus actual states; a proposer
that rewrites the source assertion; role binding that certifies a shared mistaken
interpretation; and verification questions that repeat the disputed assumption.
The tribunal can block harmful changes without supplying a better final reason.
When it abstains, the system retains the initial explanation, including any
truncation or incorrect reasoning. Structural consistency, judge agreement and
larger outputs do not by themselves solve these semantic errors.

## Reproduction and software checks

The original Python environment lacked its base interpreter. Research restored
an official Python 3.11.9 embedded runtime without modifying that environment,
and uses the existing pinned model weights. From the outer workspace:

```powershell
$python = './research/adoption_20260924/runtime/python311/python.exe'
& $python research/adoption_20260924/run_current.py --help
& $python research/adoption_20260924/score_run.py --input '<run-directory>/records.jsonl' --output-dir '<evaluation-directory>'
```

Keep `--feedback-mode disabled` in future comparisons. Full-dataset runs remain
user-run. The baseline passed 675 software tests; adding the metric passed 679,
both with four existing skips and one expected failure. Candidate tests and final
promotion status are recorded above. There are no active inference jobs from this
qualification. The frozen starting source hashes were rechecked before closure;
no original inference source file was changed by these experiments.
New-file whitespace checks passed. The repository-wide diff check flags two
preexisting extra blank lines at EOF in `agents/visual_grounding.py:1992` and
`engine/question_router.py:317`; their bytes match the frozen baseline and were
left unchanged.
