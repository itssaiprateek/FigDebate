# Where the current system's errors originate

This is a read-only diagnosis of production code and saved traces, not a model rerun. It examines the five original random ten-case batches and the latest matched ten-case rerun. The rerun repeats earlier cases; the 60 records are not 60 independent examples. Eight key production files were byte-compared with the latest run's source archive and all matched: arbiter, visual adapter/agent, claim extraction, claim graph, question router, compact proposer and V5 verification.

Detailed extracted evidence and a reproducible, model-free audit are in `../../research/system_origin_audit_20260924/`. The audit reads stored references only for offline comparison; no references were sent to a model. Production files were not changed.

## Main conclusion

The faults do not all start in the tribunal. There are at least four distinct mechanisms:

1. Initial model outputs can be semantically wrong even with valid structure.
2. Generation limits and validation rules can cut off or discard useful output.
3. Routing and information selection can turn a specific uncertainty into a broad task, or omit useful prior interpretation.
4. The arbiter and tribunal can introduce new reasoning errors despite having adequate relevant inputs.

We can now identify concrete origins for several failures. We have not isolated every semantic error's causal contribution with controlled fresh stage replacements, so this report does not claim a complete causal explanation of accuracy.

## 1. Unfinished final explanations: a confirmed initial-arbiter defect

**Code:** `arbiter/arbiter.py:432`, `:721`, `:1151`, `:1222`.

The initial arbiter is asked to produce four lines: visual evidence, evidence IDs, caption meaning and relation analysis. The generation call allows only 112 output tokens. Its wrapper records token counts but does not record/check EOS completion or reject an unfinished explanation. The generated assessment is then used in a separate binary scoring prompt and saved verbatim as the explanation. A binary result is marked valid without establishing that its explanation finished or supports that label.

Saved evidence:

| Measure | Original five batches | Latest matched rerun |
|---|---:|---:|
| Initial arbiter calls | 50 | 10 |
| Calls producing exactly 112 tokens | 38 | 9 |
| Initial explanations without terminal punctuation | 36 | 8 |
| Final reason identical to initial explanation | 47 | 10 |

Token-limit equality and missing punctuation alone are not a perfect sentence-completion test. However, the latest raw explanations include unambiguous cut endings such as `(l`, `an indication of a`, `Therefore,`, `valuable or`, and `The calm smile and`. These occur in 112-token outputs. This establishes an actual cutoff mechanism, not just a stylistic concern.

The tribunal does not repair the explanation whenever it rejects a revision. In tribunal mode the witness hearing preserves the initial decision, and a failed tribunal proposal leaves it in place. Therefore, a final-looking report can contain prose that was already incomplete before any tribunal call.

There is also input summarization: the arbiter includes only the first five OCR/fact/relation items, slices description to 700 characters and some language fields to 350/250 characters, and tokenizes the assessment prompt with a 2,048-token truncation ceiling. These are confirmed code paths. None of the 60 inspected initial-arbiter generation events reached 2,048 input tokens, so input-token overflow is not demonstrated as the cause here. Loss of deciding content through the earlier field/item limits needs separate checking.

## 2. Missing OCR: output capacity and rejection can be the origin

**Code:** `agents/visual_adapter.py:140`, `:192`, `:321`, `:352`; `agents/visual_grounding.py:454`, `:588`.

The initial OCR question asks for all readable text. Its normal budget is 180 tokens, and a truncation retry can receive 360 tokens. The answer validator separately allows at most 180 words. A long document can exceed either limit even when the model is transcribing rather than hallucinating. The OCR retry instruction again asks for the requested visible text; it does not divide a long region into guaranteed bounded pieces.

In the latest briefing-document trace:

- Whole-image OCR used 180 tokens, then 360 tokens; neither attempt ended with EOS. The final OCR response was rejected as truncated.
- The top and bottom crop retries ended normally with EOS, but contained 230 and 188 words. Both were rejected as `answer_too_long`.
- The public `visible_text` field consequently became an empty list and `ocr_usable=False`.
- A short topical summary survived in visual facts and the later witness answer, so the entire document was not wholly invisible to the pipeline. The exact OCR was lost from that field.

This is not evidence that the missing OCR was caused by a tribunal question: it happened during initial Agent 1 processing. Nor does a long response itself establish hallucination.

The latest initial visual stage had 100 question records: 17 needed retries, with 10 primary truncation errors and seven primary length errors; three remained invalid after retry. All three remaining errors were the OCR records above. Across the original five batches, 58 of 512 question records needed retries and five remained invalid. Counts include regional OCR questions; they are not counts of independent cases.

### A smaller reproducible contract mismatch

The yes/no prompt permits YES/NO/UNCLEAR followed by five factual words, while the validator permits five words total. The six-word answer `YES, symbol attached to red bottle.` follows the prompt but is rejected by the actual validator. This can create unnecessary retries. It does not explain every observed overlong yes/no response.

### Why a summary can still say complete

`_atomic_schema` requires only scene, objects, facts and scene type for its `format_valid`/`schema_complete` check. OCR and relation failures remain in `schema_issues`, but do not necessarily make `schema_complete=False`. The briefing trace has `schema_complete=True` alongside three failed OCR records. Thus that aggregate field means the minimum structural record exists, not that all requested evidence was obtained.

## 3. Broad questions: confirmed routing behavior, not necessarily tribunal invention

**Code:** `engine/claim_graph.py:48`, `:219`; `engine/question_router.py:121`; `engine/batch_runner.py:1121`; `engine/candidate_cases.py:101`.

All 50 original cases and all ten rerun cases use `UNDECOMPOSED_SOURCE`: the complete caption is preserved as one obligation because the previous decomposer was not qualified. This is an intentional source-preservation safeguard, not a failed request for an answer.

The router handles this representation before its more specialized branches and returns a generic pair of questions:

> Report visible text with its object or panel bindings, and visible actions, states and relationships. Distinguish observations from unclear details.

> Explain the complete expressed caption, preserving roles, comparisons, negation and qualifiers. Separate conventional idiom meaning and possible sarcasm from source assertions.

The latest ten stored witness records all contain these same questions. They are assembled by software before the compact tribunal review. Independent candidate questions can be recorded as judge context, but `candidate_hearing_plan` deliberately preserves the role-safe witness plan instead of substituting candidate questions.

The visual question validator checks length, question-mark count and some prohibited/leading phrases. It does not establish that one sentence requests only one fact. The generic request above passes the actual validator despite asking for text, bindings, actions, states and relationships together.

**Confirmed consequence:** the initial hearing is less case-specific than the term “targeted” suggests, and may repeat a broad description rather than resolve the exact uncertainty.

**Still unproven:** how much narrowing these questions would improve semantic accuracy. Openness is a plausible contributor to omissions and irrelevant detail, but cannot be assigned as the universal cause of hallucination. Correctly constrained questions can still receive wrong answers.

## 4. Initial visual semantics: errors can precede every tribunal question

**Code:** `agents/visual_adapter.py` standard questions; `agents/visual_grounding.py` schema assembly; `engine/evidence_ledger.py`.

The fixed questions explicitly request visible facts and discourage interpretation. Nevertheless, the answers can introduce inferred scene roles, settings or actions. Validation checks syntax, length, placeholders, prohibited verdicts and some incompleteness; it does not independently verify each visible claim.

The football/therapy trace first calls the seated person a coach in `initial_scene`, repeats coach in the object list and facts, and even includes stadium/crowd in the object list. The human reference describes a therapist context and the sporting/psychological double meaning of defensive. This is a reference-grounding discrepancy that originates in the initial visual output, before tribunal questioning. This audit did not independently inspect every image, so it does not label every disputed visual detail a proven hallucination.

The same role assumption then appears in the initial arbiter, proposer and verifier. Those repetitions show propagation; they are not independent confirmations. Whether correcting that observation alone would recover the proper final reasoning remains to be tested.

## 5. Initial language semantics: delivery works better than semantic consistency

**Code:** `agents/claim_extraction.py:312`, `:531`; `engine/claim_semantics.py`; `engine/caption_answers.py:24`; `arbiter/arbiter.py:161`.

Initial language processing generates an expressed-claim record and a separate figurative interpretation. In the inspected 60 records, neither core nor interpretation diagnostics reported a token-limit hit. All ten latest caption witness questions have an ANSWERED status. That is evidence against missing language generation being the main latest-run failure; it is not evidence of correct interpretation.

Specific observed semantic defects include:

- A generated negation list containing `is` and `good` for the bad-advice caption.
- An invented alternative that breaking bones is a natural unavoidable part of life.
- A briefing-pack interpretation saying it is “not amazing” while its `intended_polarity` says positive.

These errors already exist in Agent 2 output. Exact source preservation does not repair every auxiliary field.

The code explicitly leaves semantic qualification pending. The arbiter includes the intended-meaning field only when `semantic_qualified=True`; none of these 60 claim contracts qualified. Some other language fields still enter its summary. The compact proposer separately receives the raw caption and literal observations, not the structured Agent 2 reading or its witness answer. Consequently, a useful caption interpretation can exist without being consumed at a later decision point. This is a deliberate trust/interface choice, not a dropped network message; benefit from changing it is unqualified.

## 6. Initial arbiter semantics: full answers can already be wrong

**Code:** `arbiter/arbiter.py:432`, `:599`, `:792`, `:1210`.

The arbiter reasons from textual visual reports and caption analysis. It generates an assessment, then selects between support and conflict using position-balanced likelihoods. It can retain an internal INSUFFICIENT evidence relation while returning a binary benchmark label. All ten latest initial decisions have that insufficient relation; six assessments explicitly describe insufficient evidence or no direct support/conflict.

This explains why “binary decision valid” does not mean that the accompanying rationale proves it. There is no final check here requiring the generated explanation to justify the selected label.

The before/after comparison trace is especially informative: the visual testimony preserves the negative-before/positive-after pattern, and Agent 2 preserves the caption's initial-positive/stagnates assertion. Yet the initial arbiter describes shock/disapproval as representing the initial positive reception. The incorrect semantic reconciliation is already present in the initial arbiter, not first invented by the tribunal.

That local trace does not prove which model input caused the misconception. It does show that both relevant source sides were available and that a later reasoning stage introduced a contradiction between them.

## 7. Tribunal semantics: additional independent failures are confirmed

**Code:** `engine/simple_judge.py:75`; `engine/evidence_review_v5.py:365`, `:395`, `:433`; `engine/tribunal_repair.py:146`.

Examples demonstrate distinct mechanisms:

- **Proposer changes the claim:** Agent 2 and the caption witness correctly retain “wanted a giant cat”; the compact proposer changes it to a claim that the dog is a cat. This semantic substitution originates at the proposer in that trace, even though there are other earlier weaknesses in the image description.
- **Relation substitutes topic matching for the whole assertion:** the briefing verifier supports an evaluative caption because document subject and topics match. The evaluation itself is not established.
- **False audit error:** the fertilizer proposer and verifier agree on CONFLICT and give a compatible reason, while the challenge lists `CONFLICT` as a decision error. The error is introduced in the challenge's semantic bookkeeping.
- **Candidate/verifier target mismatch:** the challenge sees the verifier's decision plus candidate interpretation and reason, without a separately named candidate relation. This makes the object of its audit ambiguous when they disagree. The payload is confirmed; the benefit of an explicit-target replacement remains experimentally unqualified.
- **Repair cannot always revise the failed hypothesis:** eight latest follow-ups reuse a frozen compact proposal. This preserves proof identity but means a wrong proposal cannot be replaced on those paths. A second hearing is therefore not necessarily a second attempt to solve the original task.

Role mapping receives selected observation text without an image. Evidence selection is capped at four records and coverage is reported by the model. These are information constraints whose causal contribution remains unisolated. Same-model agreement is not independent truth certification.

## 8. Apparent missing answers include deliberate stops

The latest batch contains ten binary final labels and ten nonempty final reasons: it does not have ten missing final predictions. Different missing-looking fields mean different things:

| Observation | Actual mechanism |
|---|---|
| Empty initial OCR field | Generated responses rejected by token/word contracts |
| Null optional claim field | Allowed representation; may be appropriate or may omit a role; inspect related fields before judging |
| Visual witness says UNRESOLVED/ABSTAIN | Often intentional: it reports observations and is prohibited from deciding the caption label |
| No final challenge object | An earlier verification stage stopped or proposer abstained |
| Final reason incomplete | Initial arbiter prose can be copied unchanged after review rejection |
| Generic delivered explanation | Software-generated audit status, separate from the model's actual rationale |

There were 19 latest review rounds and eight without a challenge object: six stopped at relation direction, one at mapping, and one proposal abstained before proof generation. Those missing challenges are skipped stages, not eight lost model responses. Separately, two executed audit outputs were invalid/incomplete and two terminal judgments were contradictory.

## 9. Human explanations and the current measurement gap

**Code:** `evaluation/evaluate_predictions.py:169`; `engine/final_artifact.py:59`, `:74`, `:80`.

The evaluator already compares `final_reason` against `reference_explanation` with token-set F1 and ROUGE-L. Latest means are 0.2872 and 0.1837. These are lexical measures, not demonstrated semantic reasoning quality; the metric metadata explicitly says automatic semantic verification is false.

In this rerun every `final_reason` is inherited from the initial arbiter. A change in tribunal reasoning can therefore be invisible to the final explanation score when its proposal is rejected. Conversely, a low score can reflect incomplete baseline prose as well as semantic mismatch. The software-rendered `delivered_explanation` describes evidence-chain/acceptance status and should not be substituted for a substantive model explanation when evaluating reasoning.

The user's human-reference requirement should be evaluated separately for initial rationale, proposed tribunal rationale and accepted final rationale, including proposition preservation, deciding evidence, figurative mechanism and inference. Reference similarity and correct labels must remain separate from soundness. No new semantic metric was implemented in this diagnostic task.

## What is established and what remains to test

**Established origins:** arbiter output cap without completion validation; initial OCR capacity/rejection losses; generic whole-caption hearing route; initial visual-role and language-field discrepancies; new claim distortion at the proposer; relation/audit mistakes; frozen candidate repair limits; incomplete rationale propagation into export.

**Not established:** that open-ended questions are the dominant cause; that increasing token budgets alone fixes reasoning; that a neutral visual rewrite alone fixes a downstream case; that changing the gate is safe; that every difference from a human explanation is wrong; or that any proposed mechanism improves the dataset generally.

The next controlled checks should isolate these mechanisms: completion under a consistent output contract, bounded OCR coverage, one deciding witness question versus the generic hearing, correct observations/claim interpretation supplied one at a time, and sound versus defective arguments passed through the actual verifier and gate. These must remain diagnostic interventions with fixed source/configuration comparisons. Production fixes and model reruns were not part of this read-only audit.
