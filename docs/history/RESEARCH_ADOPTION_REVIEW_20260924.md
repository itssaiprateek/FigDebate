# FigDebate: research review and bounded adoption plan

Reviewed 24 September 2026. This is a research/code review and experiment specification, not a production implementation or a new model evaluation. Existing performance figures come from saved runs. The accompanying `RESEARCH_REFERENCE_SCREEN_20260924.md` accounts for every entry in the supplied reference list and distinguishes method review, abstract screening, duplicates and unresolved citations.

## Assessment

The research supports testing better visual grounding, less biased verification and checks on discrete consistency. It does not support the five proposals as written. Several describe guarantees the papers do not establish; two target modules that do not own the stated problem. No reviewed paper demonstrates an improvement on our current Qwen/Mistral configuration and figurative-language dataset.

| Proposal | Decision for FigDebate |
|---|---|
| Rewrite `source_spans.py` for multimodal coreference | Preserve exact source spans. Test image-conditioned role binding in the actual mapping stage. |
| Require a box for every visual fact | Test regional verification of disputed, localizable observations. Allow global, relational, inferred and unresolved claims with appropriate evidence. |
| Replace static validation with decode-time grammars | Decode-time masking already exists. Add missing finite dependencies or derive redundant fields in software; retain independent semantic checks. |
| Give the tribunal a fresh context | Calls already have fresh message lists. Reduce anchoring in the information supplied to the audit and measure shared errors. |
| Abolish repair and replace it with external CoVe | Keep the bounded scheduler and provenance checks. Test repairs that obtain new evidence through independently answered questions. |

## What our evidence establishes

The [system-origin audit](SYSTEM_ERROR_ORIGINS_20260924.md) and [context handoff](FINAL_RUN_CONTEXT_20260924.md) distinguish confirmed mechanisms from plausible semantic explanations.

- **Unfinished explanations:** 38/50 original initial-arbiter calls and 9/10 latest calls reached their 112-token output allowance. The latest traces contain plainly unfinished clauses. All ten latest final reasons preserve the initial explanation. This defect starts before the tribunal.
- **Missing OCR:** in the briefing-document case, whole-image transcription was truncated; regional retries ended normally but exceeded the separate word limit and were discarded. Empty OCR therefore does not establish failure to perceive text.
- **Broad questions:** all ten latest hearings used the same broad visual and language questions, selected by the software router for `UNDECOMPOSED_SOURCE`. They were not newly invented by the compact proposer. Their causal contribution to accuracy is unmeasured.
- **Semantic drift:** initial visual and language outputs can be wrong; the proposer can also introduce a new error despite a useful earlier interpretation. For example, a wish for a giant cat becomes an assertion that a depicted dog is a cat. A box cannot distinguish a wish from an assertion.
- **Audit and acceptance:** the latest diagnostic finishes all ten cases but remains 3/10 correct. Four proposals with correct changed labels are rejected; two harmful proposals are blocked. Correct proposed labels do not necessarily have correct explanations. Nine repair follow-ups yield no accepted repair. These are development examples, not held-out estimates.

The five original random batches ended at 27/50 correct versus 28/50 initially. Execution reliability, valid structure and faster runtime have not yet established a net semantic improvement. The ten-case rerun overlaps those examples and must not be counted as ten additional independent cases.

## 1. Multimodal coreference: adopt the separation of tasks

**Original method.** TikTalkCoref concerns Chinese social-media dialogues and videos, particularly references to people. Its benchmark approach separates textual coreference, visual person tracking and text-to-visual alignment. It is not a general solution to sarcasm, intent, modality or image-caption entailment. [Paper and methods](https://arxiv.org/html/2504.14321v2).

**Current code.** `engine/source_spans.py:31` produces legal token intervals and reconstructs quotations from the immutable caption. That is a source-integrity function, not an entity resolver. It already binds interval endpoints through a finite schema. Actual role binding occurs in `engine/evidence_review_v5.py:367`; the mapping call explicitly uses `picture=None`, receiving textual observations selected by the preceding visual call.

**Adaptation.** Retain exact caption spans and add image access to the actual mapping experiment. Each necessary role should identify its source span, candidate visual entity/evidence IDs, panel or time scope, and whether the mapping is literal, figurative or unresolved. Preserve modality and the complete caption independently: a participant mapping must not convert desired, hypothetical or reported states into actual states. Metaphorical participants may have no literal object box.

Compare current text-only mapping against image-plus-text mapping using identical upstream evidence and downstream gates. Include ambiguous referents, panel swaps, speaker/subject confusion and intended-versus-actual outcomes. Judge role correctness separately from final labels. On our hardware, image access consumes context and time; record both.

## 2. R-CoV: a box is a location hypothesis, not proof

**Original method.** R-CoV extracts entities, proposes regions, describes those regions independently, verifies and revises. It explicitly allows inaccurate boxes. Its reported setup uses seven region-description samples, a 48 GB GPU and default external language-model assistance, alongside an ablation without that assistance. Its evaluations emphasize object hallucination, rather than figurative entailment. [Method and experimental setup](https://arxiv.org/html/2604.20696v1).

**Current code.** Initial visual answers are filtered mainly for delivery and format; they are not independently verified against each claimed region. V5 attaches locations and selects existing catalogue evidence. That can preserve an already wrong description. Our OCR failures also show that cropping alone does not fix capacity and validation mismatches.

**Adaptation.** Start with one disputed object, attribute or OCR binding, not every fact. Record original image dimensions, coordinate convention and crop transform; reject invalid or empty regions. Independently describe the crop without the proposed object name or answer, while retaining a full-image check for relations. Compare the new description with the disputed claim and retain conflicts instead of silently replacing evidence.

Use evidence types: local observation with region, relation with relevant regions, whole-scene observation, inference citing observations, or unresolved. “This person is a therapist,” “the document is amazing,” and “the character wanted this outcome” cannot be certified merely by drawing a rectangle. Nonexistence also has no object box; regional absence alone cannot establish absence from the entire image.

A one-region, one-answer pilot is an **R-CoV-inspired adaptation**, not a reproduction of the paper. Benchmark its cost before increasing samples. Woodpecker offers a closer precedent for external visual verification: it forms questions, checks them using expert models, builds visual claims and then corrects the response. [Woodpecker](https://arxiv.org/html/2310.16045v2).

## 3. Grammars: extend enforceable consistency, not claimed semantic guarantees

**Original method.** Decode-Time Grammars binds grammar fragments to a runtime environment, so generated declarations can constrain later valid references. Its formal scope concerns available names, types and scope under stated assumptions. Compilation and functional correctness still require an oracle. It does not prove natural-language claims true. [Paper, including limitations](https://arxiv.org/html/2607.18357v1).

**Current code.** `engine/output_contracts.py:72` delegates to `engine/structured_decoder.py`, which uses pinned llguidance token masks and fails closed. `models/judge_model.py:312` installs the mask during generation. Evidence-ID enums and source intervals are already input-dependent. Thus this is not a migration from post-hoc JSON validation to constrained decoding.

**Adaptation.** Find discrete facts currently generated twice and remove redundant degrees of freedom. Generate canonical condition assessments, then compute logically determined summary fields where the contract permits it. Alternatively use supported finite schema branches or separate bounded generation stages. Check grammar-backend support with real decoding; schema syntax alone is insufficient.

Useful invariants include existing evidence IDs, legal intervals, valid box bounds and compatible audit status/error fields. Preserve honest `UNRESOLVED` states. Do not constrain a verifier to agree with the proposer, and do not force a desired polarity because an earlier generated field selected it. A perfectly consistent payload can still be consistently wrong.

The structured-decoding literature itself reports task-dependent quality tradeoffs. [Text-to-table evaluation](https://aclanthology.org/2025.r2lm-1.12/) reinforces the need to measure content quality separately from validity.

## 4. Audit isolation: change information exposure

**Original findings.** The correlated-panel study finds substantial shared errors across judges on NLI and preference tasks; it does not establish that clearing conversation history removes correlation. The separate consensus study also finds that aggregating samples can reinforce misconceptions. [Nine Judges](https://arxiv.org/html/2605.29800v1), [Consensus is Not Verification](https://arxiv.org/abs/2603.06612).

**Current code.** `models/judge_model.py:109` constructs a new system/user message list for each call. There is no accumulated proposer conversation in this interface. Shared model weights and repeated selected observations remain common influences. The relation call already reasons from source, observations and image without the proposer argument. The later challenge at `engine/evidence_review_v5.py:433` receives the proposer's interpreted assertion and draft argument. It receives the verifier decision, but no separately named proposer relation. `tribunal_process.py` mainly defines checks and the optional process-audit mode; it is not the owner of chat history.

**Adaptation.** Reuse the existing source-based relation pass instead of adding another generic judge. Preserve its independently reconstructed assertion, roles and modality. In the subsequent comparison make the audit target explicit: separate proposer claim/relation, verifier claim/relation and cited sources. Test whether withholding persuasive draft prose until after source analysis reduces drift and false objections. The audit must eventually see the candidate it is auditing.

Measure false acceptance and false rejection against human-adjudicated explanations, plus shared-error counts. Judge agreement alone is not success. Using a different model family is an optional later experiment, not a guarantee of independence.

## 5. Repair: independently answer questions and obtain new evidence

**Original methods.** CoVe drafts, plans questions, answers them and revises. Its factored variant hides the original draft from each answering step; the paper uses the same LLM across steps. It is therefore not inherently external verification. Its narrow open factual questions outperform yes/no variants in the reported comparison. The self-correction paper documents failures without external feedback, not a theorem prohibiting all repairs. [CoVe](https://arxiv.org/html/2309.11495v2), [Limits of intrinsic self-correction](https://arxiv.org/abs/2310.01798).

**Current code.** `engine/tribunal_repair.py` is a bounded scheduler with issue routing, budgets and proof/cache safeguards. Some paths obtain witness evidence; some tribunal-only repairs freeze the candidate. Deleting the scheduler would remove useful controls without supplying a stronger verifier.

**Adaptation.** Within the existing budget, route a failed requirement to a question with one answerable target. Use pixels/crops/OCR for a visual question and the complete source expression for a language question. Hide the candidate answer from the answering step. Record source, answer, uncertainty and evidence identity. Same-model crop inspection is source-based checking, but is not independent-model verification.

If no new evidence or justified changed interpretation is obtained, stop. If a candidate premise changes, create a new candidate version and invalidate every dependent proof; never reuse a frozen proof to accept revised reasoning. External does not mean searching for the dataset answer online. Human reference explanations remain evaluation-only.

The objective is fewer unsupported premises, not more successful retries. “Small Language Models Need Strong Verifiers” supports testing verifier competence, but its critique-training setup is not a drop-in inference prompt. [Paper](https://arxiv.org/html/2404.17140v2).

## Proposed experiments, in order

All arms should use mechanism-level rules, feedback disabled, identical case manifests and the real final acceptance path. Keep the source snapshot, prompt/schema versions, model revisions, budgets, seeds and cache provenance. Do not combine untested changes into a single final run.

| Experiment | One controlled change | Main outcome and guardrail |
|---|---|---|
| E0: delivery defects | Separately test completion-aware initial explanation generation; OCR chunking and aligned prompt/validator length limits | Complete explanations and preserved usable OCR; no invented continuations or silent loss. Report label effects separately. Initial-arbiter implementation remains outside the standing scope until explicitly broadened. |
| E1: finite consistency | Canonical audit decisions with derived redundant fields / compatible schema branches | Fewer contradictory or invalid payloads; preserve real disagreement and abstention. Include all enum combinations and real constrained decoding. |
| E2: audit exposure | Explicit proposer-versus-verifier targets and source-first assessment before draft prose | Fewer false objections and harmful endorsements, with no increase in empty or timed-out audits. |
| E3: multimodal mapping | Give the existing role-mapping stage image access | Better source-role/panel/modality preservation; no new caption rewriting or ungrounded identity claims. |
| E4a: regional verification | One targeted region verification on disputed localizable evidence | Improved observation precision **and** retention of necessary facts, within the time budget. |
| E4b: evidence-driven repair | Substitute independently answered targeted verification for repeat critique | Helpful accepted repairs minus harmful accepted repairs; record whether evidence actually changed. Test separately from E4a before combining. |

First use the known failures to confirm the mechanism, with previously correct cases as regression controls. Then use a preselected unseen bounded sample spanning literal claims, sarcasm, metaphor, humor, negation, intentions, comparisons and OCR. Keep reference explanations hidden from all inference stages. Repeatedly inspected cases are development evidence only. Full-dataset runs remain user-run under the standing instructions.

For causal diagnosis, replay unchanged saved upstream inputs while replacing only the stage under test, then propagate through the real downstream path. A manually corrected intermediate input can be used as a clearly labelled **oracle diagnostic upper bound**, never counted as a deployable model improvement. Do not claim end-to-end runtime improvements from partial replay.

Promotion requires the intended mechanism improvement, no observed new harmful final decisions on regression controls, and improved explanation quality on unseen checks at acceptable measured cost. Zero harms in a small sample is not a general guarantee. If the bounded sample is inconclusive, retain the baseline and report uncertainty rather than selecting a winning anecdote.

## Reasoning evaluation against the human reference

Existing token-set F1 and ROUGE-L measure lexical overlap; the latest report records approximately 0.287 and 0.184 respectively. They do not establish semantic correctness. Add the following as separate measures, without collapsing them into one headline score:

1. **Reference semantic similarity:** BERTScore precision, recall and F1 between the explanation and the human explanation, with pinned model/layer/version and explicit rescaling settings. Report case-level changes and aggregate distributions for initial, proposed and delivered final reasons. BERTScore uses contextual token embeddings; it is a similarity measure, not proof of correct entailment. [BERTScore](https://arxiv.org/abs/1904.09675).
2. **Human-reference reasoning coverage:** annotators break the reference into decisive propositions; score whether the answer preserves their participants, polarity, modality, comparison direction and figurative connection. Report supported proposition precision and reference coverage separately. Allow justified alternative explanations; flag ambiguous references for adjudication.
3. **Image faithfulness:** count unsupported visual claims independently of reference overlap. FaithScore offers an atomic visual-fact evaluation pattern, but its automatic verifier can also err and it does not replace a figurative reasoning rubric. [FaithScore](https://arxiv.org/abs/2311.01477).
4. **Decision quality and delivery:** final accuracy, helpful/harmful changes, correct-label-but-wrong-reason cases, audit false rejection/acceptance, incomplete outputs, missing evidence, abstentions and end-to-end latency.

Use blinded, shuffled A/B explanation assessment on the bounded comparison; ideally two annotators adjudicate disagreements. An independent automatic judge can assist, but must be calibrated to these human assessments. Do not let the production judge grade itself as the sole semantic metric. A short safe answer can omit the entire decisive figurative mechanism, so improved precision alone is insufficient.

## Adoption boundary

The immediate engineering fixes and E1/E2 are more directly tied to observed defects than a wholesale MCR/R-CoV rewrite. E3 and E4 are worthwhile controlled experiments. Defer fine-tuning, visual-token surgery, new multi-model ensembles and diffusion-style reasoning until evidence shows that simpler source, delivery and verification changes leave a problem those techniques specifically address.

No production files were changed and no new inference benchmark was run for this review. The attachment contains website titles and domains, with truncated citations and duplicates. All 83 entries are screened in the appendix; only entries explicitly marked as method-reviewed received that level of review. Unresolved or title-only entries are not treated as evidence for implementation claims.
