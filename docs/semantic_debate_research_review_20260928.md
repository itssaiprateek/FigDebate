# FigDebate: stability, semantic reasoning and debate review

Reviewed 28 September 2026. Status: research and proposed changes only.

## Recommendation

Repair the existing protocol before adding models, reviewers or verification stages. The strongest priorities are complete outputs, one consistent interpretation of the source claim, evidence-backed exchanges about specific disagreements, and a gate that distinguishes a bad inference from a failed verification call.

These are design recommendations supported by research and local failure traces. None is a demonstrated improvement on the current pipeline until a controlled evaluation passes. The stopped experiment remains stopped; its disabled switches should remain disabled.

## Evidence from this project

Repository inspected: C:\Users\Sai Prateek\Desktop\FigDebate_sent\FigDebate.

The two latest random diagnostics are training examples, not held-out estimates:

| Saved run | Initially correct | Finally correct | Helpful changes | Harmful changes |
|---|---:|---:|---:|---:|
| random10_20260928_131606 | 4/10 | 5/10 | 1 | 0 |
| random10_20260928_143456 | 3/10 | 2/10 | 0 | 1 |
| Combined | 7/20 | 7/20 | 1 | 1 |

Across these runs there were six correct-label change proposals, five rejected, and one harmful change proposal, accepted. These counts evaluate labels; they do not establish that every correct-label proposal had acceptable reasoning. Fourteen final explanations visibly end unfinished and preserve initial-arbiter output.

Local evidence: [first run](../outputs/random10_20260928_131606/tribunal_quality.json), [second run](../outputs/random10_20260928_143456/tribunal_quality.json), and their records.jsonl files, which include references and stage traces. Findings below compare those traces with the stored dataset explanations; this research run did not perform a new image annotation exercise.

| Observed failure | Where it starts | What the evidence supports |
|---|---|---|
| Doctor described as an angel is rejected because no literal angel appears | Initial interpretation, then compact proposal | The human reference connects attentive medical care with benevolence/protection. Literal object identity is the wrong test. |
| Child in wheelchair is treated as proof of a supportive wheelchair | Proposal and verification | Entity overlap replaces evaluation of the asserted property. The reference describes discomfort and dissatisfaction. |
| Sarcastic praise of delayed delivery becomes agreement with a complaint | Interpretation changes between stages | The verifier and challenger can evaluate different propositions while appearing to disagree over the label. |
| Happy reaction to cancelled plans has a correct contradiction proposal rejected | Relation verification/audit | A verifier can retain topic overlap and lose the opposite emotional state; later audit fields can disagree with the audit's own explanation. |
| Correct label justified only by missing evidence | Reasoning, even when the label is right | This proposal should remain rejected. Correct-label recall alone is an unsafe optimization target. |
| Email-explosion metaphor is correctly explained, but the experimental grounding call fails | Additional verification stage | Operational failure can block an otherwise useful correction. It does not establish that the interpretation was semantically invalid. |
| Incomplete initial explanations survive to final output | Initial generation and final-artifact selection | The baseline allows only 112 generated tokens and can reuse the assessment verbatim. Completion is a separate defect from interpretation. |

The paused work already tried longer/completion-aware assessments and an extra grounding obligation. Free-form results were mixed. The extra obligation blocked the harmful wheelchair change but also the helpful email change. A baseline replay contained an additional instruction, so that comparison is not a clean causal experiment. The latest compact initial schema has not received live qualification. See [paused checkpoint](in_progress_reasoning_20260928.md).

## What the research establishes

Sources were discovered through Hugging Face/public search and checked against original papers. The installed HF paper-search connector returned an error; the public HF search API and a full HF paper markdown download worked. This is a focused review, not an exhaustive systematic review.

| Paper and version | Method or finding | Adoption and limit for FigDebate |
|---|---|---|
| [V-FLUTE, NAACL 2025](https://aclanthology.org/2025.naacl-long.1/) ([methods](https://arxiv.org/html/2405.01474v2)) | Explainable visual entailment across metaphor, simile, idiom, sarcasm and humor. Annotation construction differs across source datasets. | Use the original task convention, not a generic factual entailment rubric. This is the closest evidence because it defines our task. |
| [Du et al., ICML 2024](https://proceedings.mlr.press/v235/du24e.html) | Independent initial answers followed by peer critique and revision improved the authors' reasoning/factuality tasks. | Preserve independent initial readings. Their results do not establish that additional rounds improve small VLMs on figurative entailment. |
| [Debate or Vote, 2025](https://arxiv.org/html/2508.17536v1) | Across seven benchmarks, independent voting explains much of the gain attributed to debate. A formal belief-update model supplies a conditional theoretical explanation. | Compare interaction against independent assessment at comparable cost. Neither its theory nor an oracle that knows correct answers supplies a production truth gate. |
| [When and Why Does Multi-Agent Debate Fail?, ColMAD v2, July 2026](https://arxiv.org/html/2510.20963v2) | Competitive persuasion and premature consensus can lose useful information. ColMAD asks agents to supply missing evidence and checks quotations; evaluates error detection and compute-matched baselines. | Borrow collaborative, claim-specific contributions. Do not copy automatic acceptance on agreement, self-reported confidence, or its larger-model results. Treat this version as a preprint. |
| [Why Do Multi-Agent LLM Systems Fail?, MAST v3](https://arxiv.org/html/2503.13657v3) | Classifies system-design, inter-agent and verification failures. Verifiers can perform superficial checks despite thorough-sounding instructions. | Trace the first defective stage and whether a later stage propagates or introduces error. More instructions or an extra verifier are not sufficient evidence of improvement. |
| [Stay Focused: Problem Drift in Multi-Agent Debate, Findings EACL 2026](https://aclanthology.org/2026.findings-eacl.268/) | Measures degradation during extended discussion. Targeted feedback/regeneration partially mitigates drift, with extra cost. | Keep the original claim and independent answers available; stop repetitive exchanges. Borrow the diagnosis, not additional DRIFT agents. |
| [Chain-of-Verification, Findings ACL 2024](https://arxiv.org/html/2309.11495v2) | Plans checks and answers them separately; factored verification hides the draft answer from answering prompts. The paper uses the same LLM and does not use external tools. | Rework existing witness checks to answer narrow questions from their sources. CoVe does not inherently provide an independent or external truth oracle. |
| [Reasoning Beyond Literal, January 2026](https://arxiv.org/html/2601.17197v1) | Distills image/caption/relationship/intent reasoning, then trains a Qwen2.5-VL-3B student using SFT and label/format rewards. | Supports separating observation from interpretation in explanations. Gains involve training on other figurative-style tasks, not a proven prompt-only fix for V-FLUTE. Label/format rewards do not certify every rationale step. |
| [MUG: Multi-agent Undercover Gaming, 2025](https://arxiv.org/html/2511.11182v1) | Uses edited counterfactual images in a social-deduction protocol to expose grounding failures. Also documents image-edit failures and misleading debate. | Borrow controlled, human-checked counterfactual evaluation offline. Do not add image generation, deceptive debaters or peer-elimination machinery to our runtime. |
| [PragMatch, August 2026 preprint](https://arxiv.org/html/2608.09772v1) | Contrasts pragmatic incongruity with ordinary mismatch and tests sensitivity to OCR/style cues. | Borrow paired semantic-change and meaning-preserving tests. Its task is sarcasm detection, not entailment; its construction has strong caption-only biases and excludes unparseable predictions, which we should not copy. |

Together, these papers support a testable hypothesis: useful debate exposes a missing premise or corrects a misinterpretation; simply generating more opinions may reinforce error. They do not prove that any specific redesign will improve FigDebate.

## Preserve the working foundations

Keep the models, dataset, OCR acquisition, source-span identity, evidence catalogue, stagewise GPU loading, checkpointing, bounded scheduler and disabled feedback setting for the first experiments.

Several proposed safeguards already exist:

- The compact proposer sees pixels, the complete source caption and observation catalogue; initial/gold labels are withheld.
- Verification calls already use fresh contexts. Fresh context reduces answer exposure, but repeated calls to the same weights are not independent experts.
- Decode-time constraints and evidence-ID validation already exist. They can enforce references and legal combinations, not truth.
- The current verifier already separates visual selection, participant mapping, relation assessment and argument challenge.
- Repair is already bounded and routes issues to capabilities. Preserve these controls.
- Common tribunal interpretation rules already caution against literalizing idioms and silently reversing sarcasm. The initial assessment's general instruction to compare with “intended meaning” is less precise, and existing rules are not reliably followed.

The recommendation is to make these components agree on their contract and reduce duplicate judgments. It is not a rewrite of source_spans.py or the whole runtime.

## Prioritized changes

### 1. Finish outputs and make failure states reliable

Owners: arbiter/arbiter.py, engine/reasoning_contract.py, models/judge_model.py, engine/final_artifact.py, engine/runtime_accounting.py.

Use a compact assessment with enough measured output capacity. Record actual termination cause, parsed-field coverage and whether the decisive justification completed. Do not repair truncation by appending punctuation or silently trimming the response.

Reuse the existing bounded retry for a delivery defect. A shorter complete rewrite from the same inputs is a delivery retry, not new semantic evidence. Never feed a cutoff assessment into decision scoring as though it were a complete argument.

Keep these distinct throughout the artifacts:

- execution failure: timeout, OOM, invalid JSON, incomplete output;
- semantic uncertainty: completed reasoning cannot establish the relation;
- rejected proposal: a completed review finds a material defect;
- accepted/confirmed conclusion: completed required checks support it.

If no complete justified result is available, expose that state. Where benchmark compatibility requires a binary prediction, preserve the existing forced/invalid metadata and score it separately from a verified result; do not invent a confident explanation.

Freeze model/tokenizer/processor revisions, prompts, schemas, quantization, image preprocessing, batch/order, budgets and library versions. Record a source-content fingerprint including dirty files, not only a Git commit. Reuse existing manifests and cache keys where covered; audit gaps before adding fields.

Same-machine repeatability and portability across GPU/software profiles are different tests. PyTorch does not guarantee identical results across releases or platforms. Specify a supported profile and measure output/label variation across it. [Official reproducibility guidance](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md).

This is the highest-confidence engineering fix. Better completeness is not itself evidence of better semantics.

### 2. Make all stages assess the same claim

Owners: engine/tribunal_interpretation.py, agents/claim_extraction.py, engine/claim_graph.py, arbiter/arbiter.py, engine/simple_judge.py, engine/evidence_review_v5.py.

Keep the full original caption immutable. Pass one versioned task convention to initial assessment, proposer and verifier. Preserve the asserted property, participants, polarity, modality, qualifiers and panel/time scope.

The V-FLUTE paper constructs sarcastic MuSE captions as contradictions, and non-sarcastic rewrites as entailments; New Yorker cartoon punchlines are treated as entailed through their joke. Thus “use intended meaning everywhere” can erase sarcasm contradictions, while literal object checking destroys metaphor. Apply the task convention to the actual claim, without routing from gold labels or gold phenomenon metadata. [Dataset construction](https://arxiv.org/html/2405.01474v2).

A useful compact proposer representation to test is:

| Field | Purpose |
|---|---|
| assertion | The complete contextual claim being evaluated, preserving material qualifiers. |
| reading | Short mechanism, depicted referent and asserted/transferred property; explicitly unresolved where necessary. |
| evidence_ids | Existing source-linked observations; not newly invented facts. |
| relation | SUPPORT, CONFLICT or UNRESOLVED. |
| reason | A complete, brief justification connecting those observations to that property. |

The immutable source, source spans, version and runtime status are attached by software. Do not ask the model to recopy them. Retain existing per-condition checks where a claim has multiple material conditions; do not flatten every compound sentence into one object match.

This representation replaces duplicated narrative fields rather than adding another mandatory call. The stopped compact schema is a candidate starting point, not an approved implementation. Extra fields can consume budget without improving inference, so measure this explicitly.

For the doctor example, the tested property is care/protection. The doctor need not possess wings. For the wheelchair example, the presence of a chair does not establish comfort/support. Both follow the same rule: identify the property, then test the property.

### 3. Make exchanges resolve one concrete disagreement

Owners: engine/tribunal_repair.py, engine/review_routing.py, existing hearing/witness interfaces.

Preserve independent initial outputs. Do not assign agents an obligation to defend ENTAILS or CONTRADICTS, and do not use consensus as the stopping success criterion.

Replace any remaining generic requests to reconsider with a dispute record: disputed claim, type of defect, one answerable question, available source and the premise that could change. Reuse existing routing.

Examples of appropriate questions:

| Dispute | Question and source |
|---|---|
| Visual property | What posture and contact with the chair are visible? Inspect pixels; do not ask the visual agent whether the whole caption is true. |
| OCR polarity | What does the relevant line say, and to which panel/speaker does it belong? Use the image/crop and surrounding text. |
| Metaphorical mapping | What property does the expression attribute to the depicted participant? The interpreting step needs both caption and observations. |
| Temporal comparison | Which reaction belongs before and after the event? Keep panel order and negation. |
| Audit inconsistency | Which specific inference is invalid, and why does the stated evidence invalidate it? |

A caption-only language witness can explain conventional senses and required conditions; it cannot certify which visual referent is correct. Conversely, a visual witness can report posture or text but cannot turn the caption's characterization into an observation.

Answer factual subquestions without showing the preferred answer or persuasive draft. This adapts CoVe's separation within our existing calls; source access is essential. Interpretive questions may need the hypothesis, explicitly marked as a hypothesis.

Allow at most the existing targeted follow-up allowance. Continue only for a missing obtainable fact, a distinct grounded interpretation, or an internal audit contradiction that can be reconciled. Stop when evidence and interpretation do not materially change. A paraphrase or repeated opinion is not progress.

### 4. Repair the semantic gate instead of making it uniformly stricter

Owners: engine/evidence_review_v5.py, engine/tribunal_process.py, engine/tribunal.py, engine/semantic_bridge.py.

Use literal grounding, figurative interpretation and relation assessment as connected responsibilities, not three votes or three new model calls:

| Check | Pass condition | Insufficient condition |
|---|---|---|
| Literal foundation | Deciding visual/OCR claims are supported with correct attachments. | A valid ID or a rectangle alone does not establish its description. |
| Interpretation | A context-supported mapping preserves the source claim and relevant roles/qualifiers. | A conventional metaphor alone does not prove the depicted subject has the property. |
| Relation | Evidence supports the material claim, or establishes a positive incompatibility under the same reading. | Matching the topic is not support; missing evidence alone is not contradiction. |
| Audit | Any blocking objection names a material defective inference and its basis. | Generic concern, an already resolved objection, or agreement with a different paraphrase is not a coherent veto. |

Keep the independently formed relation assessment before exposing the candidate argument. Then compare the candidate and verifier explicitly: same source, same target property, same roles and same reading convention. A disagreement about the reading needs examination, not automatic rejection or forced agreement.

Replace parallel free-text error lists plus loosely related statuses with one authoritative objection structure: error type, disputed step/claim, supporting counterevidence or missing required condition, and consequence for the relation. Generate a competing reading only when grounded and capable of changing the answer.

Derive redundant status/label fields in software from the canonical records. Validate finite consistency and provenance. These checks cannot certify a natural-language inference: an unsupported or contradictory audit remains invalid, not a reason to accept automatically. Reconcile it once within budget, then withhold verification if unresolved.

An operationally failed check must not masquerade as a substantive objection. It still prevents certification until successfully completed; preserving a prior result must retain that result's actual verification status.

The new reading fields must be available where the proposal is formed. Currently the added grounding obligation runs only after a binary-eligible compact proposal, so it cannot rescue a metaphor already abandoned as UNRESOLVED.

Retain frozen candidates for audit-only repair. If new evidence or interpretation changes the candidate, issue a new version and invalidate dependent proofs. Keep existing hash/source bindings.

## Evaluate whether these changes deserve promotion

The immediate experiment is a sequence of small, controlled ablations, with feedback disabled:

| Arm | Only intended change | Question |
|---|---|---|
| B0 | Reproduce the actual pre-experiment baseline from saved source/config | Is the comparison valid? |
| B1 | Completion and error-state handling | Do cutoff/missing results fall without semantic regression? |
| B2 | B1 plus shared claim/reading contract | Do agents preserve properties, scope and sarcasm convention? |
| B3 | B2 plus revised audit/objection contract | Are valid rationales rejected less often without more harmful acceptances? |
| B4 | B3 plus focused dispute exchange | Does interaction add useful information at acceptable cost? |
| Control | Independent compact assessment, no repair exchange | Does debate beat the simpler use of the existing model? |

For causal isolation, replay identical frozen upstream evidence first. This cannot establish full-pipeline gains; follow with bounded end-to-end checks, then a locked held-out evaluation. Do not combine all changes and attribute improvement to one component. Match budgets for semantic comparisons and report the additional budget needed by completion separately.

Use the known cases for development and mechanism regressions only. Freeze a representative validation cohort across phenomena and labels before tuning. Keep related captions/images grouped; audit official-split overlap rather than silently changing the published split. Maintain the official test set untouched, and report any supplemental group-disjoint protocol separately.

The final comparison should use identical case manifests, source snapshots, model revisions, seeds and decoding settings. Separate cold/warm runtime, randomize arm order, and repeat a small stability subset across restarts, resume and batch/order changes. Do not rerun until the desired answer appears. Full dataset runs remain user-controlled.

Measure:

- Operational: completion, truncation, parse/timeout/OOM rates, missing fields, retry count and terminal status.
- Semantic: initial/final accuracy and macro-F1; correct-to-wrong and wrong-to-correct changes; performance by phenomenon.
- Gate: acceptance precision for label-and-reasoning-valid proposals; false acceptance and false rejection against adjudicated candidate quality; abstention/verification coverage.
- Explanation: image faithfulness, preservation of claim/roles, validity of the figurative connection, completeness and consistency with the chosen label.
- Efficiency/stability: median and tail latency, generated tokens, memory, calls per case, and repeated-run agreement.

Report denominators, paired differences and uncertainty. Ten selected cases cannot establish a stable population improvement. A higher acceptance count alone is not success.

Use the existing BERTScore evaluator after inference, alongside human adjudication. Reference similarity is diagnostic: a paraphrase with a negation error can still score well. V-FLUTE additionally reports an explanation score combining BERTScore and BLEURT; adopting that for paper comparability would require matching its scoring setup, not relabeling our current BERTScore. [Evaluation method](https://arxiv.org/html/2405.01474v2), [current evaluator documentation](REASONING_SIMILARITY.md).

For human review, first inspect the image/caption and anonymized candidate, then compare the dataset reference. Judge the decisive premise and inference rather than wording. Record disputed references separately; do not silently edit gold labels. References never enter inference, debate, caching decisions or the acceptance gate.

Add a small offline contrast set: human-checked paraphrases should preserve decisions; changing a decisive polarity, role or panel relation should change the expected result when appropriate. Prefer existing paired captions and reversible text edits. Image ablation is a diagnostic, not a requirement that every individual prediction change. Do not automatically treat removal of decisive evidence as contradiction.

Promotion requires operational qualification, improved joint label-and-explanation quality, an explicitly bounded harmful-change risk and acceptable runtime on the frozen cohort. Report confidence intervals and predeclare tolerances; a small zero-harm sample does not prove safety. If evidence is inconclusive, retain the baseline.

## Defer or reject for this round

Do not add more judge agents, mandatory regional checks for every fact, online image editing, long debates, additional memories, a larger model, or a new dataset as the initial remedy. Do not turn an identified figurative type into an automatic label. Do not use a model's confidence, unanimous agreement, grammar validity or explanation similarity as a truth certificate.

If clean contracts and grounded exchanges still consistently fail, run an isolated capability assessment of the existing models. Fine-tuning could then be a separate evidence-driven project. The cross-style training paper is a precedent, not proof that longer prompts can substitute for training.

Implementation order: qualify completion and state handling; align claim/reading semantics; simplify the audit; then test whether focused interaction earns its cost. Reuse working components and promote each change only on its own evidence.
