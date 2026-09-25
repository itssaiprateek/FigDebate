# Prompt research review — 24 September 2026

Research and design proposal only. No production prompts, models, schemas or gate policies were changed, and no inference was run for this review.

## Assessment

Prompt/interface problems are a credible contributor to the observed errors, but the current traces do not establish what fraction comes from prompt design versus model capability. The strongest next experiment is to make the question and audit target unambiguous, simplify duplicated semantic decisions, and demonstrate the intended task. Adding instructions alone has already failed a paired control.

The completed rerun `worst10_retest_20260924_102021` retained four correct-label and two harmful change proposals, but accepted none. Its first relation output on case 392 matched the original run; the different recovery path explains that lost acceptance. This is evidence about pipeline behavior, not proof that a new wording will understand the image correctly.

## Observed prompt/interface problems

| Location | Observation | Consequence / qualification |
|---|---|---|
| `engine/evidence_review_v5.py`, challenge payload around lines 433–442 | The payload includes the independent checker's `decision`, the proposer's interpreted assertion and draft argument, but no explicit proposer relation. | When proposer and checker disagree, the audit target can be ambiguous. Case 3804 is consistent with this: a CONFLICT check was endorsed alongside an ENTAILS proposal. The missing field is established; its causal effect needs an isolated comparison. |
| `engine/evidence_review_v5.py`, `CHALLENGE` | Alternative text, alternative relation, alternative status, errors and explanation are independently generated. | Multiple fields encode overlapping decisions. Cases 3858 and 2152 explicitly supplied an alternative with NONE status. |
| `engine/evidence_review_v5.py`, relation contract | Necessary conditions appear in both condition checks and a separate unresolved list. | A condition can be asserted and unresolved simultaneously, as in 1466 and 3804. A validator detects this after generation; it does not prevent the model's duplicated bookkeeping. |
| `engine/evidence_review_v5.py`, `CHALLENGE_INSTRUCTIONS` | It already says to name candidate errors and counterevidence, preserve the source, and avoid objections that its own reason resolves. | Repeating the same instruction more emphatically is not a new mechanism. Case 537 still produced `["CONFLICT"]` as an error while its reason endorsed that conclusion. |
| `engine/evidence_verification.py`, `obligation_runner` | Stage instructions are followed by a substantial shared interpretation block. Explanations are preferably at most 16 words. | Instruction density and brevity may compete with scope preservation. This is a hypothesis; not evidence of a context overflow. |
| `engine/simple_judge.py`, `prompt` | The proposer must give both a contextual interpretation and decisive comparison, with each text field limited in the prompt to one sentence of at most 20 words. | Compactness helps cost, but a literal scene summary can replace the needed semantic link. Test field responsibilities rather than assume longer output is better. |
| Current independent verification | Visual selection and relation checking already omit the proposer draft; final challenge sees both draft and checker output. | Preserve the existing blindness. Reordering output fields cannot hide input already present in the same call. |
| Bounded repair | The targeted check includes prior criticism and may freeze the candidate. | Repetition does not add evidence. A disagreement should be resolved against sources, not by treating the earlier critique as established truth. |

## Research reviewed and project fit

Hugging Face paper pages were used for discovery; methods and limits were checked against primary papers. This review includes the relevant previously supplied DynaDebate, MARCH and debate-failure sources and additional work specifically on prompting. It does not claim to have reread every unrelated title in the earlier reference panes.

| Source | What it actually supports | FigDebate adaptation and limit |
|---|---|---|
| [V-FLUTE](https://aclanthology.org/2025.naacl-long.1/), [full text, Appendix H](https://arxiv.org/html/2405.01474v2), [Hugging Face](https://huggingface.co/papers/2405.01474) | On this task, multimodal few-shot examples and scene-graph prompting can help. Scene-graph explanations can overfocus on objects and lose figurative meaning. | Test a small, fixed set of task demonstrations and an explicit evidence-to-meaning link. Do not merely add a larger scene graph. Five-shot API-model gains do not predict gains for Qwen3.5-4B; text-only demonstrations would be a further adaptation. |
| [Chain-of-Verification](https://arxiv.org/html/2309.11495v2), [Hugging Face](https://huggingface.co/papers/2309.11495) | Verification answers can improve when generated without the original draft. Open factual questions outperformed leading yes/no questions in its tested setting. | Ask witnesses for source facts without supplying the desired answer. Reuse the independent relation stage. This work studies factual generation, not a proven cure for figurative entailment. |
| [MARCH](https://arxiv.org/html/2603.24579v1) | The checker sees documents and questions, without the solver response or proposed answers. The system is trained with multi-agent reinforcement learning. | Preserve information separation. Do not attribute its trained results to a copied prompt. The eventual comparator must explicitly see both judgments to resolve disagreement. |
| [Woodpecker](https://arxiv.org/html/2310.16045v2), [Hugging Face](https://huggingface.co/papers/2310.16045) | Breaks hallucination correction into concept extraction, questions, external visual validation and correction. Uses object detection and VQA models. | Route observable questions to the visual witness; keep interpretive judgments distinguishable from pixel observations. Its grounding results do not establish metaphor comprehension. |
| [DynaDebate](https://arxiv.org/html/2601.05746v2) | Audits identifiable inference steps, alongside diverse reasoning paths and triggered external-tool verification. | Audit the source interpretation and decisive evidence-to-relation transition. A compact audit is an adaptation, not the complete method or a guarantee of truth. |
| [Chain of NLI](https://arxiv.org/html/2310.03951v2) | Combines sentence-level with finer entity-level checks; uses domain-agnostic NLI demonstrations. | Examine complete propositions plus deciding subject, scope and qualifiers. Do not transfer a factual-grounding definition that collapses unsupported claims into hallucination directly into the dataset's CONTRADICTS label. Missing support remains distinct from positive incompatibility. |
| [Contrastive Chain-of-Thought](https://arxiv.org/html/2311.09277v1), [Hugging Face](https://huggingface.co/papers/2311.09277) | Demonstrates valid and invalid reasoning rather than only correct solutions. | Try very short contrasting examples of scope preservation, missing evidence and candidate audit errors. Its reasoning benchmarks are not direct evidence for our vision model. |
| [IFScale](https://arxiv.org/html/2507.11538v1) | Instruction adherence declines as instruction density increases in a keyword-inclusion task. | Supports testing instruction consolidation. It does not prove that our prompt length caused a particular semantic error, nor establish a universal ideal number of rules. |
| [Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798), [Intrinsic Self-Correction Ability](https://arxiv.org/html/2406.15673v1) | Results depend on model, feedback and experimental setup; the latter studies unbiased prompts and decoding. | Avoid assuming either that self-review always works or never works. Test neutral, source-based reconsideration with preserved correct controls. |
| [Talk Isn't Always Cheap](https://arxiv.org/html/2509.05396v1) | Debate can degrade performance in tested settings, including smaller model families. | More agents, rounds or agreement are not sufficient evidence of better verification. Retain a fixed call/time budget in comparisons. |

Some V-FLUTE appendix prompts generate dataset explanations while receiving the known label. Those are annotation prompts and must not be copied into label-blind inference. Reference explanations remain evaluation-only for the tested case.

## Recommended prompt structure

This is an engineering proposal informed by the papers, not a reproduced published pipeline.

Every call should have: (1) one stable task and decision definition, (2) one role-specific instruction, (3) a clearly delimited source-data object, (4) a minimal output contract. Demonstrations, if used, belong in a fixed example block clearly separated from the current case. Preserve the complete source expression; shorten repeated instructions and unnecessary generated prose instead.

### Compact proposer

Inputs: image, exact caption, observation catalogue with provenance. No initial arbiter label, gold label or reference explanation.

Output responsibilities:

- `interpreted_assertion`: what the caption asserts in this context, retaining speaker, negation, modality and comparisons. It must not become an image description or a convenient replacement claim.
- `evidence_ids`: observations needed for the decision.
- `decisive_reason`: the observable fact plus the context-supported semantic link, distinguishing an inference from a pixel fact.
- `relation`: SUPPORT, CONFLICT or UNRESOLVED under the unchanged task definition.

Keep this four-field shape initially. Test a clarity rewrite before expanding the schema or allocating a longer generation.

### Independent evidence and relation checks

The visual question asks for observable facts: actors, actions, text, attachment and ordering. The interpretation/relation check uses the full image-caption context to explain the relevant figurative mapping. A visible object need not literally instantiate the caption's idiom; the proposed correspondence still needs support.

For source conditions, test a single authoritative table: each complete condition has one status, evidence references and a short justification. The unresolved list is derived by code rather than independently regenerated. Preserve conjunction, alternatives, negation and modality; atomic fragments must not erase their enclosing expression. A full-claim relation remains a semantic judgment, not a naive keyword or majority calculation over rows.

An unknown condition blocks acceptance only when it is necessary to the asserted relation. The model must explain that dependency. Do not waive uncertainty solely because it concerns emotion or abstract meaning, and do not demand impossible visual proof of every inferred property.

### Final comparator / auditor

Give this call explicitly separated objects:

```json
{
  "source_caption": "<unchanged original caption>",
  "candidate": {
    "relation": "<proposer relation>",
    "interpreted_assertion": "<proposer assertion>",
    "decisive_reason": "<proposer reason>",
    "evidence_ids": ["<source IDs>"]
  },
  "independent_assessment": {
    "relation": "<independent relation>",
    "conditions": ["<source-bound assessments>"]
  }
}
```

Also retain the image and cited source observations. This outline omits their contents only for readability.

Suggested task wording for an experiment:

> Evaluate the candidate's proposed relation to the unchanged source caption. The candidate and independent assessment may each be wrong. Identify whether a material error occurs in the candidate's interpretation, cited observation, or inference to its relation. For an error, identify the specific candidate claim and the evidence or missing premise that invalidates it. A contradiction between the image and caption can support a CONFLICT candidate; it is not itself an error in that candidate. If no material error is established, report no issue. If the decisive question cannot be resolved, report uncertainty and the missing dependency. Agreement alone is not verification.

A possible smaller output is a status plus `material_issue: null | {claim, issue_type, evidence_ids, reason}`. A competing interpretation, if needed, should similarly be one optional object rather than a free-text placeholder combined with several independently generated flags. Source-identity, citation and raw-response checks remain mandatory. A valid shape does not make the judgment correct.

### Bounded reconsideration

Pass the disputed inference, original sources and any genuinely new witness evidence. Do not prepend an assertion that the previous answer is wrong. Ask one targeted question that could distinguish the competing conclusions. Record whether new evidence or a corrected inference was obtained; repeated unsupported criticism is not progress.

Changing which answer the gate accepts when proposer and verifier disagree would be a separate policy experiment. Merely exposing both labels must not automatically promote the verifier's answer or undo the candidate-freezing safeguard.

## Demonstrations and generalization

Use a small fixed bank of independently reviewed examples from a separate training cohort, excluding all current diagnostic cases and image/near-duplicate overlap. Include correct SUPPORT, correct CONFLICT, unresolved evidence and erroneous critiques across the project's phenomena. Do not select examples by nearest match to a known failed test or its reference explanation.

An abstract demonstration can show that an open gate supports CONFLICT with a caption asserting a closed gate; the source-caption contradiction is not a candidate defect. A paired incorrect SUPPORT candidate illustrates the actual audit error. A separate figurative example should show an evidenced contextual correspondence rather than literal phrase matching. Synthetic examples test instruction comprehension; they do not replace live image examples or establish dataset accuracy.

## Isolated test order

1. Explicitly name both verdicts and the audit target, keeping the existing schema and call budget.
2. Test a concise shared task contract against the existing instruction block; keep actual source content intact.
3. Test a fixed demonstration bank separately, including valid no-error audits and real errors.
4. Test the single condition-status table and smaller issue object, with validators and reporting changed together in an isolated variant.
5. Only combine variants that pass paired controls. Run fresh cases drawn from each target phenomenon and use the user's broader evaluation afterward.

For each comparison hold model, image resolution, decoding, sample order and call limits fixed. Measure correct-proposal recall, harmful acceptance, explanation adequacy, false objections, unresolved decisions, schema failures and wall time separately. Include paraphrased prompts and renamed/reordered evidence IDs as invariance checks. Review explanation quality after prediction without exposing the reference to inference. A pass on selected controls is qualification for broader testing, not a guarantee of universal improvement.

The first priorities are explicit audit targets and task demonstrations. Blind checks already exist and should be retained. Another generic agent, a longer warning block or a more elaborate process-audit matrix is not the preferred next step given the previous regression.
