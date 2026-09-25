# Research-backed change list for the three critical FigDebate failures

Date: 7 September 2026. Status: **proposal, not implemented or qualified**.

## Recommendation

Use a tested structured-generation backend, a typed claim-and-requirement representation, and factored source-grounded verification. Do not treat another self-approval prompt, larger output budget, or weaker acceptance threshold as a solution.

The research supports these methods in other settings. It does not prove that their combination will solve FigDebate, that one backend is universally best, or that a 4B/7B quantized model will match results obtained using much larger models. The acceptance tests below are therefore part of the proposed solution, not optional follow-up work.

The three target problems are: (1) incorrect Agent 2 requirements, (2) unreliable semantic auditing, and (3) unusable judge outputs. The decoder is addressed first because it is shared infrastructure and a confirmed low-level defect.

## 1. Source-to-method mapping

| Source | What it establishes or documents | Proposed use | Important limitation |
|---|---|---|---|
| [XGrammar](https://huggingface.co/papers/2411.15100) | A structured-generation engine with efficient grammar/token handling. | First backend candidate for replacing the affected LMFE adapter. | Does not prove our exact tokenizer/schema/model combination works. |
| [JSONSchemaBench](https://huggingface.co/papers/2501.10868) | Evaluate constraint coverage, efficiency and output quality separately. | Backend conformance and quality qualification, not selection by speed alone. | Valid JSON still need not express correct reasoning. |
| [Davidsonian Scene Graph / DSG](https://huggingface.co/papers/2310.18235) | Atomic semantic questions with dependency structure address coverage and consistency in text-to-image evaluation. | A figurative-aware claim/requirement graph with explicit entity, property and scope links. | Its original task is text-to-image evaluation; our figurative entailment extension is a local adaptation. |
| [Decomposed Prompting](https://huggingface.co/papers/2210.02406) | Modular subtasks can be easier to specify and replace than a combined task. | Separate expressed claim, interpretation and evidence-condition construction. | Decomposition does not supply missing knowledge or guarantee correctness. |
| [Chain-of-Verification / CoVe](https://huggingface.co/papers/2309.11495) | Factored verification reduces copying of earlier errors; open verification questions outperform yes/no questions in its reported comparison. | Replace six simultaneous approval Booleans with independent source questions and explicit comparison. | Mainly factual text tasks, using a much larger base model; local benefit must be measured. |
| [Large Language Models Cannot Self-Correct Reasoning Yet](https://huggingface.co/papers/2310.01798) | Documents failures of unaided intrinsic correction in its tested reasoning settings. | Do not give an unqualified self-audit sole authority. | Not a universal impossibility theorem. |
| [Large Language Models have Intrinsic Self-Correction Ability](https://huggingface.co/papers/2406.15673) | Reports dependence on prompting and decoding conditions. | Include neutral-prompt and ordering controls; avoid assuming every draft is wrong. | Does not overturn our observed failures or demonstrate a working FigDebate auditor. |
| [CheckList](https://huggingface.co/papers/2005.04118) | Behavioural tests expose failures hidden by aggregate accuracy. | Minimal pairs, invariance checks and directional counterexamples. | Synthetic tests complement, not replace, fresh real-example evaluation. |

### Established engineering components

[Pydantic strict validation](https://docs.pydantic.dev/latest/concepts/strict_mode/) supports rejecting wrong input types rather than silently coercing them. It must be combined with our own semantic-content invariants; strict strings alone still admit the text `None`.

[XGrammar's current installation documentation](https://xgrammar.mlc.ai/docs/latest/start/installation.html) lists Windows and Python 3.9+ support, and its [quick start](https://xgrammar.mlc.ai/docs/latest/start/quick_start.html) documents a Hugging Face Transformers logits-processor integration. This makes it a good first candidate for the existing local stack, not a pre-qualified drop-in replacement for Transformers 5.15.0 and quantized Qwen3.5.

[llguidance](https://github.com/guidance-ai/llguidance) is a credible comparison/fallback candidate. A compatible upstream-fixed LMFE release could also compete if it passes the same reproduced defect tests. We have not established that such an LMFE release fixes our issue. Do not claim that any candidate is already installed, benchmarked or proven superior locally.

## 2. Intended single-system design

```text
Original caption
  -> typed source-anchored claim atoms
  -> explicit interpretation hypotheses and mappings
  -> dependency-aware evidence requirements
  -> independent source questions and checked answers
  -> image witness observations bound to the same requirements
  -> structured tribunal resolution
  -> existing independent image verification and revision gate
  -> explanation rendered from the accepted resolution and its evidence
```

The baseline and tribunal must share the repaired common input and decoder infrastructure. Otherwise a future gain could be caused by repairing the baseline only in the tribunal condition, rather than by the tribunal itself.

## 3. Ordered changes

Each numbered item is proposed. Items 1–6 should qualify before attributing further model behaviour to prompts. Items 7–18 form one connected representation and verification change, not disconnected helper scripts.

### Phase A — remove the judge's demonstrated decoding failure

**C01. Freeze the current failing reference.**

Preserve run records, source hashes, model revisions and the existing critical probes. Store future candidate results in separate directories. Keep the original failures intact. Targets: diagnostic utilities, run integrity and experiment configuration.

Acceptance: a comparison can identify exactly which decoder, prompt, schema or model setting changed; nothing silently overwrites the old run.

**C02. Build a decoder conformance gate before choosing the replacement.**

Promote the blocked sentence-ending token into a regression case. Add legal and illegal transitions involving quotes, commas, punctuation, escapes, Unicode, whitespace, nested structures, EOS, field limits and the actual full schemas. Check both Mistral and Qwen tokenizers. Follow JSONSchemaBench's separation of correctness, coverage, quality and speed.

Acceptance: all deterministic required-format cases pass, and known illegal sequences remain blocked. There is no finite-test claim of universal correctness.

**C03. Qualify XGrammar as the preferred decoder candidate.**

Use the maintained Transformers integration rather than constructing another ad hoc token tree. First test it in isolation against the pinned local environment, vocabulary sizes, quantization and stopping behaviour. Compare a second backend if it fails requirements. Pin the successfully qualified release and record its license.

Targets: `engine/structured_decoder.py`, dependency configuration and decoder tests. Do not install or promote an arbitrary latest version solely because documentation lists platform support.

**C04. Apply the chosen backend consistently to every shared path.**

Cover Agent 2 extraction, auditing, caption witnesses, text candidates, supervisor review and independent verification—not just the judge. Preserve the same raw prompts and seeds for the decoder-only qualification first.

Acceptance: no hidden legacy path uses the defective token filtering, and decoder identity is part of cache/run fingerprints. Never silently fall back to unconstrained output and label it valid.

**C05. Make the judge's resolution a compact, internally consistent record.**

Represent the selected relation once and derive redundant binary labels in software, or use a discriminated schema that forbids contradictory combinations. Use claim-node IDs, observation IDs, selected interpretation, decisive relation and rejected alternative. Keep optional explanatory prose separate from the authoritative decision record.

Do not use a character ceiling as a grammar checker. Any overflow or incomplete required field remains an explicit failure; do not chop text and certify it. A human-readable explanation should be rendered from verified fields, not generated as a new unconstrained verdict.

Targets: `engine/output_contracts.py`, judge parser, tribunal resolution and final-artifact rendering. This is a local engineering adaptation; no source establishes that formatting alone improves accuracy.

**C06. Make repair targeted and fully replayable.**

Save each attempt's raw text, effective prompt/hash, schema, token counts, stop reason and exact failing field. Repair from original case data and the identified defect. Do not ask a model to preserve the meaning of an already cut-off sentence without its source context. Revalidate all required dependencies after repair and stop after the configured bounded attempts.

Acceptance: first attempt versus repair can be compared exactly; failure remains failure rather than becoming a semantic abstention or accepted normalization.

### Phase B — repair Agent 2's representation and requirement construction

**C07. Introduce one canonical typed data contract.**

Use strict typed objects for the claim, interpretation hypotheses, requirements and verification results. Pass these objects between components; text prompts are a presentation layer, never the authoritative store. Distinguish empty list, null, unavailable value and literal text. Reject unexpected fields and wrong types.

Targets: claim schemas, `engine/debate.py`, `engine/claim_witness.py`, dossier construction and checkpoint schemas. Acceptance: empty-list round trips preserve `[]`; no conversion through headings and `ast.literal_eval` is needed.

**C08. Anchor expressed claim components to the raw caption.**

Record source spans for entities, relations, negation, quantities and modifiers. Validate offsets and exact span contents. Preserve the full caption unchanged. Source spans establish what text was used, not whether an interpretation is correct.

Acceptance: swapped roles, lost qualifiers and altered numeric values are detectable; no guessed source span is treated as factual provenance.

**C09. Replace the single vague requirement pair with atomic relational records.**

Create distinct nodes for the entities and their properties, with clear subject/object bindings and qualifiers. Use DSG-style semantic coverage and dependencies, adapted to this task. For the products caption, disliked-product duration and loved-product duration remain separate, bound assertions.

Targets: claim extraction, comparator inputs, ledger links and hearing construction. Acceptance: correct OCR cannot be accepted as matching the opposite assignment merely because all the same words occur.

**C10. Link figurative interpretation to those claim nodes explicitly.**

Separate raw wording, expressed proposition and interpretation hypothesis. An idiom such as “heart sank” can map to emotional distress without requiring a physically descending heart. Sarcastic intended polarity must not automatically replace the expressed proposition being evaluated. Record ambiguity rather than force a figurative category to choose a label.

Acceptance: changing a reading invalidates its dependent requirements and proof; no stale literal requirement survives unnoticed after a figurative interpretation is selected. This is part of FigDebate's research-specific adaptation, not a verbatim DSG implementation.

**C11. Preserve logical scope when generating support and conflict conditions.**

Build requirements around the actual claim structure. For a genuine conjunction, supporting the whole claim requires its relevant parts, while disproving one necessary part may suffice for conflict; the opposite need not reverse every clause. Disjunctions, quantifiers and uncertain readings require their own checked handling. Unsupported constructions remain unresolved rather than being coerced into a simplistic rule.

Distinguish “not observed” from “observed to be false.” Keep “recovered” separate from “recovered relatively easily.” Do not turn difficulty, speed and outcome into interchangeable properties.

Acceptance: positive, negative, missing, partial and mixed-scope fixtures produce the intended proof obligations. Basic logic is an engineering requirement; its application to a natural-language parse still depends on a correct parse.

**C12. Generate witness questions from the checked requirements.**

Ask what visual observation would resolve an outstanding node, rather than requesting an unrestricted support/conflict essay. Caption-only checks must not claim to inspect the image. Keep conceptual dependencies conditional on the interpretation: an idiom's emotional reading must not inherit a requirement for an anatomical heart.

Acceptance: questions address the products' depicted durations, the caption's ease qualifier and the dress-code offer itself. No source-record ID or gold label is used to select the desired answer.

### Phase C — replace unqualified self-approval with factored verification

**C13. Replace six joint approval Booleans with open verification questions.**

Adapt CoVe's factored execution. Ask questions such as “Which condition is attributed to the meeting?” or “What phrase specifies the ease of recovery?” rather than “Is this extraction correct?” Cover each relevant claim obligation; do not add six calls when fewer source questions suffice.

Question generation itself must be checked for relevance and coverage. CoVe does not prove fixed templates always outperform generated questions; compare the task-specific options rather than assuming they do.

**C14. Answer verification questions without the proposal's answer or audit verdict.**

The answer-producing context should contain the raw caption and neutral question, not the generator's preferred requirement, endorsement, previous confidence or other verification answers. Return short source-backed answers or an explicit unknown. A later comparison step may see both independent answers and the proposed representation.

Acceptance: a deliberately wrong prior proposal cannot simply be copied into the verification answer. Independence means controlled information exposure, not guaranteed independent model errors.

**C15. Compare independent answers with the proposed conditions explicitly.**

Use deterministic checks for exact spans, identities, types, numeric values, duplicate states and supported operators. For genuine semantic comparison, record an evidence-backed PASS, FAIL or UNKNOWN per obligation; no unsupported Boolean vector certifies the entire claim. Model-based semantic comparison remains fallible and must be separately evaluated.

Targets: `engine/claim_semantics.py`, claim contract and witness gates. Acceptance: the four faculty-meeting diagnostic variants no longer receive the same assessment, and the contradictory reason/Boolean combination cannot be silently accepted.

**C16. Enforce minimum content invariants before consulting any model.**

Reject placeholder-only requirements, empty required content, unresolved IDs, invalid source spans and impossible self-opposition in the relevant typed fields. A literal word “None” inside a quoted caption is not globally banned; only inappropriate placeholder use in required evidence fields is rejected. Preserve explicit unknown statuses instead of inventing a condition.

Acceptance: injecting an always-approving auditor cannot make the existing `"None"`/`"None"` fixture pass. A strict string type alone is insufficient.

**C17. Repair only demonstrated defects with neutral instructions.**

Do not imply that every prior answer must be wrong. Supply the source, specific failed obligation and independent evidence; preserve validated nodes. Recheck changed nodes and downstream dependencies. Record an unresolved outcome if repair fails.

This incorporates the mixed self-correction literature: neither “self-correction never works” nor “another prompt will fix it” is justified. Existing deterministic decoding should not be presented as a new solution when it is already in use.

**C18. Qualify the auditor role before assigning it authority.**

Start with the existing models. Compare the repaired factored Mistral verifier and, if necessary, a separately scheduled text-only verifier using the already available other model family. Blind both to final labels. Promote neither by name or size: use adjudicated development checks and false-approval/false-rejection measurements. A second family is a comparison candidate, not proof of independence.

If no existing model meets the preregistered criterion, keep the role unqualified and investigate a trained specialist separately. Do not promote generic text NLI to image-grounded proof or launch new training without a scoped plan.

### Phase D — demonstrate that the fixes work together

**C19. Build a small adjudicated semantic qualification set.**

Create separate development fixtures for roles, negation, quantities, degree, scope, idioms, sarcasm and multi-entity comparisons. Include valid, invalid and genuinely ambiguous examples. Use independent review of expected outcomes; do not let the generator label its own qualification set. Preserve the underlying three-category dataset and keep test labels out of tuning.

**C20. Add CheckList-style behavioural tests.**

Use harmless paraphrases that should preserve outcomes, role/property swaps that should change them, deleted qualifiers that should trigger incompleteness, and minimal positive/negative pairs. Some counterfactual edits do not determine a new gold label; mark those as representation tests or have them adjudicated rather than assigning an automatic flip.

Acceptance: no special-case fix for “heart sank” or the faculty meeting; unseen constructions must be represented in qualification. Convert known-defect expected failures to passing tests only after the relevant implementation changes.

**C21. Run staged, paired integration experiments.**

Evaluate the decoder-only change first, then typed/graph changes, then factored auditing, then the complete system. Preserve intermediate results and costs so one component's benefit is not attributed to another. Use saved cases as regression examples and fresh internal development groups for generalization.

For tribunal-versus-no-tribunal comparisons, both conditions receive the same repaired shared components. Existing cached stages are reused only when their input/dependency fingerprints genuinely match.

**C22. Freeze release criteria before viewing qualification results.**

Require all specified deterministic invariants to pass. For semantic tests, report false approval, false rejection, unknown rate, per-phenomenon results and uncertainty intervals. Select numerical semantic thresholds from a declared research risk/utility requirement, not a convenient observed result. A finite suite with no observed failures is not an all-input guarantee.

Keep schema validity, semantic validity, abstention and runtime error separate. Do not tune acceptance thresholds to achieve a desired acceptance count.

**C23. Make final explanations consume the accepted evidence structure.**

Render the selected reading, decisive claim nodes, image observations, relation and rejected alternative from the actual accepted record. Tie each decisive statement to its evidence. Unsupported ornamental prose must not change the answer or masquerade as proof. If no revision was accepted, state the baseline fallback and precise unresolved obligation.

Acceptance: the products example cannot describe the correct opposite durations while claiming the wrong direction without a detected inconsistency. Human faithfulness review remains necessary for semantic claims.

**C24. Complete integration, cost and attribution records.**

Update the single runner/API, caches, schema versions, tests, documentation and change tracker together. Keep sequential model loading and the current consumer-GPU target; measure extra calls and latency from factored verification. Archive adopted versions, source citations and applicable code licenses. Record which adaptations are standard infrastructure and which are the research method.

Acceptance: one reproducible pipeline, not an experimental side path used only for favourable examples. No substantial accuracy gain is claimed until a later adequate held-out comparison demonstrates it.

## 4. Worked examples of the intended change

These are design illustrations, not pre-filled model answers or new gold labels.

### Products

Represent the caption as two bound assertions: disliked product -> short duration, loved product -> long duration. Represent the reported image outcomes independently: disliked -> months, loved -> a week. Compare each binding; do not score mere overlap among “product,” “love,” “hate” and duration words. The resulting relation and its source IDs should determine the explanation.

### Heart sank

Keep the idiom text and its proposed emotional interpretation linked. Under a justified emotional reading, questions concern emotional evidence, not a physically sinking organ. If the image supports only generic distress and the specific interpretation remains uncertain, expose that uncertainty instead of pretending the mapping was proved.

### Faculty meeting

Independently extract “calm and relaxing” from the caption. Compare it against proposed support/contrary states. “Chaotic and tense” cannot be approved as support merely because the same auditor emits true. A correct opposite, missing support, swapped support and duplicate states must generate distinguishable verification outcomes.

### Economy

Track the recovery event and “relatively easily” as separate but connected obligations. Growth alone does not satisfy the ease obligation. The system may use a grounded figurative illustration, but cannot silently delete the qualifier to make the case easier.

## 5. What should not be changed as a shortcut

- Do not change humor/metaphor/sarcasm composition, gold labels or stored split membership.
- Do not turn absent support into contradiction.
- Do not relax acceptance thresholds just to make the tribunal act more often.
- Do not force agents to disagree or require a correction on every case.
- Do not add giant models, paid inference or training before the existing model roles are properly isolated and tested.
- Do not copy a paper's complete pipeline and relabel it novel.
- Do not use the inspected ten-case result as a final benchmark or choose prompts against its gold answers.
- Do not claim JSONSchemaBench, CoVe or DSG proves success on this figurative task.

## 6. Preserving the project's novelty

Constrained decoding, strict validation, decomposition, dependency graphs and factored verification are prior-art methods. Cite them as infrastructure or adapted techniques. FigDebate's contribution under investigation remains the combined treatment of figurative claim preservation, targeted cross-modal disputes, current-image verification and faithful revision traces.

The figurative extension should explicitly address idiom-versus-literal reading, sarcastic expressed-versus-intended polarity, modifier preservation and entity/panel binding. This preserves the intended research direction; it does not establish novelty by assertion. The paper must position the system against relevant prior multimodal debate and scene-graph work and use ablations to show what its particular combination adds.

## 7. Source archive and limitations

The Hugging Face connector's paper-search call returned a missing-tool error. The skill's public HF API fallback succeeded; its search response was saved. Paper metadata and readable copies were obtained through HF, with original PDFs from arXiv. Official implementation documentation was checked separately.

Archive: `C:/Users/Sai Prateek/OneDrive/Desktop/FigDebate_sent/FigDebate_sent/research/critical_solutions_20260907`.

It contains eight paper PDFs, eight metadata records, seven HF markdown reading copies, BibTeX entries, official documentation, search results and download manifests. CheckList's markdown endpoint and an older XGrammar documentation URL returned 404; the original failures remain recorded and alternate official pages were saved. Thirty downloaded source files passed hash verification with zero mismatches.

Unversioned source URLs identify the downloaded snapshot through URL, retrieval time and content hash, not an assertion that all future versions are identical. Saving sources supports attribution and reproducibility; it is not a copyright license or a guarantee against claims. Check applicable implementation licenses before copying code, and cite methods even when implementing them independently.

No model was replaced, package installed, threshold changed or repair applied during this research pass. XGrammar is the preferred candidate to qualify; the graph and factored audit are proposed FigDebate adaptations whose effectiveness remains to be established.
