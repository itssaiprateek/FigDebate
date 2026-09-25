# Critical failure diagnosis — 7 September 2026

Scope: diagnosis of the three critical problems in `outputs/checkpointed_supervisor_v5_oomfix_10`. No production prompt, threshold, model, decoder, or dataset was changed. New files are diagnostic harnesses, known-defect regression reproductions, this report, and generated diagnostic artifacts. These tests are not a new accuracy evaluation.

## Findings ranked by causal certainty

### 1. Confirmed decoder defect: legal sentence-ending tokens are suppressed

Location: `engine/structured_decoder.py` calls LM Format Enforcer 0.11.3's `TokenEnforcer`; its free-text shortcut is implemented in `tokenenforcer.py::_collect_allowed_tokens` and `tokenizerprefixtree.py::JsonFreetextTokenCache.add_token` in the installed dependency.

CPU tests used the actual installed Mistral and Qwen judge tokenizers and the production tribunal JSON schema. A short complete review fails at the token containing period, closing quote, and comma (`.\",`): token 9191 for Mistral and 10152 for Qwen. This is a legal transition from a sentence inside a JSON string to the next field.

The free-text cache excludes tokens with a nonterminal quote when they are not valid *inside* a string. Dynamic traversal then considers only tokens starting with a quote, missing tokens that start with sentence punctuation and cross the string boundary. The result is an incomplete allowed-token set. This is not the model choosing an overly long sentence; its usual closing token is being removed from consideration.

Controlled intervention: disable only `JsonSchemaParser.shortcut_key` inside the CPU test process. The complete Mistral token sequence was then accepted, without altering its schema, text, tokenizer, or token IDs (captured in tool output). Exhaustive Qwen traversal, including a prefix-bounded variant, was too slow; both diagnostics were stopped by their verified process IDs. The final saved comparison instead validates the exact complete character sequence against the grammar and compares that with production token filtering: grammar accepts, token filtering rejects, for both tokenizers. This isolates the token-filtering layer without claiming a completed shortcut-disabled Qwen enumeration. Neither the installed dependency nor production adapter was edited.

At `maxLength`, the shortcut stops applying and the character parser allows only a closing quote. A separate character-level test confirms no letter can continue the word at that boundary. Saved judge responses cluster at 318–320 characters, including visibly unfinished words. Thus two mechanisms interact: suppressed ordinary exits and forced closure at the field limit. Increasing the overall 1024-token budget cannot restore a blocked token. The actual judge runtime has not yet been rerun with a corrected decoder; the fraction of live failures attributable to this defect remains unmeasured.

This adapter is shared with Agent 2. Its free-text generation is therefore exposed too, although that does not establish that all Agent 2 semantic errors originate here.

### 2. Confirmed semantic-auditor discrimination failure

Locations: `engine/claim_semantics.py::audit_core`; `agents/claim_extraction.py::_generate_section`; semantic authority in `engine/claim_contract.py` and `engine/claim_witness.py`.

Four live calls used the existing pinned, quantized Mistral, the same caption, the same seed, the same audit prompt/schema, and the same other claim fields. Only expected/opposite states changed. Caption: “The faculty meeting was calm and relaxing”.

| Probe | Expected support state | Expected conflict state | Auditor support / conflict |
|---|---|---|---|
| Complete opposition | meeting calm and relaxing | meeting chaotic and tense | true / false |
| Missing support | `None` | meeting chaotic and tense | true / false |
| Swapped direction | meeting chaotic and tense | meeting calm and relaxing | true / false |
| Identical states | meeting calm and relaxing | meeting calm and relaxing | true / false |

All six audit Booleans were identical across all four probes: true, true, true, true, true, false. All outputs passed the JSON schema and none hit the generation limit. The valid opposite was rejected; missing and reversed support were approved. Reasons sometimes assert semantic opposition even when the support is `None`, while the conflict Boolean says false.

The token-level tests separately admit both all-true and all-false audit responses, ruling out a simple grammar rule forcing the final Boolean to false. This does not rule out more subtle generation effects.

The deterministic contract promotes the auditor's Booleans to semantic authority. Consequently a fresh call is not an independent correctness guarantee. The defect is localized to the model-audit path, not label leakage, image processing, or downstream thresholding in these caption-only probes. The relative contributions of prompt/schema conditioning, model capacity, and quantization have NOT been experimentally separated. A claim that simply changing the model or rewording the audit will solve this is not supported.

### 3. Confirmed deterministic containment hole: placeholder requirements can be approved

Location: `engine/claim_witness.py`, requirement validity calculation.

Fault injection supplies a schema-valid witness with support=`"None"`, conflict=`"None"`, and an auditor that returns six true checks. The production gate returns `requirements_valid=True` with no errors.

Cause: generic string schemas admit placeholders; nonempty string truthiness treats `"None"` as evidence; `is_missing_evidence_only` does not recognize that sentinel. There is no independent minimum semantic-content check protecting against this obvious false-positive auditor output. The fault-injected auditor is deliberately not a live-model accuracy test; it proves what happens when the already-observed auditor fallibility reaches the gate.

### 4. Confirmed typed-data loss at the witness boundary

Locations: `engine/debate.py::build_agent2_challenge_prompt` and `engine/claim_witness.py`.

The hearing prompt serializes values through `str(value or "None")`, so an actual empty list becomes the text `None`. The witness reconstructs list fields with `ast.literal_eval`; `None` becomes Python null, not an empty list. That reconstructed object goes to the semantic auditor without validation against the CORE input schema. This was reproduced from all ten saved hearing inputs and in a standalone regression test.

This is a confirmed type-integrity defect. It is NOT sufficient to explain the whole audit failure: the four controlled live audit probes used correctly typed empty lists and still failed discrimination.

### 5. Agent 2's claim-to-requirement mapping fails before tribunal adjudication

Saved-output trace localizes several failures:

- For “heart sank”, the interpretation stage contains disappointment/sadness, but the core state is `None`, repair proposes “heart floated up”, and hearing support requests physical heart sinking. Interpretation and core/requirement representations are not made semantically consistent.
- The products case substitutes real-world duration testing for comparing the two durations in the image.
- The faculty-meeting requirement contains copied task instructions rather than a condition.
- The accepted rotten-heart witness uses fresh-heart evidence as its support requirement for a rotten-heart assertion.

The witness combines multiple tasks in one generation (extraction endorsement, support, opposition, figurative mechanism, ambiguity, reason). Its output schema defines free strings rather than verifiable relation structures. `audit_core` receives only CORE properties, not the interpretation hypothesis; the intended/literal mapping is not explicitly audited as a linked structure. These are confirmed architectural gaps, with observed failures at their outputs. The causal contribution of each generation instruction has not yet been isolated with ablations, so this is not evidence that one prompt edit will resolve them.

### 6. Clause validation is a boundary heuristic, not a completeness guarantee

Location: `engine/output_contracts.py::saturated_text_fields`.

Two counterexamples reproduce its limits:

- `because the` passes because it is short.
- `The sky appears blue` is flagged at a 20-character limit because it lacks terminal punctuation, although its clause is complete.

These are respectively a missed incomplete clause and a false-positive completeness flag. The function's documented scope is boundary detection; treating it as proof of general linguistic completeness exceeds that scope. Removing it without another containment mechanism would merely hide genuine truncations.

The repair path reruns with the same decoder/schema, and instructs preservation of an already incomplete response's meaning. It therefore does not remove the confirmed token-filtering cause. Saved diagnostics retain attempt timings and flags but not a separate raw text for every attempt, limiting exact first-versus-repair comparison.

## Evidence and reproducibility

- `outputs/critical_root_cause_20260907.json`: source-record hash, diagnostic script hash, CPU results, complete four live audit inputs/outputs and timing.
- `outputs/critical_decoder_probe_20260907.json`: original tokenizer probes.
- `outputs/critical_decoder_probe_20260907_d.json`: completed token-filtering versus exact character-grammar comparison; both tokenizers reject a legal sentence-closing token. Interrupted B/C diagnostics did not produce final artifact files.
- `evaluation/diagnose_critical_contracts.py`: CPU and optional live audit harness.
- `evaluation/probe_decoder_boundaries.py`: CPU decoder harness; intervention is process-local only.
- `tests/test_critical_diagnostic_reproductions.py`: four desired invariants marked **expected failures**, documenting unresolved defects. Four expected failures are not four successful repairs.

The initial decoder diagnostic invocation had a duplicate local-only keyword in the harness; that harness error was corrected before collecting decoder results. It was not a pipeline failure.

Full regression run: 325 tests in 20.793 seconds, with 321 passes and four explicitly expected failures. The expected failures reproduce the newly identified unresolved defects; they do not establish fixes. The four live semantic probes completed successfully as executions, but failed the intended discrimination checks.

## Repair acceptance criteria — not repairs performed

1. Decoder: legal cross-boundary tokens must be admitted, illegal sequences must remain blocked, and schema completeness must remain enforced. Test both tokenizers, sentence endings, escapes, whitespace, field transitions and actual live first/repair outputs. Blanket disabling of the optimization may be too slow; it is a diagnostic intervention, not the selected production repair.
2. Representation: preserve source caption and typed data; distinguish literal wording, figurative interpretation, and evidence requirements with explicit bindings. Placeholder, identical and direction-swapped requirements must not gain authority from a single Boolean approval.
3. Auditor: qualify with controlled positive/negative pairs spanning roles, negation, modifiers, comparisons and figurative readings. Measure false acceptance and false rejection separately. The current four probes must discriminate correctly before a new large tribunal run is informative.
4. Containment: expose unresolved checks as unresolved, never as proved evidence; repair must target a demonstrated defect and be revalidated. Do not lower acceptance thresholds to improve acceptance counts.
5. End-to-end: after local repairs, repeat fixed diagnostic cases, then fresh development cases, before frozen held-out evaluation. Preserve the existing three-category dataset.

No finite language-model test establishes correctness on every possible caption. The enforceable requirement is deterministic integrity and safe handling of detected failures, plus measured semantic reliability with clearly stated limits. No accuracy improvement or complete root-cause explanation of model internals is claimed here.
