# Critical repair implementation and qualification

Status: in progress, not a completed 24-item release. This document supersedes
the older implementation summary for the changes in this repair pass.

## Landed and integrated

- The atomic claim graph now connects extraction, source auditing, flat-field
  compatibility projection, hearing questions, judge node/evidence bindings and
  the final record. It preserves explicit readings and dependencies. ALL/ANY
  aggregation is deterministic; missing node evidence remains unresolved.
  Graph edits invalidate its fingerprint and earlier semantic endorsement.
  This implementation is not yet semantically qualified.

- One shared llguidance 1.8.0 decoder replaces the defective LMFE runtime adapter.
  It rejects unsupported grammars, uses the real tokenizer serialization, and
  supports the existing batch-one greedy generation path. Original Transformers
  5.15.0 and Hub 1.27.0 are restored; the incompatible XGrammar trial was removed.
- Agent 2's hearing receives dictionaries, not a parsed headings round-trip.
  Lists and nulls retain their types. Content/type guards also apply at the claim
  acceptance boundary, so model approval cannot override malformed input.
- Source verification is blind to the proposed answers. Condition comparison
  uses a neutral SUPPORT/CONFLICT/UNRESOLVED task; software derives the obligation
  result. Invalid or unsupported answers remain unknown.
- Repairs retain the original case and failed-obligation evidence. Each judge
  attempt records its actual prompt/hash and raw response. Claim generation
  records its prompt, schema, raw response, token counts and decoder identity.
- The current judge generates one relation. Software derives the label, and
  conflicting explicitly supplied historical labels are still rejected. Required
  prose is no longer forced to end at a per-field character limit.
- Source-span diagnostics now cover roles, negation, quantities, modifiers and
  scope. Offsets point into the unchanged caption; this is not semantic proof.
- Delivered explanations render decision origin and recorded directional
  evidence. Original model rationale remains separately inspectable rather
  than being presented as a newly verified explanation.
- Removed the unused lexical requirement grounding, repair and validation
  helpers (and their placeholder classifier) from Agent 2. Four tests of that
  retired behaviour were replaced by current invariant/factored-audit tests.
  LMFE is now a historical diagnostic dependency, not a production dependency.

## Qualification evidence

Final decoder gate: `outputs/decoder_qualification_20260907_final.json` passes
all 66 cases, including the exact streamlined extraction and graph/node schemas,
across all three tokenizers. The final environment check reports READY and no
broken package dependencies. All GPU experiments started in this pass have ended.

Latest graph-integrated run: `outputs/critical_graph_smoke_20260907_v2` completed
the single development sample through the canonical stagewise pipeline. It
retained the baseline answer because the generated graph still contained missing
conditions. This is a successful execution check and a failed semantic qualification,
not evidence of tribunal accuracy improvement. The preceding smoke run exposed a
raw-caption/graph identity mismatch; that defect was fixed and the rerun preserved
the source identity while still rejecting the genuinely missing conditions.

`outputs/critical_repair_final_tests_20260907.log` supersedes the earlier regression
logs below. The graph suite contains 337 tests: 335 pass and two historical
heuristic diagnostics remain expected failures. `pip check` reports no broken
requirements. The live three-case judge replay in
`outputs/review_replay_20260907_llguidance.jsonl` produced three valid records,
but valid schema is not verified semantic correctness.

- `outputs/decoder_qualification_20260907_current_schemas.json`: all 60 finite
  conformance cases pass across Mistral, Qwen judge and Qwen vision tokenizers.
  Covers current core, interpretation, witness, candidate, verification, judge,
  source-answer and comparison schemas, boundary tokens and illegal examples.
  This is not an exhaustive JSON Schema or all-token-sequence guarantee.
- `outputs/critical_repair_tests_20260907_final.log`: 327 tests, 325 passes and
  two expected failures in the retired character-limit heuristic diagnostics.
  That heuristic is no longer used by production judge generation.
- `outputs/critical_factored_audit_20260907_relation.json`: Mistral still returns
  UNRESOLVED for a valid contrary condition despite acknowledging contradiction
  in its reason. Its semantic role is not qualified.
- `outputs/critical_factored_audit_20260907_relation_qwen.json`: Qwen passes the
  four controlled variants: complete conditions pass, swapped conditions fail,
  missing and identical conditions fail deterministically. This is a diagnostic
  result, not broad auditor qualification or a production model replacement.
- Earlier unsuccessful decoder and auditor candidates are retained separately.
  No old evaluation results, dataset files, labels or splits were overwritten.

## All 24 items: explicit remaining work

| ID | State | Remaining acceptance work |
|---|---|---|
| C01 | Implemented | Reference runs retained; future experiments need separate source identities. |
| C02 | Implemented, bounded qualification | Extend finite conformance coverage as new schemas are introduced. |
| C03 | Alternative selected and execution-tested | XGrammar incompatible; llguidance passes tokenizer checks and live execution. Semantic quality remains a separate gate. |
| C04 | Integrated and execution-tested | Shared backend applies to all constrained paths; the final one-sample end-to-end run completes. Broader qualification remains. |
| C05 | Implemented, qualification ongoing | Single relation plus explicit node/evidence records and consistency gate. |
| C06 | Implemented, validation ongoing | Complete live retry/overflow/context-budget checks. |
| C07 | Integrated | Typed core, graph, per-node readings and hearing; broad live qualification remains. |
| C08 | Partial | Exact source-span diagnostics added; node-level coverage validation remains. |
| C09 | Implemented, unqualified | Atomic bound graph, projected requirements, node/evidence binding and aggregation; model semantic coverage remains unqualified. |
| C10 | Implemented, unqualified | Per-node expressed/idiomatic/unresolved reading; graph and audit fingerprints prevent stale endorsement. |
| C11 | Implemented for declared operators | SINGLE/ALL/ANY have explicit aggregation; unsupported/ambiguous parses remain UNRESOLVED. Natural-language parsing still needs qualification. |
| C12 | Implemented, unqualified | Node-linked witness questions and canonical requirements; background/normative routes remain separate. |
| C13 | Implemented, unqualified | Factored questions require behavioural coverage qualification. |
| C14 | Implemented | Blind source-answer context is tested; model-error independence is not claimed. |
| C15 | Partial | Neutral relation comparison and guards implemented; semantic reliability unresolved. |
| C16 | Implemented | Placeholder/type/span/node/dependency/citation/fingerprint invariants; no model can override these guards. |
| C17 | Implemented, unqualified | Core repairs preserve unaffected fields; graph regeneration uses identified defects and reaudits the resulting graph. |
| C18 | Not qualified | Qwen passes the controlled diagnostic; broader qualification and sequential role integration remain. |
| C19 | Packet prepared, pending independent review | evaluation/semantic_qualification_review.json covers eight dimensions; no human labels have been fabricated. |
| C20 | Partial | Current deterministic and factored regressions exist; unseen semantic minimal-pair suite remains. |
| C21 | Partial | Decoder/auditor diagnostics and judge replay; complete staged paired pipeline experiments remain. |
| C22 | Gate recorded | CRITICAL_RELEASE_GATE.md defines the strict finite-suite gate before broader outcomes; independent annotations and experiments remain pending. |
| C23 | Integrated, unqualified | Record includes graph, node resolution and cited observations; independent human faithfulness review remains. |
| C24 | Engineering integrated; research release pending | Shared runner/API, graph, caches, dependencies, tests and attribution updated. Semantic qualification, human annotations and adequate paired comparisons remain. |

## Release boundary

Do not report the full repair list as complete. Do not claim accuracy improvement,
research readiness, calibrated confidence or guaranteed semantic correctness.
Do not switch the production auditor to Qwen merely because four cases passed:
it must be integrated into the common baseline/tribunal path with sequential
loading and evaluated on independently reviewed, broader development examples.

The humor/metaphor/sarcasm dataset, stored splits, model revisions, acceptance
thresholds and project research objective are unchanged. The graph is connected
to the single pipeline; it is not a second optional route used to select favourable answers.
