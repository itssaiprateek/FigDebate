# Constrained-output integration

## Current runtime (7 September 2026)

The shared production backend is now llguidance 1.8.0 (MIT), using its public
LLTokenizer and LLMatcher APIs. No installed library is patched. The local
adapter is limited to batch-one, append-only greedy decoding and rejects
unsupported grammars instead of falling back to unconstrained generation.
Official API: https://github.com/guidance-ai/llguidance
JSON semantics: https://github.com/guidance-ai/llguidance/blob/main/docs/json_schema.md

Generation uses schema property order and explicitly permits outer JSON
whitespace for SentencePiece tokens. This does not mean every equivalent JSON
serialization or every JSON Schema feature is supported. Conformance results
are finite tokenizer checks, not semantic or universal correctness claims.

XGrammar 0.2.5.post1 was rejected for this environment because it requires
Transformers below 5. Its trial packages were removed and Transformers 5.15.0
and huggingface-hub 1.27.0 restored. No model upgrade was made.

## Historical diagnostic dependency

The following describes the retired runtime. LMFE remains available through
requirements-diagnostics.txt solely for reproducing the original failure;
it is not selected by any production generation path.

Dependency: LM Format Enforcer 0.11.3 (MIT), Noam Gat and contributors.
Primary documentation: https://github.com/noamgat/lm-format-enforcer
Pinned release metadata: https://pypi.org/project/lm-format-enforcer/0.11.3/
Reviewed 2026-09-06.

FigDebate uses JsonSchemaParser and the public TokenEnforcer API. The small
tokenizer preprocessing adapter in engine/structured_decoder.py is adapted
from the MIT-licensed 0.11.3 integration because that release imports an alias
removed in Transformers 5.15. No installed package is patched. The complete
copyright notice and license are retained in lm-format-enforcer-LICENSE.txt.
The schemas are project-specific; no paper figures or passages are copied.

The schema check is syntactic only. Independent semantic checks, numeric
range validation, truncation detection, and held-out qualification remain
mandatory. A grammar-valid claim can still be wrong.
