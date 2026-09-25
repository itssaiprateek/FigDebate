# Tribunal outcome contract

The V5 mapping response and its acceptance conditions are separate.

- `response_valid` checks the generation schema, original wire payload, exact source binding, known citations, and completed fields. An empty binding list with explicit unmatched roles is a valid response.
- `verified` additionally requires a nonempty mapping with no unmatched roles. Valid unresolved responses still stop verification and cannot authorize replacement.
- The gate recomputes these checks from the saved record; saved flags alone are not evidence.

Each `stage_outcomes` row keeps its existing `status` and now adds three dimensions:

| Field | Values | Meaning |
|---|---|---|
| `execution` | `NOT_RUN`, `COMPLETED`, `FAILED` | Whether the operation ran and returned |
| `validation` | `NOT_PRODUCED`, `VALID`, `INVALID`, `TRUNCATED` | Whether a complete response satisfies its structural contract |
| `semantic` | `NOT_ASSESSED`, `UNRESOLVED`, `CONTRADICTORY`, `DISAGREEMENT`, `CHECK_NOT_PASSED`, `ASSESSED` | What the observed reasoning path established about the check |

`ASSESSED` does not mean objectively correct. A valid explanation can still be wrong. A timeout is not a semantic abstention, and a contradictory but well-formed judgment is not a JSON failure.

Top-level `CONTRADICTORY_JUDGMENT` preserves the original answer. The tribunal explicitly blocks a proposal with `SEMANTIC_INCONSISTENCY` even when other fields appear valid. Syntax validity does not grant acceptance eligibility.

No confidence threshold, model, initial arbiter, feedback setting, or retry budget changes accompany this contract. Compact semantic comparison experiments remain outside production under `research/ordered_implementation_20260922`; they must qualify separately before any default changes.
