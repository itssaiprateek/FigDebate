# Implemented adaptations and limits — 6 September 2026

The full papers, original BibTeX, reading copies and metadata are retained in
`C:/Users/Sai Prateek/OneDrive/Desktop/FigDebate_sent/FigDebate_sent/research/tribunal_audit_20260905`.
The ten-source manifest records hashes for 40 files. This is an attribution
record, not permission to republish copyrighted content.

## Independent arguments and bounded debate

Du et al. (2305.14325), Liang et al. (2305.19118) and ReConcile (2309.13007)
motivate independent initial answers, exchange of competing arguments, diversity
and bounded discussion. `engine/candidate_cases.py` generates a Qwen visual case
and a Mistral caption-plus-observations case before peer exposure. Candidate
claims are not themselves corroborating evidence. `engine/deliberation.py`
and the canonical runner route specific disputes to witnesses, bound tribunal
reviews to two rounds and stop without new information.

`engine/control_conditions.py` supplies a local strict-majority two-agent vote
and a conventional two-round control. Both second-round debaters see only the
other agent's first-round public argument. Ties, failed agents and unresolved
votes retain the baseline rather than manufacturing a majority. These are local
controls, not exact reproductions of any cited paper. Compute matching is not
claimed merely because the controls use the same models.

Debate-or-vote (2508.17536), MAD evaluation (2502.08788), and confidence/diversity
(2601.19921) motivate stronger controls and qualified uncertainty. No oracle
gold locking, source-specific label shortcut or claimed reproduction of trained
confidence modulation is implemented.

## Structured generation and claim semantics

JSONSchemaBench (2501.10868) motivates measuring validity, task quality and cost
separately. Typed claim/witness/judge schemas, bounded field repair, complete-JSON
stopping and saturation detection are local implementations. They do not prove
that a valid response preserves a modifier or correctly sees an image.

The LM Format Enforcer tokenizer adapter includes a small MIT-licensed adaptation;
see STRUCTURED_OUTPUT_ATTRIBUTION.md and lm-format-enforcer-LICENSE.txt. No other
paper author's implementation is claimed as copied or reproduced here.

## Calibration

Guo et al. (1706.04599), section 4.1, motivates held-out score calibration while
the classifier remains fixed. `evaluation/confidence_calibration.py` fits an
L2-regularized sigmoid to the exported `final_confidence`, targeting delivered
answer correctness on the locked internal calibration partition. It is not
temperature scaling or an exact paper reproduction. Application requires matching
method identities and disjoint image groups. It does not change labels or loosen
acceptance gates. No fitted or validated calibrator exists yet.

## Dataset and novelty

V-FLUTE (2405.01474) supplies the task and explanation objective. The existing
humor/metaphor/sarcasm subset is preserved. Expressed captions remain distinct
from inferred sarcastic intentions. MAD-Sherlock (2410.20140) establishes relevant
multimodal debate prior art; FigDebate must not claim to invent multimodal debate.

The research contribution under investigation is the combined treatment of
figurative claim preservation, targeted cross-modal evidence disputes, fresh
image-bound verification, evidence ancestry and faithful revision traces.
Whether this combination improves accuracy and explanation faithfulness remains
an empirical question. Current model failures and conservative false rejections
are limitations, not successful research results.
