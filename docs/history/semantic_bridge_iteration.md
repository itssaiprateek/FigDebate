# Semantic Bridge Iteration

This experimental branch adds general semantic reasoning as an auditable
evidence bridge. It does not replace Agent 1, Agent 2, the evidence ledger, or
the deterministic Review Board.

## Decision path

1. Agent 1 records typed, label-blind visual observations.
2. Agent 2 preserves the source caption and validates three typed field groups.
3. The tribunal reports its best semantic judgment separately from admissibility.
4. The bridge generator joins only cited visual evidence to the immutable
   caption proposition and selects a family from structural evidence.
5. The independent verifier checks provenance, entity identity, scope,
   affirmative direction, counter-interpretation strength, confidence, and
   position-order consistency.
6. Plausible or insufficient bridges remain diagnostic. A corroborated bridge
   becomes decision-grade only with independent visual and caption roots.
7. The unchanged Review Board accepts a revision only when the proposed
   direction is stronger and no stronger opposing evidence exists.

The tribunal constructs support and conflict cases symmetrically. A broad humor
classification is secondary to the typed relation family, and humor or
incongruity alone is never directional evidence. There is one post-hearing
review. Formatting repair may rewrite the same response into the contract, but
it cannot ask the model to reason about the case a second time. Normative
evaluations cannot be reversed from a charitable alternative or missing
context.

Tribunal responses use a descriptive schema that keeps the semantic judgment
separate from deterministic admissibility. Cosmetic optional-field defects may
be normalized, and one malformed response may be rewritten with a contract-only
repair prompt. Judgment/relation mismatches, unknown evidence IDs, and missing
semantic fields remain invalid. The repair does not invent or promote evidence;
every revision still passes the unchanged evidence gate.

## Checkpointed supervisor design

The supervisor runs once after the optional targeted hearing. It receives a
canonical, gold-free dossier containing both agents' semantic outputs, the
comparator, the hidden-label decision audit, the hearing, the complete active
evidence catalogue, and ordered stage history. The initial prediction remains
hidden to avoid anchoring.

The dossier is persistent logical memory, not persistent GPU residency. Qwen is
loaded once for the supervisor batch and unloaded before another large model is
loaded. This preserves the full case history on 8 GB hardware without reducing
image input quality or generation limits. Higher-VRAM profiles may improve
batch throughput but do not receive a different reasoning contract.

Evidence requirements are dependency-aware. Immutable caption meaning is
always mandatory. Generated expected/opposite states are required only when no
independently verified directional relation or valid Agent 2 hearing frame is
available. Optional reasoning-profile metadata is diagnostic and cannot veto a
direct proof that does not depend on it.

## Reproducibility contract

Official evaluations use a nested, checksummed sample manifest and the fixed
`paper-8gb` profile. Every generation stage receives a seed derived from the
global seed, sample ID, stage, round, and retry. Stage checkpoints include the
pipeline/manifest/profile fingerprint. `evaluation.check_reproducibility`
locates the first stage that differs across repeat, order, size, and resume
runs. Quantized CUDA execution may not be bit-for-bit identical across GPU
families, so both exact field agreement and final-label agreement are reported.

Gold labels and dataset phenomenon labels are not available to any inference
component. They are used only by the evaluator after predictions are saved.

## Rollout

Use `--semantic-bridge-mode shadow` for the paired safety run. Inspect
`semantic_bridge_analysis.csv` for proposal accuracy, counterfactual
corrections, counterfactual harms, family balance, and verification failures.
Use `corroborated` only after freezing the shadow thresholds. Disable the mode
immediately if accepted harm rises or any provenance check is bypassed.
