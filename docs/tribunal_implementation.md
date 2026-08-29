# Tribunal implementation contract

This document is the twelve-phase implementation checklist for the experimental
`tribunal-minicpm` branch. It describes general pipeline behavior and contains
no sample-specific rules.

## Phase 1 — Baseline and decision trace

The initial Arbiter result, raw debate proposal, constrained proposal, review
outcome, and tribunal outcome are stored as immutable decision checkpoints.
The current label is never silently rewritten.

## Phase 2 — Typed claim contract

Agent 2 preserves caption entities, polarity, comparisons, actions, and
outcomes. Expected and opposite states are audited as an actual relation pair.
Background and normative premises are marked explicitly instead of being
treated as pixel-level facts.

## Phase 3 — Evidence lifecycle

Every ledger item is one of `OBSERVATION`, `BINDING`, `RELATION_CANDIDATE`, or
`VERIFIED_RELATION`. Only the final level may be decision grade. Reliability is
combined by independent provenance root, so repeated wording cannot create
artificial evidence strength.

## Phase 4 — Agent 1 visual witness

Qwen3-VL answers one neutral visual question at a time. Python validates and
assembles its evidence. Agent 1 records visible facts and bindings but never
assigns `ENTAILS` or `CONTRADICTS`; failures and genuine absence remain distinct.

## Phase 5 — Universal issue routing

Questions are routed by relation and failure type: OCR binding, comparison or
outcome, directional state, symbol attachment, normative reasoning, background
knowledge, conflicting verification, insufficient observation, or semantic
relation. Routing is independent of sample ID, dataset label, and phenomenon.

## Phase 6 — Raw proposal preservation

The Arbiter's unconstrained debate proposal is logged before evidence rules are
applied. This separates model reasoning quality from gate behavior during
evaluation.

## Phase 7 — Label-blind mediation

Before debate, the mediator sees the image, caption, agent records, comparator,
and current ledger, but not the current Arbiter label or gold label. Agents see
only their own neutral questions; the mediator's vote, confidence, and rationale
are hidden.

## Phase 8 — Bounded tribunal

After both agents answer, the mediator returns `RESOLVE`, `FOLLOW_UP`, or
`ABSTAIN`. At most one targeted follow-up is allowed, making two review rounds
the hard maximum. Invalid output and unresolved evidence preserve the current
decision.

## Phase 9 — Independent verification and Review Board

Before resolution, a validated Agent 1 witness and Agent 2 claim audit may create
one cross-agent verified relation with full provenance. For semantic or normative
cases that cannot be reduced to automatic visual rules, a resolved mediator
relation may be combined with the current-round visual witness and the independently
preserved claim audit. None of those three sources is sufficient alone. The
combined relation is decision grade only when all contracts, confidence, citation,
question, and provenance checks pass. Every resolution then passes the ordinary
deterministic Review Board, and cannot defeat stronger opposing verified evidence.

## Phase 10 — Feedback containment

Feedback memory supplies procedural diagnostic questions only. It never stores
or replays a gold direction and cannot bypass current-image evidence or the
Review Board.

## Phase 11 — One runtime path and bounded VRAM

The public `FigDebate` API delegates to the same `StagewiseRunner` used by
experiments. Qwen3-VL, Mistral, and the judge are loaded sequentially and cleared
between stages; no two large model runtimes are intentionally resident together.

## Phase 12 — Reproducibility and evaluation

The runner exports tribunal state, rounds, stop reason, verified evidence ID,
revision result, and the complete decision trace. Environment validation checks
the control plane and pinned model identities. Unit tests cover contracts and a
model-free end-to-end tribunal resolution; full GPU accuracy evaluation remains
a separate locked-sample experiment.

## Non-negotiable invariants

- Missing evidence is not contradiction.
- Lexical overlap and generic text NLI are candidates, never visual proof.
- Agent 1 does not vote on the final label.
- The mediator never sees the gold label or current Arbiter label.
- The mediator cannot turn its own observation or relation judgment into
  evidence without both the current Agent 1 witness and Agent 2 claim audit.
- Model-generated questions cannot expose a dataset decision label.
- Only current-sample evidence IDs are accepted.
- Every revision passes the same deterministic Review Board.
- The tribunal stops after two reviews and may abstain.
