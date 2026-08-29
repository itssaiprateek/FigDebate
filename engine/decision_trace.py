"""Immutable, JSON-serializable checkpoints for decision provenance."""

from copy import deepcopy


TRACE_SCHEMA_VERSION = "1.0"


def decision_checkpoint(stage, decision=None, *, ledger=None, metadata=None):
    """Return a compact snapshot without mutating the supplied decision."""
    decision = decision or {}
    return {
        "stage": str(stage),
        "label": decision.get("label"),
        "confidence": decision.get("confidence"),
        "method": decision.get("decision_method"),
        "relation_status": decision.get("_relation_status"),
        "revision_status": decision.get("_revision_status"),
        "raw_proposed_label": decision.get(
            "_raw_debate_proposed_label",
            decision.get("_unconstrained_proposed_label"),
        ),
        "cited_evidence_ids": list(
            decision.get("_model_cited_evidence_ids", []) or []
        ),
        "ledger_ids": [
            item.get("id") for item in (ledger or []) if item.get("id")
        ],
        "metadata": deepcopy(metadata or {}),
    }


def append_decision_checkpoint(
    trace, stage, decision=None, *, ledger=None, metadata=None
):
    """Append a new immutable checkpoint and return a new trace list."""
    output = deepcopy(list(trace or []))
    output.append(
        decision_checkpoint(
            stage, decision, ledger=ledger, metadata=metadata
        )
    )
    return output


def attach_decision_trace(decision, trace):
    output = dict(decision or {})
    output["_decision_trace_schema"] = TRACE_SCHEMA_VERSION
    output["_decision_trace"] = deepcopy(list(trace or []))
    return output
