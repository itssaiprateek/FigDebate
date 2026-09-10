"""Content-based stopping: new identifiers alone do not constitute evidence."""
import hashlib
import json
from engine.evidence_ledger import is_admissible_evidence


def evidence_signature(ledger):
    rows = {json.dumps(
        {key: item.get(key) for key in ("text", "relation", "grounded",
                                       "decision_grade", "evidence_level")},
        sort_keys=True, ensure_ascii=False)
        for item in ledger if is_admissible_evidence(ledger, item)}
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()


def deliberation_signature(result):
    """A clarified claim can be useful even when visual facts are unchanged."""
    language = result.get("language_output", {}) or {}
    witness = (result.get("debate_details", {}) or {}).get("agent2_critique", {}) or {}
    state = {"evidence": evidence_signature(result.get("evidence_ledger", [])),
             "claim": {key: language.get(key) for key in (
                 "caption_proposition", "claim_subject", "claim_predicate", "claim_object",
                 "negation", "quantities", "claim_modifiers", "comparison_direction", "time_or_panel_scope")},
             "graph": (language.get("claim_graph") or {}).get("fingerprint"),
             "graph_readings": (language.get("claim_graph") or {}).get("reading_fingerprint"),
             "audit": {key: (language.get("_caption_semantic_audit") or {}).get(key) for key in
                       ("audit_status", "claim_graph_fingerprint")},
             "claim_witness": {key: witness.get(key) for key in (
                 "support_requirement", "conflict_requirement", "requirements_valid", "reading_clarification", "question_answers")}}
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
