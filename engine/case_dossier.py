"""Canonical, label-blind case history for checkpointed judge supervision.

The dossier is the judge's durable memory.  It keeps the complete active
evidence catalogue in system memory and renders a bounded GPU prompt without
discarding the existence, provenance, or direction of lower-priority entries.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

from engine.evidence_ledger import evidence_provenance_roots, is_active_evidence, is_admissible_evidence


_PRIVATE_SEMANTIC_KEYS = {
    "_raw_output", "_raw_response", "_raw_primary_response",
    "_raw_retry_response", "_raw_relation_response",
    "_raw_relation_retry_response", "_generation_diagnostics", "_timing",
    "_evidence_ledger", "_decision_trace",
}


def _clean(value, limit=None):
    text = " ".join(str(value or "").split())
    return text[:limit] if limit else text


def _semantic_copy(value):
    """Copy public semantic fields while removing raw/debug duplication."""
    if isinstance(value, dict):
        return {
            str(key): _semantic_copy(item)
            for key, item in value.items()
            if key not in _PRIVATE_SEMANTIC_KEYS
            and not str(key).startswith("_raw_")
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_copy(item) for item in value]
    return deepcopy(value)


def _decision_audit(decision):
    decision = decision or {}
    review = decision.get("_review_board", {}) or {}
    audit = decision.get("_evidence_audit", {}) or {}
    # The initial label remains hidden to avoid anchoring.  The judge still
    # receives the complete procedural state and makes an independent verdict.
    return {
        "label_hidden": True,
        "confidence": decision.get("confidence"),
        "decision_method": decision.get("decision_method"),
        "final_contract_valid": decision.get("_final_decision_valid"),
        "relation_status": decision.get("_relation_status"),
        "revision_status": decision.get("_revision_status"),
        "deficiencies": list(decision.get("_deficiencies", []) or []),
        "review_board_status": review.get("status"),
        "directionally_grounded": review.get("directionally_grounded"),
        "source_grounded": review.get("source_grounded"),
        "evidence_status": audit.get("status"),
        "cited_evidence_ids": list(
            audit.get("source_cited_evidence_ids", []) or []
        ),
    }


def _ledger_entry(item, ledger):
    return {
        "id": item.get("id"),
        "source": item.get("source"),
        "type": item.get("type"),
        "evidence_class": item.get("evidence_class", "OTHER"),
        "text": _clean(item.get("text")),
        "relation": item.get("relation"),
        "grounded": bool(item.get("grounded", False)),
        "decision_grade": bool(
            item.get("decision_grade", False)
            or (item.get("verification", {}) or {}).get(
                "decision_grade", False
            )
        ),
        "evidence_level": item.get("evidence_level"),
        "verification_method": item.get("verification_method"),
        "lifecycle_status": item.get("lifecycle_status", "ACTIVE"),
        "question_id": item.get("question_id"),
        "provenance_roots": evidence_provenance_roots(ledger, item),
    }


def build_case_dossier(
    caption,
    visual_output,
    language_output,
    comparison,
    decision,
    evidence_ledger,
    debate_details=None,
    pre_hearing=None,
):
    """Build the immutable, gold-free record of one sample's reasoning."""
    active = [
        item for item in (evidence_ledger or []) if is_admissible_evidence(evidence_ledger, item)
    ]
    ledger = [_ledger_entry(item, active) for item in active]
    history = [
        {"stage": "VISUAL_GROUNDING", "completed": bool(visual_output)},
        {"stage": "CLAIM_ANALYSIS", "completed": bool(language_output)},
        {"stage": "COMPARISON", "completed": bool(comparison)},
        {"stage": "INITIAL_DECISION", "completed": bool(decision)},
    ]
    if pre_hearing:
        history.append({
            "stage": "UNCERTAINTY_ROUTING",
            "completed": True,
            "issue_type": pre_hearing.get("issue_type"),
            "reasons": list(pre_hearing.get("reasons", []) or []),
        })
    if debate_details:
        history.append({
            "stage": "TARGETED_HEARING",
            "completed": True,
            "review_status": debate_details.get("review_status"),
            "agent1_response_status": (
                debate_details.get("agent1_critique", {}) or {}
            ).get("response_status"),
            "agent2_requirements_valid": debate_details.get(
                "agent2_requirements_valid"
            ),
        })
    from engine.candidate_cases import visible_candidates
    return {
        "candidate_cases": visible_candidates(visual_output or {}, language_output or {}),
        "schema_version": "1.0",
        "gold_label_visible": False,
        "initial_label_visible": False,
        "source_caption": caption,
        "visual_agent": _semantic_copy(visual_output or {}),
        "claim_agent": _semantic_copy(language_output or {}),
        "deterministic_comparator": _semantic_copy(comparison or {}),
        "initial_decision_audit": _decision_audit(decision),
        "uncertainty_route": _semantic_copy(pre_hearing or {}),
        "targeted_hearing": _semantic_copy(debate_details or {}),
        "evidence_catalog": ledger,
        "history": history,
    }


def _pick(record, fields):
    return {key: deepcopy(record[key]) for key in fields
            if key in record and record[key] not in (None, "", [], {})}


def compact_graph(graph):
    """Lossless task view of obligations; omit debug history and duplicate spans."""
    result = {k: deepcopy(graph[k]) for k in ("composition", "fingerprint", "errors", "condition_origin",
              "representation", "decomposition_status", "semantic_verified", "reading_fingerprint") if k in graph}
    keys = ("id", "source_quote", "subject", "predicate", "object", "qualifiers",
            "reading", "interpreted_proposition", "depends_on")
    if not graph.get("condition_origin"):
        keys += ("support_condition", "conflict_condition")
    result["nodes"] = [{k: deepcopy(n[k]) for k in keys if k in n} for n in graph.get("nodes", [])]
    return result


def _context_id(kind, value):
    text = json.dumps([kind, value], sort_keys=True, ensure_ascii=False)
    return "CTX_" + hashlib.sha256(text.encode()).hexdigest()[:16]


def _reading_view(record):
    """Retain partial answers and uncertainty; never promote them to facts."""
    fields = ("question_id", "question", "answer", "source_ids",
              "unknown", "answer_status", "delivery_status", "source_anchored",
              "observed_entity", "observed_state", "image_region", "response_status")
    view = _pick(record, fields)
    if record.get("requested_question"):
        view["question"] = record["requested_question"]
    view["authority"] = "fallible_witness_testimony_not_verified_relation"
    if record.get('response_status') == 'QUESTION_BINDING_MISMATCH':
        for key in ('answer', 'observed_state', 'observed_entity', 'image_region'):
            view.pop(key, None)
        view['content_withheld_reason'] = 'question_binding_mismatch'
    if record.get("source_anchored") is False:
        view.pop("answer", None)
        view["content_withheld_reason"] = "invalid_source_identity"
    return view


def context_catalog(dossier):
    """Retrievable arguments/readings are not evidence-ledger observations."""
    records = []
    current_answers = [
        _reading_view(answer)
        for role in ("agent1_critique", "agent2_critique")
        for answer in dossier.get("targeted_hearing", {}).get(role, {}).get("question_answers", [])]
    for candidate in dossier.get("candidate_cases", []):
        records.append({"kind": "candidate_argument", "content": deepcopy(candidate)})
    for reading in (dossier.get("claim_agent", {}).get("claim_graph") or {}).get("readings", []):
        view = _reading_view(reading)
        # The current full Q/A is mandatory below. Do not pay for it twice.
        if view not in current_answers:
            records.append({"kind": "caption_reading", "content": view})
    for hearing in dossier.get("targeted_hearing", {}).get("hearing_history", []):
        for role in ("agent1_critique", "agent2_critique"):
            reply = hearing.get(role, {})
            for answer in reply.get("question_answers", [reply] if reply else []):
                records.append({"kind": "prior_witness_answer", "role": role, "content": _reading_view(answer)})
    unique = {}
    for record in records:
        identifier = _context_id(record["kind"], record)
        unique[identifier] = dict(record, context_id=identifier, decision_grade=False)
    return list(unique.values())


def retrieve_dossier_context(dossier, identifiers):
    by_id = {item["context_id"]: item for item in context_catalog(dossier)}
    if any(identifier not in by_id for identifier in identifiers):
        raise ValueError("Unknown or stale context ID")
    return [deepcopy(by_id[i]) for i in dict.fromkeys(identifiers)]


def disclose_review_context(packet, dossier):
    """Expose the complete finite argument/reading catalogue before proposing.

    Observations stay in their own ledger. Context remains fallible testimony;
    exposing it is not independent verification. Return IDs that must stay in
    the prompt through every retrieval, or fail explicitly if they cannot fit.
    """
    packet = deepcopy(packet)
    catalogue = context_catalog(dossier)
    expected = {item['context_id'] for item in catalogue}
    present = {item['context_id'] for key in ('candidate_cases', 'context_records')
               for item in packet.get(key, [])}
    unknown = {item['context_id'] for item in packet.get('remaining_context_index', [])} - expected
    if unknown:
        raise ValueError('Unknown or stale context IDs in review packet')
    packet.setdefault('context_records', []).extend(
        deepcopy(item) for item in catalogue if item['context_id'] not in present)
    packet['remaining_context_index'] = []
    packet['context_delivery'] = {'policy': 'complete_before_proposal',
        'record_count': len(expected), 'semantic_correctness': 'not_established_by_delivery'}
    return packet, expected


def render_judge_dossier(dossier, detailed_evidence_limit=18):
    """Allowlisted inference view. The durable dossier is never mutated.

    Indexed evidence has NOT been read and cannot be cited until retrieved.
    Previous decisions and free-form procedural narratives are excluded.
    """
    dossier = dossier or {}
    claim = dossier.get("claim_agent", {}) or {}
    visual = dossier.get("visual_agent", {}) or {}
    hearing = dossier.get("targeted_hearing", {}) or {}
    caption = dossier.get("source_caption", "")
    packet = {
        "schema_version": "2.0",
        "gold_label_visible": False,
        "initial_label_visible": False,
        "source_caption": caption,
        "initial_decision_audit": {"label_hidden": True},
        "visual_agent": _pick(visual, (
            "initial_scene", "salient_objects", "visible_text",
            "uncertain_observations",
        )),
        "claim_agent": _pick(claim, (
            "caption_proposition",
            "linguistic_cue", "polarity_reversal", "claim_subject",
            "claim_predicate", "claim_object", "claim_source", "claim_target",
            "asserted_property", "expected_visual_state", "opposite_visual_state",
            "literal_polarity", "intended_polarity", "comparison_direction",
            "evaluation_target", "time_or_panel_scope",
            "negation", "quantities", "claim_modifiers",
            "claim_graph",
        )),
        "targeted_hearing": {},
        "candidate_cases": [],
        "context_records": [],
        "remaining_context_index": [],
    }
    if hearing.get("tribunal_semantic_questions"):
        packet["tribunal_semantic_questions"] = deepcopy(hearing["tribunal_semantic_questions"])
    for item in context_catalog(dossier):
        if item["kind"] == "candidate_argument":
            packet["candidate_cases"].append(dict(item["content"], context_id=item["context_id"]))
        else:
            packet["context_records"].append(item)
    for role in ("agent1_critique", "agent2_critique"):
        record = hearing.get(role, {}) or {}
        unusable = (record.get("_format_valid") is False
                    or (role == "agent2_critique" and record.get("requirements_valid") is not True)
                    or record.get("response_status") in {"INVALID_RESPONSE", "FAILED"})
        if record and unusable:
            packet["targeted_hearing"][role] = {"status": "UNUSABLE_WITNESS_RECORD", "content_withheld": True,
                "question_answers": [{"question_id": answer.get("question_id"),
                    "question": answer.get("question", answer.get("requested_question")),
                    "answer_status": answer.get("answer_status", answer.get("response_status")),
                    "content_withheld": True} for answer in record.get("question_answers", [])]}
            continue
        packet["targeted_hearing"][role] = _pick(record, (
            "question_id", "observed_entity", "observed_state", "image_region",
            "response_status", "support_requirement", "conflict_requirement",
            "requirements_valid", "_format_valid", "reading_clarification",
        ))
        if record.get("question_answers") is not None:
            # The top-level fields are compatibility aliases of one answer.
            # Full quotes/spans/hashes stay in the durable record; the prompt
            # has one exact source and one copy of each current answer.
            packet["targeted_hearing"][role] = _pick(record, ("requirements_valid", "_format_valid"))
            packet["targeted_hearing"][role]["question_answers"] = [
                _reading_view(answer) for answer in record["question_answers"]]
            packet["targeted_hearing"][role]["communication"] = deepcopy(record.get("communication", {}))
        if role == "agent2_critique" and record.get("question_answers") is None:
            reading = record.get("reading_clarification") or {}
            if reading:
                packet["targeted_hearing"][role]["reading_clarification"] = (
                    _reading_view(reading)
                    if reading.get("source_anchored") else
                    {"status": "UNRESOLVED", "reason": "Reading unavailable or not anchored to the source caption."})
    contract = claim.get("claim_contract") or {}
    if claim.get("claim_graph") is not None:
        packet["claim_agent"]["claim_graph"] = compact_graph(claim["claim_graph"])
        if contract.get("flat_predicate_authoritative") is False:
            # The canonical graph already carries the authoritative obligations.
            # Obsolete flat paraphrases must not form a competing claim frame.
            packet["claim_agent"] = _pick(packet["claim_agent"], (
                "caption_proposition", "claim_graph", "expected_visual_state", "opposite_visual_state"))
    if contract:
        packet["claim_agent"]["extraction_status"] = contract.get("decomposition_status", "UNVERIFIED_EXTRACTION")
        if not (contract.get("field_groups", {}).get("relation", {}) or {}).get("valid", False):
            for key in ("expected_visual_state", "opposite_visual_state"):
                packet["claim_agent"].pop(key, None)
    ledger = list(dossier.get("evidence_catalog", []) or [])
    caption_tokens = set(re.findall(r"\w+", caption.casefold()))

    def priority(item):
        overlap = len(caption_tokens & set(re.findall(
            r"\w+", str(item.get("text", "")).casefold())))
        # No gold, prior label, predicted direction or phenomenon routing.
        tie = hashlib.sha256(str(item.get("id")).encode()).hexdigest()
        return (-overlap, tie)

    ordered = sorted(ledger, key=priority)
    detailed = ordered[:max(1, int(detailed_evidence_limit or 18))]
    detailed_ids = {item.get("id") for item in detailed}
    packet["evidence_ledger"] = [{k: deepcopy(item[k]) for k in
        ("id", "text", "type", "source", "grounded", "decision_grade", "provenance_roots") if k in item}
        for item in detailed]
    packet["remaining_evidence_index"] = [
        {"id": item.get("id"), "type": item.get("type")}
        for item in ordered if item.get("id") not in detailed_ids
    ]
    _update_rendering(packet, len(ledger))
    return packet


def compact_evidence_record(item):
    # Provenance remains in the durable ledger; exact text and IDs remain here.
    return {k: deepcopy(item[k]) for k in ("id", "text", "type", "source") if k in item}


def compact_evidence_packet(packet, evidence_first=True):
    """Remove repeated presentation metadata, never evidence or source content."""
    packet = deepcopy(packet)
    packet["evidence_ledger"] = [compact_evidence_record(i) for i in packet.get("evidence_ledger", [])]
    packet["claim_agent"] = {"claim_graph": packet.get("claim_agent", {}).get("claim_graph")}
    packet.pop("visual_agent", None)  # Its exact observations are already in the ledger.
    if evidence_first:
        for item in packet.get("candidate_cases", []):
            packet.setdefault("remaining_context_index", []).append({
                "context_id": item["context_id"], "kind": "fallible_candidate_argument", "requires_retrieval": True})
        packet["candidate_cases"] = []
    for role in packet.get("targeted_hearing", {}).values():
        if role.get("question_answers") is not None:
            # Compatibility aliases repeat the same latest answer.
            retained = {key: role[key] for key in ("question_answers", "status", "content_withheld") if key in role}
            role.clear()
            role.update(retained)
    # Render observations before interpretations without changing what is available.
    first = {key: packet[key] for key in ("source_caption", "evidence_ledger", "claim_agent") if key in packet}
    return dict(first, **{key: value for key, value in packet.items() if key not in first})


def _update_rendering(packet, active_count):
    detailed = packet.get("evidence_ledger", [])
    indexed = packet.get("remaining_evidence_index", [])
    packet["evidence_rendering"] = {
        "active_count": active_count, "detailed_count": len(detailed),
        "indexed_count": len(indexed),
        "all_active_ids_visible": len(detailed) + len(indexed) == active_count,
        **({} if packet.get("protocol") else {"citable_ids": [item["id"] for item in detailed]}),
    }


def fit_judge_packet(packet, prompt_builder, token_counter=None, max_tokens=3072, protected_ids=()):
    """Budget the actual rendered text, dropping complete optional records.

    With no tokenizer, UTF-8 bytes are a conservative bound for injected test
    runtimes. Production supplies its tokenizer. Never truncate the raw claim
    or retain a partial evidence quotation.
    """
    packet = deepcopy(packet)
    count = token_counter or (lambda value: len(value.encode("utf-8")))
    active_count = packet.get("evidence_rendering", {}).get("active_count", 0)
    removed = []
    mandatory_graph = deepcopy(packet.get("claim_agent", {}).get("claim_graph"))
    # Every active ID remains either fully visible or explicitly retrievable.
    catalog_ids = {item["id"] for item in packet.get("evidence_ledger", []) + packet.get("remaining_evidence_index", [])}
    context_ids = {item["context_id"] for key in ("candidate_cases", "context_records", "remaining_context_index")
                   for item in packet.get(key, []) if item.get("context_id")}
    while count(prompt_builder(packet)) > max_tokens:
        if packet.get("visual_agent"):
            packet["visual_agent"].pop(next(reversed(packet["visual_agent"])))
            removed.append("auxiliary_visual_summary")
        elif any(item.get("context_id") not in protected_ids for item in packet.get("candidate_cases", [])):
            index = next(i for i in reversed(range(len(packet["candidate_cases"])))
                         if packet["candidate_cases"][i].get("context_id") not in protected_ids)
            item = packet["candidate_cases"].pop(index)
            if item.get("context_id"):
                packet.setdefault("remaining_context_index", []).append(
                    {"context_id": item["context_id"], "kind": "candidate_argument", "requires_retrieval": True})
            removed.append("candidate_argument_moved_to_retrieval_index")
        elif any(item.get("context_id") not in protected_ids for item in packet.get("context_records", [])):
            index = next(i for i in reversed(range(len(packet["context_records"])))
                         if packet["context_records"][i].get("context_id") not in protected_ids)
            item = packet["context_records"].pop(index)
            packet.setdefault("remaining_context_index", []).append(
                {"context_id": item["context_id"], "kind": item["kind"], "requires_retrieval": True})
            removed.append("context_moved_to_retrieval_index")
        elif any(key != "claim_graph" for key in packet.get("claim_agent", {})):
            removable = [key for key in packet["claim_agent"] if key != "claim_graph"]
            packet["claim_agent"].pop(removable[-1])
            removed.append("derived_claim_field")
        elif len(packet.get("evidence_ledger", [])) > 1:
            removable = [i for i, item in enumerate(packet["evidence_ledger"])
                         if item.get("id") not in protected_ids]
            if not removable:
                raise ValueError("Requested evidence cannot fit the context budget")
            item = packet["evidence_ledger"].pop(removable[-1])
            packet.setdefault("remaining_evidence_index", []).append(
                    {"id": item["id"], "type": item.get("type")})
            removed.append("evidence_moved_to_retrieval_index")
        else:
            raise ValueError("context budget cannot fit mandatory caption, graph, hearing and evidence index")
        _update_rendering(packet, active_count)
    if mandatory_graph != packet.get("claim_agent", {}).get("claim_graph"):
        raise AssertionError("Mandatory claim graph changed during budgeting")
    visible_ids = {item["id"] for item in packet.get("evidence_ledger", []) + packet.get("remaining_evidence_index", [])}
    if visible_ids != catalog_ids:
        raise AssertionError("Active evidence disappeared during budgeting")
    remaining_context = {item["context_id"] for key in ("candidate_cases", "context_records", "remaining_context_index")
                         for item in packet.get(key, []) if item.get("context_id")}
    if remaining_context != context_ids:
        raise AssertionError("Context disappeared during budgeting")
    return packet, {"text_input_tokens": count(prompt_builder(packet)),
                    "mandatory_graph_preserved": mandatory_graph is not None,
                    "all_evidence_discoverable": True,
                    "all_context_discoverable": True,
                    "text_budget": max_tokens, "removed_fields": removed,
                    "counter": "tokenizer" if token_counter else "utf8_upper_bound"}


def retrieve_dossier_evidence(dossier, evidence_ids):
    """Resolve exact active IDs; callers must rebudget and disclose the packet."""
    by_id = {item["id"]: item for item in dossier.get("evidence_catalog", [])}
    ids = list(dict.fromkeys(evidence_ids))
    unknown = [item for item in ids if item not in by_id]
    if unknown:
        raise ValueError(f"Unknown or inactive requested evidence: {unknown}")
    return [deepcopy(by_id[item]) for item in ids]
