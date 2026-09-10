"""Strict parser for the independent multimodal judge contract."""

import json
import re


VALID_VERDICTS = {"ENTAILS", "CONTRADICTS", "ABSTAIN"}
VALID_RELATIONS = {"SUPPORT", "CONFLICT", "UNRESOLVED"}
VERDICT_FOR_RELATION = {
    "SUPPORT": "ENTAILS", "CONFLICT": "CONTRADICTS",
    "UNRESOLVED": "ABSTAIN",
}
REQUIRED_FIELDS = {
    "verdict", "confidence", "evidence_ids", "visual_observations", "reason",
}
MEDIATION_FIELDS = {
    "status", "provisional_verdict", "confidence", "evidence_ids",
    "issue", "agent1_question", "agent2_question", "verification_request",
}
TRIBUNAL_REVIEW_FIELDS = {
    "status", "relation", "confidence", "evidence_ids",
    "visual_observations", "issue", "agent1_question", "agent2_question",
    "verification_request", "reason",
}
TRIBUNAL_REVIEW_V2_FIELDS = {
    "node_relations",
    "best_semantic_judgment", "relation", "admissibility",
    "visual_premise", "caption_premise", "semantic_bridge_type",
    "semantic_bridge", "evidence_ids", "counter_interpretation",
    "counter_interpretation_strength", "confidence",
    "requested_follow_up", "reason",
}
TRIBUNAL_REVIEW_V2_REQUIRED_FIELDS = {
    "best_semantic_judgment", "relation", "confidence", "evidence_ids",
    "reason",
}
TRIBUNAL_SYMMETRIC_FIELDS = {
    "support_case", "support_evidence_ids", "support_strength",
    "conflict_case", "conflict_evidence_ids", "conflict_strength",
}
VALID_BRIDGE_TYPES = {
    "AFFECTIVE_OPPOSITION", "LITERAL_INTENDED_POLARITY",
    "TEMPORAL_SEQUENCE", "CAUSE_EFFECT", "PARTICIPANT_ACTION_OUTCOME",
    "COMPARISON_DIRECTION", "QUOTED_STATEMENT_REACTION",
    "SYMBOL_TARGET_ATTACHMENT", "OBJECT_FUNCTION", "SOCIAL_CONVENTION",
    "HUMOR_INCONGRUITY", "GENERAL_SEMANTIC_RELATION",
}
LEGACY_TRIBUNAL_REVIEW_FIELDS = (
    TRIBUNAL_REVIEW_FIELDS - {"relation"}
) | {"provisional_verdict"}
LABEL_LEAK_PATTERN = re.compile(
    r"\b(?:entails?|entailment|contradicts?|contradiction|"
    r"supports?|conflicts?|verdict|prediction|final\s+label|correct\s+label)\b",
    flags=re.IGNORECASE,
)


def _questions_are_label_blind(*questions):
    return not any(
        LABEL_LEAK_PATTERN.search(str(question or ""))
        for question in questions
    )


def _invalid(raw_output, error):
    return {
        "verdict": "ABSTAIN",
        "confidence": 0.0,
        "evidence_ids": [],
        "visual_observations": [],
        "reason": "",
        "_format_valid": False,
        "_format_error": error,
        "_raw_output": raw_output or "",
    }


def _json_object(raw_output):
    text = str(raw_output or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        raise ValueError("response_is_not_a_json_object")
    value, end = json.JSONDecoder().raw_decode(text)
    if text[end:].strip():
        raise ValueError("text_after_json_object")
    if not isinstance(value, dict):
        raise ValueError("response_is_not_a_json_object")
    return value


def parse_judge_response(raw_output):
    """Return a normalized judgment and never infer missing contract fields."""
    try:
        payload = _json_object(raw_output)
    except (ValueError, json.JSONDecodeError) as error:
        return _invalid(raw_output, str(error))

    missing = sorted(REQUIRED_FIELDS - set(payload))
    unexpected = sorted(set(payload) - REQUIRED_FIELDS)
    if missing:
        return _invalid(raw_output, "missing_fields:" + ",".join(missing))
    if unexpected:
        return _invalid(raw_output, "unexpected_fields:" + ",".join(unexpected))

    verdict = str(payload.get("verdict", "")).strip().upper()
    if verdict not in VALID_VERDICTS:
        return _invalid(raw_output, "invalid_verdict")

    confidence = payload.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return _invalid(raw_output, "invalid_confidence")

    evidence_ids = payload.get("evidence_ids")
    observations = payload.get("visual_observations")
    reason = payload.get("reason")
    if not isinstance(evidence_ids, list) or not all(
        isinstance(item, str) for item in evidence_ids
    ):
        return _invalid(raw_output, "invalid_evidence_ids")
    if not isinstance(observations, list) or not all(
        isinstance(item, str) for item in observations
    ):
        return _invalid(raw_output, "invalid_visual_observations")
    if not isinstance(reason, str) or not reason.strip():
        return _invalid(raw_output, "invalid_reason")

    normalized_ids = []
    for item in evidence_ids[:12]:
        item_id = item.strip().upper()
        if item_id and item_id not in normalized_ids:
            normalized_ids.append(item_id)
    normalized_observations = [
        item.strip()[:500] for item in observations[:6] if item.strip()
    ]
    return {
        "verdict": verdict,
        "confidence": float(confidence),
        "evidence_ids": normalized_ids,
        "visual_observations": normalized_observations,
        "reason": reason.strip()[:2000],
        "_format_valid": True,
        "_format_error": "",
        "_raw_output": str(raw_output or ""),
    }


def _invalid_mediation(raw_output, error):
    return {
        "status": "ABSTAIN",
        "provisional_verdict": "ABSTAIN",
        "confidence": 0.0,
        "evidence_ids": [],
        "disputed_issues": [],
        "agent1_questions": [],
        "agent2_questions": [],
        "verification_requests": [],
        "reason": "",
        "_format_valid": False,
        "_format_error": error,
        "_raw_output": raw_output or "",
    }


def parse_mediation_response(raw_output):
    """Validate a label-blind debate-mediation plan without filling gaps."""
    try:
        payload = _json_object(raw_output)
    except (ValueError, json.JSONDecodeError) as error:
        return _invalid_mediation(raw_output, str(error))

    missing = sorted(MEDIATION_FIELDS - set(payload))
    unexpected = sorted(set(payload) - MEDIATION_FIELDS)
    if missing:
        return _invalid_mediation(raw_output, "missing_fields:" + ",".join(missing))
    if unexpected:
        return _invalid_mediation(
            raw_output, "unexpected_fields:" + ",".join(unexpected)
        )

    status = str(payload.get("status", "")).strip().upper()
    verdict = str(payload.get("provisional_verdict", "")).strip().upper()
    confidence = payload.get("confidence")
    if status not in {"MEDIATE", "ABSTAIN"}:
        return _invalid_mediation(raw_output, "invalid_status")
    if verdict not in VALID_VERDICTS:
        return _invalid_mediation(raw_output, "invalid_provisional_verdict")
    if status == "ABSTAIN" and verdict != "ABSTAIN":
        return _invalid_mediation(raw_output, "abstention_requires_abstain_verdict")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return _invalid_mediation(raw_output, "invalid_confidence")

    if not isinstance(payload.get("evidence_ids"), list) or not all(
        isinstance(item, str) for item in payload["evidence_ids"]
    ):
        return _invalid_mediation(raw_output, "invalid_evidence_ids")
    text_fields = (
        "issue", "agent1_question", "agent2_question", "verification_request",
    )
    if any(not isinstance(payload.get(field), str) for field in text_fields):
        return _invalid_mediation(raw_output, "invalid_text_field")
    issue = payload["issue"].strip()
    if not issue:
        return _invalid_mediation(raw_output, "mediation_requires_issue")
    if status == "MEDIATE" and not any(
        payload.get(field, "").strip()
        for field in (
            "agent1_question", "agent2_question", "verification_request"
        )
    ):
        return _invalid_mediation(raw_output, "mediation_requires_targeted_question")
    if not _questions_are_label_blind(
        payload.get("agent1_question"), payload.get("agent2_question")
    ):
        return _invalid_mediation(raw_output, "mediation_question_exposes_label")

    evidence_ids = []
    for item in payload["evidence_ids"][:16]:
        item_id = item.strip().upper()
        if item_id and item_id not in evidence_ids:
            evidence_ids.append(item_id)

    def one_item(name):
        value = payload[name].strip()[:600]
        return [value] if value else []

    return {
        "status": status,
        "provisional_verdict": verdict,
        "confidence": float(confidence),
        "evidence_ids": evidence_ids,
        "disputed_issues": [issue[:600]],
        "agent1_questions": one_item("agent1_question"),
        "agent2_questions": one_item("agent2_question"),
        "verification_requests": one_item("verification_request"),
        "reason": issue[:2000],
        "_format_valid": True,
        "_format_error": "",
        "_raw_output": str(raw_output or ""),
    }


def parse_tribunal_review_response(raw_output):
    """Parse a mediator review after both agents have answered."""
    try:
        payload = _json_object(raw_output)
    except (ValueError, json.JSONDecodeError) as error:
        payload = None
        parse_error = str(error)
    if payload is None:
        return {
            "status": "ABSTAIN", "relation": "UNRESOLVED",
            "provisional_verdict": "ABSTAIN",
            "confidence": 0.0, "evidence_ids": [],
            "visual_observations": [], "issue": "", "agent1_questions": [],
            "agent2_questions": [], "verification_requests": [], "reason": "",
            "_format_valid": False, "_format_error": parse_error,
            "_raw_output": str(raw_output or ""),
        }
    normalized_fields = []
    # Current generation has one decision variable. Retain V2 reading only for
    # archived runs; never normalize conflicting explicitly supplied decisions.
    if "best_semantic_judgment" not in payload and "semantic_bridge_type" in payload:
        from engine.output_contracts import schema_for_contract, validate_shape
        if "context_requests" not in payload:
            payload["context_requests"] = []
            normalized_fields.append("legacy_context_requests_absent")
        if not validate_shape(payload, schema_for_contract("tribunal_review")):
            return parse_tribunal_review_response("") | {
                "_format_error": "invalid_v3_resolution", "_raw_output": str(raw_output or "")}
        payload["best_semantic_judgment"] = VERDICT_FOR_RELATION[payload["relation"]]
        normalized_fields.append("verdict_derived_from_relation_v3")
    if "best_semantic_judgment" in payload:
        missing = sorted(TRIBUNAL_REVIEW_V2_REQUIRED_FIELDS - set(payload))
        unexpected = sorted(
            set(payload) - TRIBUNAL_REVIEW_V2_FIELDS - TRIBUNAL_SYMMETRIC_FIELDS - {"context_requests"}
        )
        if missing:
            error = "missing_fields:" + ",".join(missing)
            return parse_tribunal_review_response("") | {
                "_format_error": error, "_raw_output": str(raw_output or "")
            }
        if unexpected:
            normalized_fields.extend(
                f"ignored_extra_field:{field}" for field in unexpected
            )
        defaults = {
            "admissibility": "INSUFFICIENT",
            "visual_premise": "",
            "caption_premise": "",
            "semantic_bridge_type": "GENERAL_SEMANTIC_RELATION",
            "semantic_bridge": "",
            "counter_interpretation": "",
            "counter_interpretation_strength": 0.0,
            "requested_follow_up": "NONE",
        }
        for field, default in defaults.items():
            if field not in payload:
                payload[field] = default
                normalized_fields.append(f"defaulted_optional_field:{field}")
        if isinstance(payload.get("evidence_ids"), str):
            payload["evidence_ids"] = re.findall(
                r"\b[A-Za-z]{1,4}[-_ ]?\d{1,5}\b",
                payload["evidence_ids"],
            )
            normalized_fields.append("evidence_ids_from_string")
        for field in ("confidence", "counter_interpretation_strength"):
            value = payload.get(field)
            if isinstance(value, str):
                try:
                    payload[field] = float(value.strip())
                except ValueError:
                    pass
                else:
                    normalized_fields.append(f"numeric_string:{field}")
        judgment = str(payload["best_semantic_judgment"]).strip().upper()
        relation = str(payload["relation"]).strip().upper()
        admissibility = str(payload["admissibility"]).strip().upper()
        follow_up = str(payload["requested_follow_up"]).strip().upper()
        bridge_type = str(payload["semantic_bridge_type"]).strip().upper()
        valid_followups = {
            "NONE", "VISUAL_PREMISE", "CAPTION_PREMISE", "ENTITY_BINDING",
            "SCOPE_BINDING", "COUNTER_INTERPRETATION",
        }
        if judgment not in VALID_VERDICTS:
            error = "invalid_best_semantic_judgment"
        elif relation not in VALID_RELATIONS:
            error = "invalid_relation"
        elif relation != {
            "ENTAILS": "SUPPORT", "CONTRADICTS": "CONFLICT",
            "ABSTAIN": "UNRESOLVED",
        }[judgment]:
            error = "judgment_relation_mismatch"
        else:
            error = ""
        if not error and admissibility not in {
            "VERIFIED", "CORROBORATED", "PLAUSIBLE", "INSUFFICIENT"
        }:
            payload["admissibility"] = "INSUFFICIENT"
            admissibility = "INSUFFICIENT"
            normalized_fields.append("invalid_admissibility_to_insufficient")
        if not error and follow_up not in valid_followups:
            payload["requested_follow_up"] = "NONE"
            follow_up = "NONE"
            normalized_fields.append("invalid_follow_up_to_none")
        if not error and bridge_type not in VALID_BRIDGE_TYPES:
            payload["semantic_bridge_type"] = "GENERAL_SEMANTIC_RELATION"
            bridge_type = "GENERAL_SEMANTIC_RELATION"
            normalized_fields.append("unknown_bridge_type_ignored")
        numeric_fields = ("confidence", "counter_interpretation_strength")
        if not error and any(
            isinstance(payload[field], bool)
            or not isinstance(payload[field], (int, float))
            or not 0.0 <= float(payload[field]) <= 1.0
            for field in numeric_fields
        ):
            error = "invalid_numeric_field"
        symmetric_present = bool(set(payload) & TRIBUNAL_SYMMETRIC_FIELDS)
        if not error and symmetric_present:
            if not TRIBUNAL_SYMMETRIC_FIELDS.issubset(payload):
                error = "incomplete_symmetric_cases"
            elif (
                not isinstance(payload.get("support_case"), str)
                or not isinstance(payload.get("conflict_case"), str)
                or not isinstance(payload.get("support_evidence_ids"), list)
                or not isinstance(payload.get("conflict_evidence_ids"), list)
                or any(
                    isinstance(payload.get(field), bool)
                    or not isinstance(payload.get(field), (int, float))
                    or not 0.0 <= float(payload.get(field)) <= 1.0
                    for field in ("support_strength", "conflict_strength")
                )
            ):
                error = "invalid_symmetric_cases"
        context_requests = payload.get("context_requests", [])
        if not error and (not isinstance(context_requests, list)
                          or any(not isinstance(item, str) for item in context_requests)):
            error = "invalid_context_requests"
        text_fields = (
            "visual_premise", "caption_premise", "semantic_bridge",
            "counter_interpretation", "reason",
        )
        if not error and (
            not isinstance(payload.get("evidence_ids"), list)
            or not all(isinstance(item, str) for item in payload["evidence_ids"])
            or any(not isinstance(payload.get(field), str) for field in text_fields)
        ):
            error = "invalid_field_type"
        if not error and not payload["reason"].strip():
            error = "missing_reason"
        if error:
            return parse_tribunal_review_response("") | {
                "_format_error": error, "_raw_output": str(raw_output or "")
            }
        if admissibility in {"VERIFIED", "CORROBORATED"} and relation != "UNRESOLVED":
            status = "RESOLVE"
        elif follow_up != "NONE":
            status = "FOLLOW_UP"
        else:
            status = "ABSTAIN"
        ids = list(dict.fromkeys(
            item.strip().upper() for item in payload["evidence_ids"][:12]
            if item.strip()
        ))
        visual_premise = payload["visual_premise"].strip()[:800]
        return {
            "status": status,
            "relation": relation,
            "provisional_verdict": judgment,
            "best_semantic_judgment": judgment,
            "node_relations": payload.get("node_relations", []),
            "context_requests": list(dict.fromkeys(context_requests)),
            "admissibility": admissibility,
            "confidence": float(payload["confidence"]),
            "evidence_ids": ids,
            "visual_observations": [visual_premise] if visual_premise else [],
            "visual_premise": visual_premise,
            "caption_premise": payload["caption_premise"].strip()[:800],
            "semantic_bridge_type": bridge_type,
            "semantic_bridge": payload["semantic_bridge"].strip()[:1200],
            "counter_interpretation": payload["counter_interpretation"].strip()[:800],
            "counter_interpretation_strength": float(payload["counter_interpretation_strength"]),
            "support_case": str(payload.get("support_case") or "").strip()[:1000],
            "support_evidence_ids": list(dict.fromkeys(
                str(item).strip().upper()
                for item in payload.get("support_evidence_ids", [])[:12]
                if str(item).strip()
            )),
            "support_strength": (
                float(payload["support_strength"]) if symmetric_present else None
            ),
            "conflict_case": str(payload.get("conflict_case") or "").strip()[:1000],
            "conflict_evidence_ids": list(dict.fromkeys(
                str(item).strip().upper()
                for item in payload.get("conflict_evidence_ids", [])[:12]
                if str(item).strip()
            )),
            "conflict_strength": (
                float(payload["conflict_strength"]) if symmetric_present else None
            ),
            "symmetric_cases_present": symmetric_present,
            "requested_follow_up": follow_up,
            "issue": payload["reason"].strip()[:1000],
            "agent1_questions": [], "agent2_questions": [],
            "verification_requests": [],
            "reason": payload["reason"].strip()[:2000],
            "_format_valid": True, "_format_error": "",
            "_contract_version": "2.0-tolerant",
            "_normalized_fields": normalized_fields,
            "_raw_output": str(raw_output or ""),
        }
    # `issue` and `reason` are both explanatory text in this contract.  Reuse
    # existing model text when only the duplicate reason key is omitted; this
    # is schema normalization, not invented evidence or reasoning.
    if "reason" not in payload and isinstance(payload.get("issue"), str):
        if payload["issue"].strip():
            payload["reason"] = payload["issue"]
            normalized_fields.append("reason_from_issue")
    if "issue" not in payload and isinstance(payload.get("reason"), str):
        if payload["reason"].strip():
            payload["issue"] = payload["reason"]
            normalized_fields.append("issue_from_reason")
    for field, default in {
        "visual_observations": [],
        "agent1_question": "",
        "agent2_question": "",
        "verification_request": "",
    }.items():
        if field not in payload:
            payload[field] = default
            normalized_fields.append(f"defaulted_optional_field:{field}")
    if isinstance(payload.get("evidence_ids"), str):
        payload["evidence_ids"] = re.findall(
            r"\b[A-Za-z]{1,4}[-_ ]?\d{1,5}\b",
            payload["evidence_ids"],
        )
        normalized_fields.append("evidence_ids_from_string")
    if isinstance(payload.get("visual_observations"), str):
        payload["visual_observations"] = [payload["visual_observations"]]
        normalized_fields.append("visual_observations_from_string")
    if isinstance(payload.get("confidence"), str):
        try:
            payload["confidence"] = float(payload["confidence"].strip())
        except ValueError:
            pass
        else:
            normalized_fields.append("numeric_string:confidence")
    keys = set(payload)
    if "relation" in keys:
        expected_fields = TRIBUNAL_REVIEW_FIELDS
    else:
        expected_fields = LEGACY_TRIBUNAL_REVIEW_FIELDS
    missing = sorted(expected_fields - keys)
    unexpected = sorted(keys - expected_fields)
    if missing:
        error = "missing_fields:" + ",".join(missing)
        return parse_tribunal_review_response("") | {
            "_format_error": error, "_raw_output": str(raw_output or "")
        }
    normalized_fields.extend(
        f"ignored_extra_field:{field}" for field in unexpected
    )
    status = str(payload["status"]).strip().upper()
    relation = str(payload.get("relation", "")).strip().upper()
    verdict = str(payload.get("provisional_verdict", "")).strip().upper()
    if relation:
        if relation not in VALID_RELATIONS:
            return parse_tribunal_review_response("") | {
                "_format_error": "invalid_relation", "_raw_output": str(raw_output or "")
            }
        verdict = VERDICT_FOR_RELATION[relation]
    else:
        relation = {
            "ENTAILS": "SUPPORT", "CONTRADICTS": "CONFLICT",
            "ABSTAIN": "UNRESOLVED",
        }.get(verdict, "")
    confidence = payload["confidence"]
    if status not in {"RESOLVE", "FOLLOW_UP", "ABSTAIN"}:
        return parse_tribunal_review_response("") | {
            "_format_error": "invalid_status", "_raw_output": str(raw_output or "")
        }
    if verdict not in VALID_VERDICTS:
        return parse_tribunal_review_response("") | {
            "_format_error": "invalid_provisional_verdict",
            "_raw_output": str(raw_output or ""),
        }
    if status != "RESOLVE" and relation != "UNRESOLVED":
        return parse_tribunal_review_response("") | {
            "_format_error": "non_resolution_requires_abstain",
            "_raw_output": str(raw_output or ""),
        }
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        return parse_tribunal_review_response("") | {
            "_format_error": "invalid_confidence", "_raw_output": str(raw_output or "")
        }
    list_fields = ("evidence_ids", "visual_observations")
    text_fields = (
        "issue", "agent1_question", "agent2_question",
        "verification_request", "reason",
    )
    if any(
        not isinstance(payload.get(field), list)
        or not all(isinstance(item, str) for item in payload[field])
        for field in list_fields
    ) or any(not isinstance(payload.get(field), str) for field in text_fields):
        return parse_tribunal_review_response("") | {
            "_format_error": "invalid_field_type", "_raw_output": str(raw_output or "")
        }
    if not payload["issue"].strip() or not payload["reason"].strip():
        return parse_tribunal_review_response("") | {
            "_format_error": "missing_issue_or_reason",
            "_raw_output": str(raw_output or ""),
        }
    if status == "FOLLOW_UP" and not (
        payload["agent1_question"].strip() or payload["agent2_question"].strip()
    ):
        return parse_tribunal_review_response("") | {
            "_format_error": "follow_up_requires_question",
            "_raw_output": str(raw_output or ""),
        }
    if not _questions_are_label_blind(
        payload.get("agent1_question"), payload.get("agent2_question")
    ):
        return parse_tribunal_review_response("") | {
            "_format_error": "tribunal_question_exposes_label",
            "_raw_output": str(raw_output or ""),
        }
    ids = list(dict.fromkeys(
        item.strip().upper() for item in payload["evidence_ids"][:16]
        if item.strip()
    ))
    one = lambda field: [payload[field].strip()[:600]] if payload[field].strip() else []
    return {
        "status": status,
        "relation": relation,
        "provisional_verdict": verdict,
        "confidence": float(confidence),
        "evidence_ids": ids,
        "visual_observations": [
            item.strip()[:600] for item in payload["visual_observations"][:6]
            if item.strip()
        ],
        "issue": payload["issue"].strip()[:1000],
        "agent1_questions": one("agent1_question"),
        "agent2_questions": one("agent2_question"),
        "verification_requests": one("verification_request"),
        "reason": payload["reason"].strip()[:2000],
        "_format_valid": True,
        "_format_error": "",
        "_normalized_fields": normalized_fields,
        "_raw_output": str(raw_output or ""),
    }
