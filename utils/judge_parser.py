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
    # `issue` and `reason` are both explanatory text in this contract.  Reuse
    # existing model text when only the duplicate reason key is omitted; this
    # is schema normalization, not invented evidence or reasoning.
    if "reason" not in payload and isinstance(payload.get("issue"), str):
        if payload["issue"].strip():
            payload["reason"] = payload["issue"]
            normalized_fields.append("reason_from_issue")
    keys = set(payload)
    if "relation" in keys:
        expected_fields = TRIBUNAL_REVIEW_FIELDS
    else:
        expected_fields = LEGACY_TRIBUNAL_REVIEW_FIELDS
    missing = sorted(expected_fields - keys)
    unexpected = sorted(keys - expected_fields)
    if missing or unexpected:
        error = (
            "missing_fields:" + ",".join(missing) if missing
            else "unexpected_fields:" + ",".join(unexpected)
        )
        return parse_tribunal_review_response("") | {
            "_format_error": error, "_raw_output": str(raw_output or "")
        }
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
