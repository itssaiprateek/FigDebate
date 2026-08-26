"""Strict parser for the independent multimodal judge contract."""

import json
import re


VALID_VERDICTS = {"ENTAILS", "CONTRADICTS", "ABSTAIN"}
REQUIRED_FIELDS = {
    "verdict", "confidence", "evidence_ids", "visual_observations", "reason",
}
MEDIATION_FIELDS = {
    "status", "provisional_verdict", "confidence", "evidence_ids",
    "issue", "agent1_question", "agent2_question", "verification_request",
}


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
