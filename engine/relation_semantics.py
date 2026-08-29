"""Shared evidence semantics used by every decision and revision path."""

import re


MISSING_ONLY_RE = re.compile(
    r"^(?:there (?:is|are) )?(?:no (?:clear )?(?:evidence|indication|sign)|"
    r"not (?:shown|visible|present)|nothing (?:shows|indicates)|"
    r"cannot be (?:seen|determined)|unclear|absent|missing)\b",
    flags=re.IGNORECASE,
)


def is_missing_evidence_only(value):
    """True only for an absence/uncertainty statement with no affirmative cue."""
    text = " ".join(str(value or "").split()).strip()
    if not text or not MISSING_ONLY_RE.search(text):
        return False
    return not re.search(
        r"\b(?:but|however|instead|while|whereas|shows?|displays?|reads?|"
        r"rises?|falls?|increases?|decreases?|breaks?|smiles?|frowns?)\b",
        text,
        flags=re.IGNORECASE,
    )


def relation_from_evidence_status(status):
    """Map only verified directions; all missing/unknown states stay neutral."""
    normalized = str(status or "").strip().upper()
    if normalized in {"SUPPORTED", "SUPPORT"}:
        return "SUPPORT"
    if normalized in {"CONFLICTING", "CONFLICT"}:
        return "CONFLICT"
    return "NEUTRAL"
