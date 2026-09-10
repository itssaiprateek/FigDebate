"""Small, domain-neutral English pragmatic construction recognizer."""

from __future__ import annotations

import re


SCHEMAS = (
    (
        "EXEMPLAR_COMPARISON",
        re.compile(
            r"\bif\s+you\s+(?:ever\s+)?feel\s+.+?\b(?:think|remember)\s+"
            r"(?:about|of)\b",
            re.I,
        ),
    ),
    (
        "RHETORICAL_REJECTION",
        re.compile(r"\b(?:imagine|fancy)\s+(?:actually\s+)?(?:thinking|believing)\b", re.I),
    ),
    (
        "IRONIC_EVALUATION",
        re.compile(r"\b(?:great|nice|brilliant|wonderful)\s+(?:job|idea|work)\b", re.I),
    ),
    (
        "QUOTED_REACTION",
        re.compile(r"[\"“”].+[\"“”].+\b(?:said|says|reaction|responds?|replies?)\b", re.I),
    ),
)


def detect_pragmatic_schema(value):
    text = " ".join(str(value or "").split())
    for name, pattern in SCHEMAS:
        if pattern.search(text):
            return name
    return "NONE"
