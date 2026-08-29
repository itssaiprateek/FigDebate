"""Deterministic typed observations derived from Agent 1's visual record.

These records describe what was observed and where. They never assign a
dataset label; directional promotion remains an independent verification step.
"""

import re


REGION_RE = re.compile(
    r"\b(upper left|upper right|lower left|lower right|left|right|top|bottom|"
    r"first|second|third|panel\s*\d+|center|foreground|background)\b",
    flags=re.IGNORECASE,
)
REACTION_RE = re.compile(
    r"\b(smiles?|laughs?|frowns?|cries|angry|surprised|shocked|worried|"
    r"frustrated|reacts?|looks at|points at)\b",
    flags=re.IGNORECASE,
)
EVENT_RE = re.compile(
    r"\b(hits?|pushes?|pulls?|drops?|falls?|breaks?|causes?|blames?|"
    r"before|after|then|later|next)\b",
    flags=re.IGNORECASE,
)
ATTACHMENT_RE = re.compile(
    r"\b(attached to|worn by|held by|above|on top of|belongs to|labels?)\b",
    flags=re.IGNORECASE,
)


def _clean(value):
    return " ".join(str(value or "").split()).strip()


def _regions(text):
    return list(dict.fromkeys(
        match.group(1).casefold() for match in REGION_RE.finditer(text)
    ))


def _record(kind, text, *, regions=None, source_field="", binding_complete=False):
    return {
        "record_type": kind,
        "text": _clean(text),
        "regions": list(regions or []),
        "source_field": source_field,
        "binding_complete": bool(binding_complete),
        "directional": False,
    }


def build_structured_observations(visual_output):
    """Extract conservative OCR, layout, event, reaction, and attachment records."""
    visual_output = visual_output or {}
    records = []
    for raw in visual_output.get("visible_text", []) or []:
        text = _clean(raw)
        if not text:
            continue
        left, separator, right = text.partition("=>")
        if not separator:
            match = re.match(
                r"^([^:]{1,80}\b(?:panel|side|region|object|sign|label|"
                r"bottle|screen|poster|person)\b[^:]*)\s*:\s*(.+)$",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                left, right = match.group(1), match.group(2)
                separator = ":"
        regions = _regions(left if separator else text)
        records.append(_record(
            "OCR_REGION_BINDING" if separator and _clean(right) else "OCR_TEXT",
            text,
            regions=regions,
            source_field="visible_text",
            binding_complete=bool(separator and _clean(left) and _clean(right)),
        ))

    relations = list(visual_output.get("visual_relations", []) or [])
    facts = list(visual_output.get("visual_facts", []) or [])
    for raw, source_field in (
        [(item, "visual_relations") for item in relations]
        + [(item, "visual_facts") for item in facts]
    ):
        text = _clean(raw)
        if not text:
            continue
        regions = _regions(text)
        if EVENT_RE.search(text) or len(regions) >= 2:
            kind = "PANEL_EVENT_OR_COMPARISON"
        elif REACTION_RE.search(text):
            kind = "REACTION_CUE"
        elif ATTACHMENT_RE.search(text):
            kind = "SYMBOL_OR_TEXT_ATTACHMENT"
        elif regions:
            kind = "SPATIAL_BINDING"
        else:
            continue
        records.append(_record(
            kind,
            text,
            regions=regions,
            source_field=source_field,
            binding_complete=bool(regions or ATTACHMENT_RE.search(text)),
        ))

    unique = []
    seen = set()
    for record in records:
        key = (record["record_type"], record["text"].casefold())
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def structural_summary(records):
    counts = {}
    for record in records or []:
        kind = record.get("record_type", "UNKNOWN")
        counts[kind] = counts.get(kind, 0) + 1
    return counts
