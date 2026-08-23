"""Structured, label-free bindings derived from Agent 1 observations."""

import re


PLACEHOLDERS = {
    "", "none", "n/a", "na", "unknown", "unclear", "unavailable",
    "not visible", "not specified",
}
REGIONS = (
    "top left", "top right", "bottom left", "bottom right",
    "left", "right", "top", "bottom", "center", "foreground", "background",
)
OBSERVATION_VERBS = (
    "is", "are", "has", "have", "holds", "hold", "wears", "wear",
    "shows", "show", "displays", "display", "contains", "contain",
    "reads", "read", "says", "say", "labels", "label", "points",
    "point", "faces", "face", "stands", "stand", "sits", "sit",
    "lies", "lie", "rises", "rise", "falls", "fall", "smiles",
    "smile", "cries", "cry",
)


def _clean(value):
    return " ".join(str(value or "").strip().strip("-*• ").split())


def _normalized(value):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", _clean(value).lower()).split())


def _usable(value):
    return _normalized(value) not in PLACEHOLDERS


def _region(text):
    normalized = _normalized(text)
    for region in REGIONS:
        if re.search(rf"\b{re.escape(region)}\b", normalized):
            return region
    return "unspecified"


def parse_binding_line(line, source_type="entity_state_binding"):
    """Parse one explicit or complete natural-language observation."""
    source_text = _clean(line)
    if not source_text:
        return None

    fields = {}
    for part in re.split(r"\s*\|\s*", source_text):
        match = re.match(
            r"(?i)^(entity|state|region|evidence)\s*[:=]\s*(.+)$", part
        )
        if match:
            fields[match.group(1).lower()] = _clean(match.group(2))

    entity = fields.get("entity", "")
    state = fields.get("state", "")
    region = fields.get("region", "") or _region(source_text)
    method = "explicit_agent1_binding"

    if not (entity and state):
        verbs = "|".join(re.escape(verb) for verb in OBSERVATION_VERBS)
        natural = re.match(
            rf"(?i)^(?P<entity>(?:(?:in|on|at)\s+the\s+)?"
            rf"(?:[a-z0-9'\-]+\s+){{0,7}}?[a-z0-9'\-]+)\s+"
            rf"(?P<verb>{verbs})\b\s*(?P<state>.+)$",
            source_text.strip(" ."),
        )
        if natural:
            entity = _clean(natural.group("entity"))
            verb = _clean(natural.group("verb"))
            remainder = _clean(natural.group("state"))
            state = remainder if verb in {"is", "are", "has", "have"} else (
                f"{verb} {remainder}".strip()
            )
            method = "parsed_complete_observation"

    if not (_usable(entity) and _usable(state)):
        return None
    return {
        "entity": entity,
        "state": state,
        "region": region if _usable(region) else "unspecified",
        "source_text": fields.get("evidence") or source_text,
        "source_type": source_type,
        "grounded": True,
        "complete": True,
        "method": method,
    }


def build_visual_bindings(parsed):
    """Build deduplicated bindings without assigning NLI direction."""
    parsed = parsed or {}
    sources = [
        ("entity_state_binding", parsed.get("entity_state_bindings", []) or []),
        ("visual_relation", parsed.get("visual_relations", []) or []),
        ("visual_fact", parsed.get("visual_facts", []) or []),
    ]
    bindings = []
    seen = set()
    for source_type, lines in sources:
        for line in lines:
            binding = parse_binding_line(line, source_type=source_type)
            if not binding:
                continue
            key = (
                _normalized(binding["entity"]),
                _normalized(binding["state"]),
                _normalized(binding["region"]),
            )
            if key in seen:
                continue
            seen.add(key)
            bindings.append(binding)
    return bindings
