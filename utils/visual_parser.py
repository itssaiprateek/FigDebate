import re

HEADINGS = [
    "Literal Scene",
    "People",
    "Objects",
    "Actions",
    "Environment",
    "Scene Type",
    "Visible Text",
    "Symbolic Elements",
    "Possible Visual Metaphors",
    "Visual Facts",
    "Visual Relations",
    "Entity-State Bindings",
    "Uncertain Observations",
    "Confidence",
]

# LLaVA occasionally renames a requested heading. Recognizing common aliases
# prevents a whole section from leaking into the preceding evidence field.
HEADING_ALIASES = {
    "Visible Relations": "Visual Relations",
    "Visual Relation": "Visual Relations",
    "Relationships": "Visual Relations",
    "Object Names and Phrases": "Objects",
    "Key Objects": "Objects",
    "Visual Objects": "Objects",
    "Overall Scene": "Literal Scene",
    "Visual Fact": "Visual Facts",
    "Observed Facts": "Visual Facts",
    "Observed Relations": "Visual Relations",
    "Text Bindings": "Visual Relations",
    "Entity State Bindings": "Entity-State Bindings",
    "Entity-State Binding": "Entity-State Bindings",
    "Symbolic Element": "Symbolic Elements",
    "Possible Visual Metaphor": "Possible Visual Metaphors",
    "Possible Visual Metaph": "Possible Visual Metaphors",
    "Uncertainties": "Uncertain Observations",
}

IGNORED_BOUNDARY_HEADINGS = set()


PLACEHOLDER_LINES = {
    "directly observed fact",
    "important visible object",
    "only genuinely uncertain detail, or none",
    "left right top bottom",
    "left right top bottom tilted rotated trend arrow contrast object to object",
    "actual spatial or directional relationships never yes no counts",
}

ABSENCE_LINES = {
    "none",
    "nothing",
    "no text",
    "no readable text",
    "no visible text",
    "not visible",
    "not applicable",
    "n a",
}


def _normalized_item(text):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def parse_list(text):
    """
    Converts bullet, numbered, comma-separated or newline-separated
    sections into a clean Python list.
    """

    if not text:
        return []

    text = text.strip()

    # The extraction prompt requires real OCR to be quoted or region-bound.
    # Bare absence words from a model must not become grounded image text.
    if (
        _normalized_item(text) in ABSENCE_LINES
        and not re.search(r"[\"']", text)
    ):
        return []

    items = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Remove bullets
        line = re.sub(r"^[-*\u2022]\s*", "", line)

        # Remove numbering
        line = re.sub(r"^\d+[\).\s]+", "", line)

        # A bullet is one observation. Splitting it on commas corrupts place
        # names and compound facts such as "Pittsburgh, Pennsylvania".
        items.append(line)

    cleaned = []
    seen = set()

    for item in items:

        normalized = _normalized_item(item)
        if (
            normalized in ABSENCE_LINES
            and not re.search(r"[\"']", item)
        ) or normalized in PLACEHOLDER_LINES:
            continue

        if normalized and normalized not in seen:
            # OCR decoders can loop over the same comma/semicolon-delimited
            # phrases until the token limit. Compact only clearly long runs;
            # ordinary text such as "Pittsburgh, Pennsylvania" is untouched.
            delimiter = None
            if item.count(";") >= 4:
                delimiter = ";"
            elif item.count(",") >= 8:
                delimiter = ","
            if delimiter:
                segments = []
                segment_seen = set()
                for segment in item.split(delimiter):
                    segment = segment.strip()
                    segment_key = _normalized_item(segment)
                    if segment_key and segment_key not in segment_seen:
                        segments.append(segment)
                        segment_seen.add(segment_key)
                    if len(segments) == 8:
                        break
                item = f"{delimiter} ".join(segments)
            cleaned.append(item)
            seen.add(normalized)

    # Greedy decoders sometimes emit a progressively longer version of the
    # same observation. Keep the most complete line so it cannot consume the
    # downstream evidence budget several times.
    compact = []
    normalized_items = [_normalized_item(item) for item in cleaned]
    for index, item in enumerate(cleaned):
        normalized = normalized_items[index]
        is_prefix_fragment = any(
            normalized != other
            and len(normalized.split()) >= 3
            and normalized in other
            for other in normalized_items
        )
        if not is_prefix_fragment:
            compact.append(item)

    return compact


def clean_scalar(text):
    lines = []
    for line in str(text or "").splitlines():
        line = re.sub(r"^[-*\u2022]\s*", "", line.strip())
        if line and line.lower() != "none":
            lines.append(line)
    return " ".join(lines)


def parse_confidence(text):

    match = re.search(r"([01](?:\.\d+)?)", text)

    if match:
        try:
            value = float(match.group(1))
            return max(0.0, min(1.0, value))
        except (TypeError, ValueError):
            pass

    return None


def parse_visual_response(response):

    result = {
        "literal_scene": "",
        "people": [],
        "objects": [],
        "actions": [],
        "environment": "",
        "scene_type": "",
        "visible_text": [],
        "symbolic_elements": [],
        "possible_visual_metaphors": [],
        "visual_facts": [],
        "visual_relations": [],
        "entity_state_bindings": [],
        "uncertain_observations": [],
        "confidence": None,
        "raw_output": response,
    }

    # Collect headings by their actual position rather than relying on a fixed
    # order. Agent 1 intentionally reports visible text before scene prose so
    # screenshots, charts, and memes retain their decision-critical content.
    matches = []
    heading_specs = (
        [(heading, heading) for heading in HEADINGS]
        + list(HEADING_ALIASES.items())
        + [(heading, None) for heading in sorted(IGNORED_BOUNDARY_HEADINGS)]
    )
    for observed_heading, canonical_heading in heading_specs:
        match = re.search(
            rf"^{re.escape(observed_heading)}"
            rf"(?:[ \t]*\([^:\r\n]*\))?[ \t]*(?::[ \t]*|$)",
            response,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if match:
            matches.append(
                (match.start(), match.end(), observed_heading, canonical_heading)
            )

    matches.sort()
    for index, (_, start, _, canonical_heading) in enumerate(matches):
        end = matches[index + 1][0] if index + 1 < len(matches) else len(response)
        value = response[start:end].strip()

        if canonical_heading is None:
            continue
        key = canonical_heading.lower().replace("-", "_").replace(" ", "_")

        if key == "confidence":

            result["confidence"] = parse_confidence(value)

        elif key in {
            "people",
            "objects",
            "actions",
            "visible_text",
            "symbolic_elements",
            "possible_visual_metaphors",
            "visual_facts",
            "visual_relations",
            "entity_state_bindings",
            "uncertain_observations",
        }:

            parsed_items = parse_list(value)
            result[key] = parse_list(
                "\n".join([*(result.get(key, []) or []), *parsed_items])
            )

        else:

            result[key] = clean_scalar(value)

    return result
