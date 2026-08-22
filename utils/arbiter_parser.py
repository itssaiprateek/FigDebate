import re

HEADINGS = {
    "evidence_summary": ["Evidence Summary"],
    "visual_support": ["Visual Support"],
    "contradictions": ["Contradictions"],
    "missing_evidence": ["Missing Evidence"],
    "reasoning": ["Reasoning"],
    "label": ["Final Decision"],
    "confidence": ["Confidence"],
}

def clean_text(text):

    if text is None:
        return ""

    text = text.replace("\r", "")
    text = re.sub(r"-{5,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def unique(items):

    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def parse_list(text):

    text = clean_text(text)

    if not text:
        return []

    items = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        line = re.sub(r"^[-\u2022*]\s*", "", line)
        line = re.sub(r"^\d+[\).\s]+", "", line)

        if line.lower() in {"none", "none.", "n/a"}:
            continue

        items.append(line)

    return unique(items)


def parse_confidence(text):

    text = clean_text(text)

    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])", text)

    if m:
        value = float(m.group(1))
        return value if 0.0 <= value <= 1.0 else None

    m = re.search(r"(\d+)\s*%", text)

    if m:
        value = float(m.group(1))
        return round(value / 100, 2) if 0.0 <= value <= 100.0 else None

    return None


def parse_label(text):
    """
    Accept a label only when the Final Decision field contains exactly one
    valid binary decision. Do not accept labels embedded in explanations,
    examples, or template text.
    """
    text = clean_text(text).upper()

    if re.fullmatch(r"(ENTAILS|CONTRADICTS)", text):
        return text

    return None


def parse_arbiter_response(response):

    response = clean_text(response)

    result = {
        "evidence_summary": "",
        "visual_support": [],
        "contradictions": [],
        "missing_evidence": [],
        "reasoning": "",
        "label": None,
        "confidence": None,
        "raw_output": response,
    }

    matches = []

    for field, aliases in HEADINGS.items():

        for alias in aliases:

            m = re.search(
                rf"^{re.escape(alias)}\s*:",
                response,
                flags=re.MULTILINE | re.IGNORECASE,
            )

            if m:
                matches.append((m.start(), m.end(), field))
                break

    matches.sort()

    for i in range(len(matches)):

        _, start, field = matches[i]

        end = matches[i + 1][0] if i + 1 < len(matches) else len(response)

        value = clean_text(response[start:end])

        if field == "confidence":
            result[field] = parse_confidence(value)

        elif field == "label":
            result[field] = parse_label(value)

        elif field in {"visual_support", "contradictions", "missing_evidence"}:
            result[field] = parse_list(value)

        else:
            result[field] = value

    # A label is valid only when it was parsed from the dedicated
    # "Final Decision:" section above. Do not search the whole response:
    # words such as "ENTAILS" and "CONTRADICTS" can appear in reasoning,
    # examples, or instructions and are not a final prediction.
    return result
