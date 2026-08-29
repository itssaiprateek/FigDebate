import re

# ==========================================================
# Headings used by Agent 2
# ==========================================================

HEADINGS = {
    "literal_meaning": [
        "Literal Meaning",
    ],

    "key_entities": [
        "Key Entities",
    ],

    "key_actions": [
        "Key Actions",
    ],

    "non_literal_expressions": [
        "Non-literal Expressions",
    ],

    "underlying_message": [
        "Underlying Message",
    ],

    "explicit_claims": [
        "Explicit Claims",
    ],

    "caption_proposition": [
        "Caption Proposition",
    ],

    "claim_subject": ["Claim Subject"],

    "claim_predicate": ["Claim Predicate"],

    "claim_object": ["Claim Object"],

    "claim_source": ["Claim Source"],

    "claim_target": ["Claim Target"],

    "asserted_property": ["Asserted Property"],

    "transferred_property": ["Transferred Property"],

    "incongruity": ["Incongruity"],

    "caption_polarity": ["Caption Polarity"],

    "alternative_interpretation": ["Alternative Interpretation"],

    "relation_family": ["Relation Family"],

    "expected_visual_state": ["Expected Visual State"],

    "opposite_visual_state": ["Opposite Visual State"],

    "reasoning_requirement": ["Reasoning Requirement"],

    "structural_reasoning_type": ["Structural Reasoning Type"],

    "figurative_mechanism_candidates": ["Figurative Mechanism Candidates"],

    "literal_polarity": ["Literal Polarity"],

    "intended_polarity": ["Intended Polarity"],

    "comparison_direction": ["Comparison Direction"],

    "evaluation_target": ["Evaluation Target"],

    "time_or_panel_scope": ["Time or Panel Scope"],

    "implicit_claims": [
        "Implicit Claims",
    ],

        "linguistic_notes": [
        "Linguistic Notes",
    ],

    "figurative_type": [
        "Figurative Type",
    ],

    "linguistic_cue": [
        "Linguistic Cue",
    ],

    "polarity_reversal": [
        "Polarity Reversal",
    ],

    "background_knowledge": [
        "Background Knowledge",
    ],

    "confidence": [
        "Confidence",
    ],
}


# ==========================================================
# Utility
# ==========================================================

def clean_text(text):

    if text is None:
        return ""

    text = text.replace("\r", "")

    # remove long separator lines
    text = re.sub(r"-{5,}", "", text)

    # collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ==========================================================
# Remove duplicates
# ==========================================================

def unique(items):

    seen = set()
    result = []

    for item in items:

        if item not in seen:

            seen.add(item)
            result.append(item)

    return result


# ==========================================================
# Parse bullet lists
# ==========================================================

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

        lower = line.lower()

        if lower in {
            "none",
            "none.",
            "n/a",
            "no",
            "no.",
        }:
            continue

        items.append(line)

    return unique(items)


# ==========================================================
# Parse confidence
# ==========================================================

def parse_confidence(text):

    text = clean_text(text)

    # decimal

    m = re.search(r"\b([01](?:\.\d+)?)\b", text)

    if m:
        value = float(m.group(1))
        return max(0.0, min(value, 1.0))

    # fraction

    m = re.search(r"(\d+)\s*/\s*(\d+)", text)

    if m:

        num = float(m.group(1))
        den = float(m.group(2))

        if den != 0:
            return round(num / den, 2)

    # percentage

    m = re.search(r"(\d+)\s*%", text)

    if m:

        return round(min(float(m.group(1))/100,1.0),2)

    return None
# ==========================================================
# Parse Non-literal Expressions
# ==========================================================

def parse_non_literal_expressions(text):

    text = clean_text(text)

    if not text:
        return []

    if text.lower() in {
        "none",
        "none.",
        "n/a",
    }:
        return []

    expressions = []
    blocks = re.split(r"\n\s*[-\u2022*]\s*", "\n" + text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        fields = {"phrase": "", "literal": "", "contextual": "", "reason": ""}
        current = None
        recognized_heading = False

        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue

            heading = re.match(
                r"^(phrase|literal interpretation|contextual interpretation|reason)\s*:\s*(.*)$",
                line,
                flags=re.IGNORECASE,
            )
            if heading:
                recognized_heading = True
                heading_name = heading.group(1).lower()
                current = {
                    "phrase": "phrase",
                    "literal interpretation": "literal",
                    "contextual interpretation": "contextual",
                    "reason": "reason",
                }[heading_name]
                fields[current] = heading.group(2).strip()
                continue

            if current:
                fields[current] = " ".join(
                    part for part in (fields[current], line) if part
                )

        if recognized_heading:
            if fields["phrase"]:
                expressions.append({
                    "expression": fields["phrase"],
                    "literal": fields["literal"],
                    "contextual": fields["contextual"],
                    "reason": fields["reason"],
                })
            continue

        if ":" in block:
            expression, description = block.split(":", 1)
            if expression.strip() and description.strip():
                expressions.append({
                    "expression": expression.strip(),
                    "description": description.strip(),
                })

    return expressions


# ==========================================================
# Main Parser
# ==========================================================

def parse_claim_response(response):

    response = clean_text(response)

    result = {

        "literal_meaning": "",

        "key_entities": [],

        "key_actions": [],

        "non_literal_expressions": [],

        "underlying_message": "",

        "explicit_claims": [],

        "caption_proposition": "",

        "claim_subject": "",

        "claim_predicate": "",

        "claim_object": "",

        "claim_source": "",

        "claim_target": "",

        "asserted_property": "",

        "transferred_property": "",

        "incongruity": "",

        "caption_polarity": "",

        "alternative_interpretation": "",

        "relation_family": "",

        "expected_visual_state": "",

        "opposite_visual_state": "",

        "reasoning_requirement": "",

        "structural_reasoning_type": "",

        "figurative_mechanism_candidates": "",

        "literal_polarity": "",

        "intended_polarity": "",

        "comparison_direction": "",

        "evaluation_target": "",

        "time_or_panel_scope": "",

        "implicit_claims": [],

        "linguistic_notes": [],

        "figurative_type": "",

        "linguistic_cue": "",

        "polarity_reversal": "",

        "background_knowledge": "",

        "confidence": None,

        "raw_output": response,
    }

    matches = []

    # ------------------------------------------------------
    # Locate headings
    # ------------------------------------------------------

    for field, aliases in HEADINGS.items():

        for alias in aliases:

            m = re.search(
                rf"^{re.escape(alias)}(?:\s*\([^:\n]*\))?\s*:",
                response,
                flags=re.MULTILINE | re.IGNORECASE,
            )

            if m:

                matches.append(
                    (
                        m.start(),
                        m.end(),
                        field,
                    )
                )

                break

    matches.sort()

    # ------------------------------------------------------
    # Slice sections
    # ------------------------------------------------------

    for i in range(len(matches)):

        _, start, field = matches[i]

        if i + 1 < len(matches):
            end = matches[i + 1][0]
        else:
            end = len(response)

        value = clean_text(response[start:end])

        if field == "confidence":

            result[field] = parse_confidence(value)

        elif field == "key_entities":

            result[field] = parse_list(value)

        elif field == "key_actions":

            result[field] = parse_list(value)

        elif field == "explicit_claims":

            result[field] = parse_list(value)

        elif field == "implicit_claims":

            result[field] = parse_list(value)

        elif field == "linguistic_notes":

            result[field] = parse_list(value)

        elif field == "non_literal_expressions":

            result[field] = parse_non_literal_expressions(value)

        else:

            result[field] = value

    return result
