import re
from typing import Dict, List

# ==========================================================
# Theme Dictionary
# ==========================================================

THEMES = {
    "finance": [
        "money",
        "cash",
        "wealth",
        "economy",
        "economic",
        "market",
        "stock",
        "bank",
        "profit",
        "business",
        "trade",
        "investment",
        "financial",
        "income",
        "salary",
    ],

    "growth": [
        "grow",
        "growth",
        "increase",
        "rise",
        "up",
        "upward",
        "recover",
        "recovery",
        "improve",
        "progress",
        "expand",
        "gain",
        "success",
    ],

    "decline": [
        "decline",
        "fall",
        "drop",
        "collapse",
        "loss",
        "crash",
        "failure",
        "down",
        "decrease",
        "recession",
    ],

    "emotion": [
        "happy",
        "smile",
        "sad",
        "cry",
        "anger",
        "fear",
        "joy",
        "excited",
        "worried",
        "frustrated",
    ],

    "conflict": [
        "fight",
        "war",
        "battle",
        "attack",
        "violence",
        "weapon",
        "enemy",
    ],

    "nature": [
        "tree",
        "forest",
        "river",
        "ocean",
        "mountain",
        "flower",
        "sun",
        "rain",
        "cloud",
        "animal",
    ],

    "health": [
        "doctor",
        "hospital",
        "medicine",
        "patient",
        "disease",
        "virus",
        "health",
        "medical",
    ],

    "technology": [
        "computer",
        "robot",
        "ai",
        "artificial intelligence",
        "technology",
        "machine",
        "internet",
        "software",
        "digital",
    ],
}

# ==========================================================
# Utilities
# ==========================================================

def normalize(text: str) -> str:
    if text is None:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = " ".join(text.split())

    return text


def detect_themes(text: str) -> List[str]:
    text = normalize(text)

    found = []

    for theme, words in THEMES.items():

        for word in words:

            if re.search(rf"\b{re.escape(word)}\b", text):
                found.append(theme)
                break

    return found


# ==========================================================
# Comparator V2
# ==========================================================

def compare(
    visual_output: Dict,
    language_output: Dict,
) -> Dict:

    visual_summary = visual_output.get(
        "visual_description",
        "",
    )

    objects = visual_output.get(
        "objects",
        [],
    )

    scene_type = visual_output.get(
        "scene_type",
        "",
    )

    symbolic_tone = visual_output.get(
        "symbolic_tone",
        "",
    )

    language_summary = language_output.get(
        "intended_meaning",
        "",
    )

    figurative_type = language_output.get(
        "figurative_type",
        "",
    )

    background = language_output.get(
        "background_knowledge",
        "",
    )

    # ------------------------------------------------------
    # Theme Detection
    # ------------------------------------------------------

    visual_text = " ".join([
        visual_summary,
        " ".join(objects),
        scene_type,
        symbolic_tone,
    ])

    visual_evidence_text = " ".join([
        visual_summary,
        " ".join(objects),
        scene_type,
        symbolic_tone,
    ])

    language_text = " ".join([
        language_summary,
        figurative_type,
        background,
    ])

    visual_themes = detect_themes(visual_text)

    language_themes = detect_themes(language_text)

    shared = sorted(
        list(
            set(visual_themes) &
            set(language_themes)
        )
    )

    total = len(
        set(visual_themes) |
        set(language_themes)
    )

    if total == 0:
        alignment_score = 0.0
    else:
        alignment_score = len(shared) / total

    # ------------------------------------------------------
    # Supporting Evidence
    # ------------------------------------------------------

    supporting = []
    conflicting = []

    if shared:
        supporting.append(
            f"Shared semantic themes: {', '.join(shared)}."
        )

    if (
        "financial" in scene_type.lower()
        and "finance" in language_themes
    ):
        supporting.append(
            "Visual scene represents a financial setting that aligns with the caption."
        )

    if (
        "upward" in visual_evidence_text.lower()
        or "rise" in visual_evidence_text.lower()
        or "arrow" in visual_evidence_text.lower()
    ):
        if "growth" in language_themes:
            supporting.append(
                "Upward visual elements support the idea of growth or recovery."
            )

    if (
        "smile" in visual_evidence_text.lower()
        and "growth" in language_themes
    ):
        supporting.append(
            "Positive facial expression is consistent with a successful outcome."
        )

    symbolic = symbolic_tone.lower().strip()

    if symbolic and symbolic != "none":

        matched = False

        for theme in language_themes:

            if theme in symbolic:

                matched = True
                break

        if matched:

            supporting.append(
                f"Relevant symbolism: {symbolic_tone}"
        )

        else:

            conflicting.append(
                "Visual symbolism does not clearly support the figurative meaning."
        )
    # ------------------------------------------------------
# Missing Visual Concepts
# ------------------------------------------------------

    missing = []

    for theme in language_themes:

        if theme not in visual_themes:

            missing.append(
                f"Visual evidence for '{theme}' is missing."
        )
    # ------------------------------------------------------
    # Conflicting Evidence
    # ------------------------------------------------------

    

    if not shared:
        conflicting.append(
            "Image and caption do not appear to discuss the same semantic themes."
        )

    if (
        "finance" in visual_themes
        and "health" in language_themes
    ):
        conflicting.append(
            "Financial image conflicts with health-related caption."
        )

    if (
        "nature" in visual_themes
        and "technology" in language_themes
    ):
        conflicting.append(
            "Natural scene conflicts with technology-focused caption."
        )

    if (
        "conflict" in visual_themes
        and "growth" in language_themes
    ):
        conflicting.append(
            "Violent imagery conflicts with positive growth message."
        )

    # ------------------------------------------------------
    # Possible Alignment
    # ------------------------------------------------------

    if alignment_score >= 0.75:
        alignment = (
            "Strong semantic alignment between the image and caption."
        )

    elif alignment_score >= 0.40:
        alignment = (
            "Partial semantic alignment; some concepts overlap."
        )

    elif alignment_score > 0:
        alignment = (
            "Weak semantic alignment."
        )

    else:
        alignment = (
            "No meaningful semantic alignment detected."
        )
# ------------------------------------------------------
# Recommendation
# ------------------------------------------------------

    if alignment_score >= 0.75:

        recommendation = "LEAN_ENTAILS"

    elif alignment_score >= 0.40:

        recommendation = "UNCERTAIN"

    elif len(conflicting) >= 2:

        recommendation = "LEAN_CONTRADICTS"

    else:

        recommendation = "UNCERTAIN"
    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    return {

        "visual_summary": visual_summary,

        "language_summary": language_summary,

        "scene_type": scene_type,

        "figurative_type": figurative_type,

        "visual_themes": visual_themes,

        "language_themes": language_themes,

        "shared_themes": shared,

        "possible_alignment": alignment,

        "supporting_points": supporting,

        "conflicting_points": conflicting,

        "missing_visual_concepts": missing,

        "recommendation": recommendation,

        "alignment_score": round(alignment_score, 2),
}