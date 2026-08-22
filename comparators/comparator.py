import re
from typing import Dict, List, Set

# ==========================================================
# Synonym Mapping
# ==========================================================

SYNONYMS = {

    # people
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "people": "person",
    "human": "person",

    # vehicles
    "automobile": "car",
    "vehicle": "car",

    # bicycles
    "bike": "bicycle",
    "cycle": "bicycle",

    # actions
    "eating": "eat",
    "eats": "eat",
    "ate": "eat",

    "running": "run",
    "runs": "run",
    "ran": "run",

    "walking": "walk",
    "walks": "walk",
    "walked": "walk",

    "talking": "talk",
    "talks": "talk",
    "talked": "talk",

    "sitting": "sit",
    "sat": "sit",

    "standing": "stand",
    "stood": "stand",
}


# ==========================================================
# Normalize Text
# ==========================================================

def normalize(text: str) -> str:

    text = text.lower()

    # remove bracketed text
    text = re.sub(r"\([^)]*\)", "", text)

    # remove punctuation
    text = re.sub(r"[^a-z0-9 ]", "", text)

    text = text.strip()

    if text in SYNONYMS:
        text = SYNONYMS[text]

    return text


# ==========================================================
# Convert list -> normalized set
# ==========================================================

def normalize_set(items: List[str]) -> Set[str]:

    result = set()

    for item in items:

        value = normalize(item)

        if value:
            result.add(value)

    return result


# ==========================================================
# Compare Sets
# ==========================================================

def compare_sets(left: Set[str], right: Set[str]):

    shared = sorted(left & right)

    missing = sorted(right - left)

    extra = sorted(left - right)

    return shared, missing, extra


# ==========================================================
# Alignment Score
#
# IMPORTANT: this measures LITERAL keyword overlap between the visual
# nouns/verbs and the caption's nouns/verbs. For a figurative-language
# task, near-zero literal overlap is the EXPECTED, NORMAL case for both
# ENTAILS and CONTRADICTS samples -- that is the whole premise of
# figurative language. This score does NOT measure figurative alignment
# and must not be read as evidence of contradiction on its own.
# ==========================================================

def compute_alignment_score(
    shared_entities,
    visual_entities,
    language_entities,
    shared_actions,
    visual_actions,
    language_actions,
):

    entity_total = max(
        len(visual_entities),
        len(language_entities),
        1,
    )

    action_total = max(
        len(visual_actions),
        len(language_actions),
        1,
    )

    entity_score = len(shared_entities) / entity_total

    action_score = len(shared_actions) / action_total

    score = (
        0.6 * entity_score +
        0.4 * action_score
    )

    return round(score, 2)


# ==========================================================
# Main Comparator
# ==========================================================

def compare(agent1_output: Dict, agent2_output: Dict):

    # ------------------------------------------------------
    # Visual entities
    # ------------------------------------------------------

    visual_entities = []

    visual_entities.extend(
        agent1_output.get("people", [])
    )

    visual_entities.extend(
        agent1_output.get("objects", [])
    )

    # ------------------------------------------------------
    # Language entities
    # ------------------------------------------------------

    language_entities = agent2_output.get(
        "key_entities",
        [],
    )

    # ------------------------------------------------------
    # Normalize
    # ------------------------------------------------------

    visual_entity_set = normalize_set(
        visual_entities
    )

    language_entity_set = normalize_set(
        language_entities
    )

    shared_entities, missing_language_entities, extra_visual_entities = compare_sets(
        visual_entity_set,
        language_entity_set,
    )

    # ------------------------------------------------------
    # Actions
    # ------------------------------------------------------

    visual_action_set = normalize_set(
        agent1_output.get(
            "actions",
            [],
        )
    )

    language_action_set = normalize_set(
        agent2_output.get(
            "key_actions",
            [],
        )
    )

    shared_actions, missing_language_actions, extra_visual_actions = compare_sets(
        visual_action_set,
        language_action_set,
    )

    # ------------------------------------------------------
    # Context Notes vs Genuine Conflicts
    #
    # These used to be lumped together under "potential_conflicts",
    # which meant near-every sample got 2-3 boilerplate "conflict"
    # strings regardless of the correct label -- because literal
    # non-overlap is the DEFAULT state for figurative language, not
    # a sign of contradiction. Splitting these into two buckets so
    # the Arbiter isn't fed the expected/normal case as if it were
    # evidence.
    # ------------------------------------------------------

    context_notes = []
    genuine_conflicts = []

    if len(shared_entities) == 0 and len(language_entity_set) > 0:
        context_notes.append(
            "No literal keyword overlap between visual objects and caption "
            "entities. This is EXPECTED for figurative captions and is not "
            "itself evidence of contradiction."
        )

    if len(shared_actions) == 0 and len(language_action_set) > 0:
        context_notes.append(
            "No literal keyword overlap between visual actions and caption "
            "actions. This is EXPECTED for figurative captions and is not "
            "itself evidence of contradiction."
        )

    if len(
        agent2_output.get(
            "non_literal_expressions",
            [],
        )
    ) > 0:
        context_notes.append(
            "Caption contains non-literal language, as expected for this task."
        )

    # This one IS a potentially genuine signal -- a large number of visual
    # entities with zero connection to the caption at all can indicate the
    # image is generic/unrelated to what the caption is about.
    if len(extra_visual_entities) > 5:
        genuine_conflicts.append(
            "Image contains many entities not referenced in the caption at "
            "all, which may indicate the image is generic or unrelated to "
            "the caption's subject matter."
        )

    # ------------------------------------------------------
    # Alignment Score
    # ------------------------------------------------------

    score = compute_alignment_score(
        shared_entities,
        visual_entity_set,
        language_entity_set,
        shared_actions,
        visual_action_set,
        language_action_set,
    )

    # ------------------------------------------------------
    # Final Comparison Object
    # ------------------------------------------------------

    comparison = {

        "shared_entities":
            shared_entities,

        "missing_language_entities":
            missing_language_entities,

        "extra_visual_entities":
            extra_visual_entities,

        "shared_actions":
            shared_actions,

        "missing_language_actions":
            missing_language_actions,

        "extra_visual_actions":
            extra_visual_actions,

        "non_literal_expressions":
            agent2_output.get(
                "non_literal_expressions",
                [],
            ),

        # Renamed from "potential_conflicts". Only contains signals that
        # are genuinely informative about a possible mismatch -- NOT the
        # expected baseline non-overlap of figurative language.
        "genuine_conflicts":
            genuine_conflicts,

        # Expected/normal observations about the sample that should NOT
        # be read as evidence toward CONTRADICTS.
        "context_notes":
            context_notes,

        "literal_keyword_overlap_score":
            score,

        "literal_keyword_overlap_score_note": (
            "This score reflects literal keyword overlap only. Near-zero "
            "values are expected and normal for figurative captions and do "
            "NOT indicate contradiction -- do not use this score alone to "
            "decide the label."
        ),
    }

    return comparison