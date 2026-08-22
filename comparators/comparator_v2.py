"""Evidence-aware incongruity analysis for FigDebate.

The comparator characterizes the image-caption gap. It does not make the
final NLI decision, and weak word or theme overlap is never treated as proof.
"""

import re
from typing import Dict, List

from engine.relation_schema import build_claim_relation, nominate_visual_relations


THEMES = {
    "finance": ("money", "cash", "wealth", "economy", "economic", "market", "stock", "bank", "profit", "financial"),
    "growth": ("grow", "growth", "increase", "rise", "upward", "recover", "recovered", "recovering", "recovery", "improve", "progress", "success"),
    "decline": ("decline", "drop", "collapse", "loss", "crash", "decrease", "recession", "downward"),
    "emotion": ("happy", "smile", "sad", "cry", "anger", "fear", "joy", "excited", "worried", "frustrated"),
    "conflict": ("fight", "war", "battle", "attack", "violence", "weapon", "enemy"),
    "nature": ("tree", "forest", "river", "ocean", "mountain", "flower", "sun", "rain", "cloud", "animal"),
    "health": ("doctor", "hospital", "medicine", "patient", "disease", "virus", "health", "medical"),
    "technology": ("computer", "robot", "artificial intelligence", "technology", "machine", "internet", "software", "digital"),
}

DOWNWARD_TREND_CUES = (
    "downward trend", "declining chart", "decreasing chart", "falling graph",
    "negative trend", "arrow points downward", "arrow pointing downward",
    "line falls", "line is falling", "red line falling",
)
UPWARD_TREND_CUES = (
    "upward trend", "increasing chart", "rising graph", "positive trend",
    "arrow points upward", "arrow pointing upward", "line rises", "line is rising",
    "green line rising", "upward arrow",
)
GROWTH_CLAIM_CUES = (
    "recover", "recovered", "recovering", "recovery", "rise", "rises",
    "rising", "increase", "increases", "increasing", "growth", "improve",
    "improved", "improving", "bounced back",
)
DECLINE_CLAIM_CUES = (
    "decline", "declined", "declining", "drop", "dropped", "dropping",
    "fall", "fell", "falling", "decrease", "decreased", "decreasing",
    "collapse", "collapsed", "crash", "crashed", "loss", "recession",
)
RELATIONAL_CAPTION_CUES = (" while ", " whereas ", " but ", " versus ", " vs ")
TEXT_SURFACE_CUES = (
    "sign", "poster", "label", "screen", "chart", "meme", "advertisement",
    "packaging", "bottle", "board", "banner", "caption", "speech bubble",
)
SYMBOL_OBJECT_CUES = (
    "heart", "wing", "wings", "chain", "crown", "halo", "mask", "scales",
    "fire", "flame", "shadow", "cage", "key", "lock", "bridge", "wall",
)
STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "being",
    "both", "caption", "could", "does", "doesnt", "during", "each", "from", "have",
    "for", "image", "into", "just", "like", "more", "most", "one", "only", "over", "people",
    "shows", "speaker", "that", "the", "their", "them", "there", "these", "this", "those",
    "through", "under", "with", "while", "would",
}
MIN_DIRECT_TERM_MATCHES = 2


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower()).split())


def content_terms(text: str) -> List[str]:
    return sorted({
        token for token in normalize(text).split()
        if len(token) >= 3 and token not in STOPWORDS
    })


def numeric_terms(text: str) -> List[str]:
    """Numbers are strong anchors in image-caption examples such as prices."""
    return sorted(set(re.findall(r"\b\d+(?:\.\d+)?\b", str(text or ""))))


def shared_numeric_terms(visual_text: str, claim_text: str) -> List[str]:
    """Match displayed and caption numbers with small formatting tolerance."""
    visual_numbers = numeric_terms(visual_text)
    claim_numbers = numeric_terms(claim_text)
    shared = set()
    for visual_number in visual_numbers:
        for claim_number in claim_numbers:
            visual_value = float(visual_number)
            claim_value = float(claim_number)
            tolerance = max(1e-4, abs(claim_value) * 1e-5)
            if abs(visual_value - claim_value) <= tolerance:
                shared.add(claim_number)
    return sorted(shared, key=float)


def detect_themes(text: str) -> List[str]:
    normalized = normalize(text)
    return [
        theme for theme, words in THEMES.items()
        if any(re.search(rf"\b{re.escape(word)}\b", normalized) for word in words)
    ]


def first_fact_with(facts, cues):
    for fact in facts:
        fact_text = str(fact)
        normalized = normalize(fact_text)
        for cue in cues:
            match = re.search(rf"\b{re.escape(normalize(cue))}\b", normalized)
            if not match:
                continue
            prefix = normalized[:match.start()].split()
            if re.search(
                r"\b(?:no|not|never|without|isn t|aren t|doesn t|didn t)\b",
                " ".join(prefix[-3:]),
            ):
                continue
            return fact_text
    return ""


def directional_intent(*texts):
    """Resolve the asserted direction, prioritizing the original caption."""
    for text in texts:
        normalized = normalize(text)
        if not normalized:
            continue
        has_growth = any(cue in normalized for cue in GROWTH_CLAIM_CUES)
        has_decline = any(cue in normalized for cue in DECLINE_CLAIM_CUES)
        negated_growth = bool(
            re.search(
                r"\b(?:not|never|failed to|did not)\s+"
                r"(?:recover|rise|increase|grow|improve)\b",
                normalized,
            )
        )
        if negated_growth:
            has_growth = False
            has_decline = True
        if has_growth != has_decline:
            return "growth" if has_growth else "decline"
    return None


def compare(visual_output: Dict, language_output: Dict, caption: str = "") -> Dict:
    """Describe evidence available for cross-modal semantic review.

    This component deliberately makes a deterministic recommendation only for
    relations it can verify, such as an explicit chart direction. Figurative
    pairs with little lexical overlap are routed to semantic review rather than
    being described repeatedly as missing evidence.
    """
    visual_summary = str(visual_output.get("visual_description", ""))
    objects = [str(item) for item in (visual_output.get("objects", []) or [])]
    scene_type = str(visual_output.get("scene_type", ""))
    visual_facts = [str(item) for item in (visual_output.get("visual_facts", []) or [])]
    visual_relations = [
        str(item) for item in (visual_output.get("visual_relations", []) or [])
    ]
    visible_text = [str(item) for item in (visual_output.get("visible_text", []) or [])]
    symbolic_elements = [
        str(item)
        for item in (visual_output.get("possible_visual_metaphors", []) or [])
        if str(item).strip().lower() not in {"", "none", "unclear"}
    ]
    symbolic_tone = str(visual_output.get("symbolic_tone", "")).strip()
    explicit_symbolic_evidence = bool(
        symbolic_elements
        or symbolic_tone.lower() not in {"", "none", "unclear", "unspecified"}
    )
    surface_meaning = str(language_output.get("surface_meaning", ""))
    intended_meaning = str(language_output.get("intended_meaning", ""))
    figurative_type = str(language_output.get("figurative_type", ""))
    explicit_claims = [str(item) for item in (language_output.get("explicit_claims", []) or [])]
    caption_proposition = str(language_output.get("caption_proposition", ""))
    linguistic_cue = str(language_output.get("linguistic_cue", ""))
    claim_relation = language_output.get("claim_relation") or build_claim_relation(
        caption, language_output
    )
    claim_contract = claim_relation.get("claim_contract", {}) or {}
    relation_candidates = nominate_visual_relations(visual_output, claim_relation)
    grounded_object_text = normalize(" ".join(
        objects + visual_facts + visual_relations
    ))
    symbolic_object_candidate = bool(
        figurative_type == "metaphor"
        and any(
            re.search(rf"\b{re.escape(cue)}\b", grounded_object_text)
            for cue in SYMBOL_OBJECT_CUES
        )
    )
    has_symbolic_evidence = bool(
        explicit_symbolic_evidence or symbolic_object_candidate
    )

    direct_visual_text = " ".join(
        visible_text + visual_relations + visual_facts + objects + [scene_type, visual_summary]
    )
    claim_text = " ".join(
        [caption, caption_proposition, surface_meaning, intended_meaning, linguistic_cue]
        + explicit_claims
    ).strip()
    direct_terms = sorted(
        term
        for term in set(content_terms(direct_visual_text)) & set(content_terms(claim_text))
        if not term.isdigit()
    )
    shared_numbers = shared_numeric_terms(direct_visual_text, claim_text)

    visual_context_text = direct_visual_text
    language_text = " ".join([surface_meaning, intended_meaning, figurative_type, claim_text])
    visual_themes = detect_themes(visual_context_text)
    language_themes = detect_themes(language_text)
    shared_themes = sorted(set(visual_themes) & set(language_themes))
    claim_direction = (
        directional_intent(
            claim_relation.get("intended_proposition"),
            caption_proposition,
            caption,
        )
        if claim_relation.get("resolved", False)
        else None
    )

    relation_text = " ".join(visual_relations + visual_facts)
    visible_text_terms = set(content_terms(" ".join(visible_text)))
    relation_terms = set(content_terms(relation_text))
    normalized_relation_text = f" {normalize(relation_text)} "
    region_pairs = (
        ("left", "right"),
        ("top", "bottom"),
        ("first", "second"),
        ("white", "red"),
    )
    has_explicit_region_pair = any(
        all(
            re.search(rf"\b{re.escape(region)}\b", normalized_relation_text)
            for region in pair
        )
        for pair in region_pairs
    )
    normalized_caption = f" {normalize(caption)} "
    visual_layout_text = normalize(" ".join(
        visual_relations + visible_text + objects + [scene_type, visual_summary]
    ))
    has_text_surface = any(
        re.search(rf"\b{re.escape(cue)}\b", visual_layout_text)
        for cue in TEXT_SURFACE_CUES
    )
    has_layout_cue = bool(re.search(
        r"\b(left|right|top|bottom|panel|side|labels?|attached|above|below)\b",
        visual_layout_text,
    ))
    text_surface_without_ocr = bool(has_text_surface and not visible_text)
    relation_binding_required = bool(
        (
            (
                any(cue in normalized_caption for cue in RELATIONAL_CAPTION_CUES)
                and (bool(visible_text) or has_text_surface)
            )
            or len(visible_text) >= 2
            or (bool(visible_text) and has_layout_cue)
            or text_surface_without_ocr
        )
    )
    relation_binding_observed = bool(
        relation_binding_required
        and has_explicit_region_pair
        and len(visible_text_terms.intersection(relation_terms)) >= 2
    )
    # The four-crop verifier assumes a very specific graphic: two compared
    # items across the top with their outcomes below.  Text binding can be
    # unresolved in many other memes, but those must be reviewed full-image.
    region_pair_verifier_eligible = bool(
        claim_relation.get("relation_family") == "pace"
        and relation_binding_required
        and has_layout_cue
        and (len(visible_text) >= 2 or has_text_surface)
    )

    direct_support = []
    direct_conflict = []
    grounded_anchors = []
    missing_evidence = []
    neutral_notes = []
    review_questions = []

    for candidate in relation_candidates:
        if (
            len(set(candidate.get("matched_cues", []))) < 2
            or not candidate.get("matched_entities")
            or not claim_contract.get("safe_for_directional_reasoning", False)
        ):
            continue
        evidence = (
            f"[VISUAL] {candidate.get('text')} [CAPTION] Entity-bound "
            f"{candidate.get('relation_family')} cues "
            f"({', '.join(candidate.get('matched_cues', []))}) "
            "match the structured caption relation."
        )
        if candidate.get("proposed_relation") == "SUPPORT":
            direct_support.append(evidence)
        elif candidate.get("proposed_relation") == "CONFLICT":
            direct_conflict.append(evidence)

    if len(direct_terms) >= MIN_DIRECT_TERM_MATCHES or shared_numbers:
        anchors = direct_terms[:8] + [f"number:{number}" for number in shared_numbers]
        grounded_anchors.append(
            "[VISUAL] Direct entity/action anchors require polarity review: "
            + ", ".join(anchors) + "."
        )
    if shared_themes:
        neutral_notes.append(
            "[HINT] Shared broad themes require Arbiter verification: "
            + ", ".join(shared_themes) + "."
        )

    directional_facts = visual_relations + visual_facts
    upward_fact = first_fact_with(directional_facts, UPWARD_TREND_CUES)
    downward_fact = first_fact_with(directional_facts, DOWNWARD_TREND_CUES)
    if upward_fact and claim_direction == "growth":
        direct_support.append(
            f"[VISUAL] {upward_fact} [CAPTION] The caption explicitly expresses growth or recovery."
        )
    if downward_fact and claim_direction == "decline":
        direct_support.append(
            f"[VISUAL] {downward_fact} [CAPTION] The caption explicitly expresses decline."
        )
    if downward_fact and claim_direction == "growth":
        direct_conflict.append(
            f"[VISUAL] {downward_fact} [CAPTION] The caption explicitly expresses growth or recovery."
        )
    if upward_fact and claim_direction == "decline":
        direct_conflict.append(
            f"[VISUAL] {upward_fact} [CAPTION] The caption explicitly expresses decline."
        )

    has_visual_evidence = bool(
        visible_text or visual_relations or visual_facts or objects or visual_summary
    )
    if not has_visual_evidence:
        missing_evidence.append(
            "[MISSING] Agent 1 produced no usable observed image evidence."
        )

    if visible_text:
        review_questions.append(
            "Check whether the exact visible wording supports, reverses, or satirizes the caption proposition."
        )
    if relation_binding_required and not relation_binding_observed:
        review_questions.append(
            "Reinspect which exact text belongs to each panel or object before deciding polarity."
        )
        neutral_notes.append(
            "[CAUTION] Text-bearing or comparison regions were detected, but "
            "their text-to-object bindings are not grounded."
        )
        if text_surface_without_ocr:
            missing_evidence.append(
                "[MISSING] A visible text-bearing surface was detected, but no usable OCR was grounded."
            )
    if visual_relations:
        review_questions.append(
            "Check whether spatial layout, direction, or contrast realizes the caption meaning."
        )
    if figurative_type in {"sarcasm", "metaphor", "humor"}:
        review_questions.append(
            "Compare the intended meaning, not only the caption's literal wording."
        )
    else:
        review_questions.append(
            "The caption may be literal while the figurative mechanism is visual; compare the full scene."
        )
    if not claim_contract.get("safe_for_directional_reasoning", False):
        review_questions.append(
            "Resolve the caption subject, target, numbers, and polarity before assigning directional evidence."
        )
        neutral_notes.append(
            "[CAUTION] Agent 2's claim frame did not pass the immutable-caption preservation audit."
        )

    if direct_support and direct_conflict:
        recommendation = "REVIEW_MIXED_EVIDENCE"
        evidence_status = "MIXED_VERIFIED_EVIDENCE"
        incongruity = (
            "Decision-grade visual support and conflict are both present."
        )
    elif direct_conflict:
        recommendation = "LEAN_CONTRADICTS"
        evidence_status = "CONFLICTING"
        incongruity = "Direct visual evidence conflicts with an explicit caption direction."
    elif direct_support:
        recommendation = "LEAN_ENTAILS"
        evidence_status = "SUPPORTED"
        incongruity = "Direct visual facts overlap with a caption claim; figurative fit still requires Arbiter review."
    elif grounded_anchors:
        recommendation = "REVIEW_GROUNDED_ANCHORS"
        evidence_status = "GROUNDED_REVIEW_REQUIRED"
        incongruity = "The image and caption share concrete anchors, but their polarity requires review."
    elif has_visual_evidence:
        recommendation = "SEMANTIC_REVIEW"
        evidence_status = "SEMANTIC_REVIEW_REQUIRED"
        incongruity = (
            "Both modalities contain evidence, but their figurative relationship cannot be "
            "decided from lexical overlap alone."
        )
    else:
        recommendation = "UNCERTAIN"
        evidence_status = "INSUFFICIENT_VISUAL_EVIDENCE"
        incongruity = "Agent 1 did not provide enough observed evidence for comparison."

    evidence_quality = {
        "SUPPORTED": 1.0,
        "CONFLICTING": 1.0,
        "MIXED_VERIFIED_EVIDENCE": 0.7,
        "GROUNDED_REVIEW_REQUIRED": 0.8,
        "SEMANTIC_REVIEW_REQUIRED": 0.6,
        "INSUFFICIENT_VISUAL_EVIDENCE": 0.25,
    }[evidence_status]
    if relation_binding_required and not relation_binding_observed:
        evidence_quality = min(evidence_quality, 0.55)
    if not claim_contract.get("safe_for_directional_reasoning", False):
        evidence_quality = min(evidence_quality, 0.45)

    return {
        "visual_summary": visual_summary,
        "language_summary": intended_meaning,
        "scene_type": scene_type,
        "figurative_type": figurative_type,
        "caption": caption,
        "claim_proposition": caption_proposition or intended_meaning or surface_meaning,
        "claim_relation": claim_relation,
        "claim_contract": claim_contract,
        "claim_contract_valid": bool(
            claim_contract.get("safe_for_directional_reasoning", False)
        ),
        "claim_contract_warnings": claim_contract.get("warnings", []),
        "structured_relation_candidates": relation_candidates,
        "visual_themes": visual_themes,
        "language_themes": language_themes,
        "claim_direction": claim_direction,
        "shared_themes": shared_themes,
        "shared_terms": direct_terms,
        "direct_evidence_terms": direct_terms + [f"number:{number}" for number in shared_numbers],
        "incongruity": incongruity,
        "possible_alignment": (
            "Direct grounded relationship." if direct_support or direct_conflict
            else "Grounded anchors require polarity review." if grounded_anchors
            else "Semantic cross-modal review required." if has_visual_evidence
            else "Insufficient observed visual evidence."
        ),
        "supporting_evidence": direct_support,
        "contradicting_evidence": direct_conflict,
        "grounded_anchor_evidence": grounded_anchors,
        "missing_evidence": missing_evidence,
        "unsupported_inferences": [],
        "neutral_notes": neutral_notes,
        "review_questions": review_questions,
        "supporting_points": direct_support,
        "conflicting_points": direct_conflict,
        "missing_visual_concepts": missing_evidence,
        "required_evidence_status": evidence_status,
        "recommendation": recommendation,
        "evidence_quality": evidence_quality,
        "visual_schema_complete": bool(visual_output.get("schema_complete", False)),
        "has_visual_relations": bool(visual_relations),
        "has_visible_text": bool(visible_text),
        "has_symbolic_evidence": has_symbolic_evidence,
        "has_explicit_symbolic_evidence": explicit_symbolic_evidence,
        "has_symbolic_object_candidate": symbolic_object_candidate,
        "has_text_surface": has_text_surface,
        "text_surface_without_ocr": text_surface_without_ocr,
        "relation_binding_required": relation_binding_required,
        "relation_binding_observed": relation_binding_observed,
        "region_pair_verifier_eligible": region_pair_verifier_eligible,
        "alignment_score": round(
            len(direct_terms) / max(1, len(content_terms(claim_text))),
            2,
        ),
    }
