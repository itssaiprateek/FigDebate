"""Conservative capability routing; semantic questions never go to pixel witnesses."""
import re


def observable_followup(question):
    """Separate an explicit inspection request from its proposed semantic use.

    Never return the answer, caption meaning, or implied relation. Preserve the
    original request in the caller's audit. Other questions keep their normal route.
    """
    text = str(question or '')
    # A generic compiler must not erase a region, quoted target, or comparison.
    # Such scoped requests retain their original route until explicitly split.
    if re.search(r'\b(left|right|above|below|top|bottom|beside|between|before|after)\b|["“”]', text, re.I):
        return ''
    if re.search(r'\b(visible text|readable text|signage)\b', text, re.I):
        if re.search(r'\b(support\w*|refut\w*|confirm\w*|claim\w*)\b', text, re.I):
            question = 'Transcribe readable text with its panel, speaker or attached object.'
            if re.search(r'\b(facial|emotional|expression)\b', text, re.I):
                question += ' Describe visible facial expression and posture.'
            return question
    return ''


def new_semantic_questions(questions, previous):
    """A repeated question is not new information or a new resolving check."""
    normalize = lambda value: ' '.join(str(value).casefold().split()).rstrip('?.! ')
    seen = {normalize(q) for q in previous}
    new = []
    for question in questions:
        key = normalize(question)
        if key and key not in seen:
            new.append(question)
            seen.add(key)
    return new[:1]


def route_question(question, requested=""):
    from agents.visual_adapter import AtomicVisualQuestionController
    text = str(question or "").strip()
    if not text or len(text) > 320 or re.search(
            r'\b(ground[ -]?truth|verdict|prediction|dataset label|final (?:label|decision))\b', text, re.I):
        return "blocked", "invalid_or_decision_exposing_question"
    if (re.search(r'\b(metadata|EXIF|previous designs|historical records)\b',text,re.I)
            and not re.search(r'\b(transcribe|visible text|caption|phrase|wording)\b',text,re.I)):
        return 'blocked','requested_information_not_available_to_witness'
    # Witnesses cannot decide entailment or contradiction. The tribunal can
    # examine that relation; the witness prohibition must not suppress its job.
    if re.search(r'\b(entail\w*|contradict\w*)\b', text, re.I):
        return 'tribunal', 'semantic_relation_requires_tribunal'
    visual = bool(re.search(r"\b(image|pictured|visible|photo|picture|panel|shirt|scene|ocr)\b", text, re.I))
    # A facial expression is an observable configuration, not a language idiom.
    routing_text = re.sub(r'\bfacial expressions?\b', 'facial cues', text, flags=re.I)
    caption = bool(re.search(r"\b(caption|idiom|phrase|wording|expression)\b", routing_text, re.I))
    semantic = bool(re.search(r"\b(imply|implies|mean|means|meaning|interpret\w*|intend\w*|intent|metaphor\w*|sarcas\w*|"
                              r"ironic|joke|symboli[sz]\w*|represent\w*|express\w*|suggest\w*|infer\w*|"
                              r"convey\w*|prove\w*|support\w*|motives?|useful|useless)\b", routing_text, re.I))
    valid, reason = AtomicVisualQuestionController.validate_question(
        text, AtomicVisualQuestionController.infer_question_type(text))
    if requested == "CAPTION_PREMISE" and not visual:
        return "language", "caption_only"
    if caption and semantic and not visual:
        return "language", "caption_interpretation"
    if semantic or not valid or (caption and visual):
        return "tribunal", reason or "cross_modal_interpretation"
    if requested == "COUNTER_INTERPRETATION" and not re.search(
            r"\b(transcribe|visible|readable|where|is there|are there|exact text)\b", text, re.I):
        return "tribunal", "counter_interpretation_requires_tribunal"
    if valid:
        return "visual", "observable_question"
    return "blocked", reason


def route_followup(review):
    routed = {"visual": [], "language": [], "tribunal": [], "blocked": []}
    questions = []
    if review.get("targeted_question"):
        questions.append((review["targeted_question"], review.get("requested_follow_up", "")))
    else:
        for field, target in (("agent1_questions", "VISUAL_PREMISE"), ("agent2_questions", "CAPTION_PREMISE"),
                              ("verification_requests", "COUNTER_INTERPRETATION")):
            questions.extend((q, target) for q in review.get(field, []) or [])
    audit = []
    for question, target in questions:
        role, reason = route_question(question, target)
        if question not in routed[role]:
            routed[role].append(question)
        audit.append({"question": question, "requested_target": target, "routed_to": role, "reason": reason})
    return routed, audit
