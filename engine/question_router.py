"""Universal, issue-specific tribunal question routing."""

from dataclasses import asdict, dataclass

from agents.visual_adapter import AtomicVisualQuestionController
from engine.reasoning_schema import (
    normalize_figurative_mechanism,
    normalize_structural_type,
)


ISSUE_TYPES = {
    "CLAIM_CONTRACT_INVALID",
    "OCR_BINDING",
    "COMPARISON_OR_OUTCOME",
    "DIRECTIONAL_RELATION",
    "SYMBOL_ATTACHMENT",
    "NORMATIVE_REASONING",
    "BACKGROUND_KNOWLEDGE",
    "CONFLICTING_VERIFIED_EVIDENCE",
    "INSUFFICIENT_VISUAL_OBSERVATION",
    "SEMANTIC_RELATION",
    "TEMPORAL_CAUSAL_SEQUENCE",
    "QUOTED_STATEMENT_AND_REACTION",
    "AFFECTIVE_SCENE",
    "METAPHOR_MAPPING",
    "SARCASM_POLARITY",
}


@dataclass(frozen=True)
class TribunalQuestionPlan:
    issue_type: str
    issue: str
    agent1_question: str
    agent2_question: str
    verification_request: str
    visually_resolvable: bool

    def to_dict(self):
        return asdict(self)


def _clean(value, fallback="the claim subject"):
    text = " ".join(str(value or "").split()).strip()
    return text or fallback


def classify_issue(comparison):
    comparison = comparison or {}
    contract = comparison.get("claim_contract", {}) or {}
    relation = comparison.get("claim_relation", {}) or {}
    status = comparison.get("required_evidence_status")
    if not contract.get(
        "safe_for_tribunal_reasoning",
        contract.get("safe_for_directional_reasoning", False),
    ):
        return "CLAIM_CONTRACT_INVALID"
    structural = normalize_structural_type(
        contract.get("structural_reasoning_type")
        or relation.get("structural_reasoning_type")
    )
    mechanism = normalize_figurative_mechanism(
        relation.get("figurative_mechanism")
    )
    if contract.get("requires_normative_reasoning", False):
        return "NORMATIVE_REASONING"
    if (
        contract.get("requires_background_knowledge", False)
        or structural == "BACKGROUND_REQUIRED"
    ):
        return "BACKGROUND_KNOWLEDGE"
    if structural == "TEMPORAL_CAUSAL_SEQUENCE":
        return "TEMPORAL_CAUSAL_SEQUENCE"
    if structural == "QUOTED_STATEMENT_AND_REACTION":
        return "QUOTED_STATEMENT_AND_REACTION"
    if mechanism == "METAPHOR_MAPPING":
        return "METAPHOR_MAPPING"
    if mechanism == "SARCASM_POLARITY":
        return "SARCASM_POLARITY"
    if structural == "AFFECTIVE_SCENE":
        return "AFFECTIVE_SCENE"
    if (
        comparison.get("relation_binding_required", False)
        and not comparison.get("relation_binding_observed", False)
    ) or structural == "OCR_REGION_BINDING":
        return "OCR_BINDING"
    if status == "MIXED_VERIFIED_EVIDENCE":
        return "CONFLICTING_VERIFIED_EVIDENCE"
    if (
        relation.get("relation_family") in {"pace", "outcome", "quantity"}
        or structural == "COMPARATIVE_LAYOUT"
    ):
        return "COMPARISON_OR_OUTCOME"
    if status in {
        "SUPPORT_CANDIDATE", "CONFLICT_CANDIDATE",
        "MIXED_RELATION_CANDIDATES",
    }:
        return "DIRECTIONAL_RELATION"
    if comparison.get("has_symbolic_evidence", False):
        return "SYMBOL_ATTACHMENT"
    if status == "INSUFFICIENT_VISUAL_EVIDENCE":
        return "INSUFFICIENT_VISUAL_OBSERVATION"
    return "SEMANTIC_RELATION"


def build_question_plan(comparison):
    comparison = comparison or {}
    relation = comparison.get("claim_relation", {}) or {}
    contract = comparison.get("claim_contract", {}) or {}
    issue_type = classify_issue(comparison)
    subject = _clean(relation.get("subject"))
    expected = _clean(relation.get("expected_visual_state"), "expected state")
    opposite = _clean(relation.get("opposite_visual_state"), "opposite state")

    if issue_type == "CLAIM_CONTRACT_INVALID":
        return TribunalQuestionPlan(
            issue_type,
            "The caption claim frame is incomplete or changed.",
            f"What directly visible state or action involves {subject}?",
            "Restate the exact caption proposition, preserving entities, polarity, comparisons, and outcomes.",
            "Validate the repaired claim contract before directional reasoning.",
            False,
        )
    if issue_type == "TEMPORAL_CAUSAL_SEQUENCE":
        return TribunalQuestionPlan(
            issue_type,
            "The claim depends on event order, actual cause, or attributed cause.",
            "For each relevant panel in order, what actor, action, object, immediate effect, and later outcome are directly visible?",
            "State the caption's claimed actor, cause, outcome, and any blamed target separately.",
            "Build and verify the ordered event chain before assigning a relation.",
            True,
        )
    if issue_type == "QUOTED_STATEMENT_AND_REACTION":
        return TribunalQuestionPlan(
            issue_type,
            "A displayed statement and a reaction to it have different semantic roles.",
            "What exact statement is shown, what exact reaction is shown, and what visible item is the reaction directed toward?",
            "Separate the quoted proposition from the speaker's evaluation or reaction to it.",
            "Verify statement-to-reaction target binding and reaction polarity.",
            True,
        )
    if issue_type == "AFFECTIVE_SCENE":
        return TribunalQuestionPlan(
            issue_type,
            "The claim depends on an emotional or evaluative scene state.",
            f"Describe the directly visible facial, posture, interaction, motion, and environmental cues involving {subject}.",
            f"State which observable cues would instantiate '{expected}' versus '{opposite}' without imagining the image.",
            "Aggregate only mutually consistent observable affective cues.",
            True,
        )
    if issue_type == "METAPHOR_MAPPING":
        return TribunalQuestionPlan(
            issue_type,
            "A literal source entity may represent a different caption target.",
            f"Describe the directly visible behavior and state of the source entity associated with {subject}, without interpreting what it represents.",
            "State the caption target and the exact property that a source image would need to share with it.",
            "Verify source property, target property, and their shared mapping separately.",
            True,
        )
    if issue_type == "SARCASM_POLARITY":
        return TribunalQuestionPlan(
            issue_type,
            "The caption's literal and intended polarity may differ.",
            f"What directly visible subject and state corresponding to {subject} are shown?",
            "State the literal polarity, intended polarity, and exact linguistic reversal cues separately.",
            "Verify visual referent grounding separately from linguistic polarity reversal.",
            True,
        )
    if issue_type == "OCR_BINDING":
        return TribunalQuestionPlan(
            issue_type,
            "Visible text is not reliably bound to its object or region.",
            "Which exact text belongs to each relevant visible object, person, panel, or region?",
            "Which caption entities and comparison roles must the visible labels instantiate?",
            "Verify each object-to-text binding from targeted image regions.",
            True,
        )
    if issue_type == "COMPARISON_OR_OUTCOME":
        return TribunalQuestionPlan(
            issue_type,
            "The disputed claim depends on a comparison, action, or displayed outcome.",
            f"What separate visible action and resulting outcome are shown for each instance of {subject}?",
            "Which actions, outcomes, and comparison direction does the caption actually assert?",
            "Verify participant-to-action and participant-to-outcome bindings separately.",
            True,
        )
    if issue_type == "DIRECTIONAL_RELATION":
        return TribunalQuestionPlan(
            issue_type,
            "A lexical relation candidate has not been visually verified.",
            f"What exact visible state is shown for {subject}?",
            f"Explain why '{expected}' and '{opposite}' are mutually opposing conditions for the caption.",
            "Independently verify entity identity and directional state.",
            True,
        )
    if issue_type == "SYMBOL_ATTACHMENT":
        return TribunalQuestionPlan(
            issue_type,
            "A possible symbol must be identified and bound to the correct subject.",
            f"What symbol is visibly attached to {subject}, and what is its observable condition?",
            "What property does the caption assert, without assuming a particular visible symbol?",
            "Verify symbol identity, attachment, condition, and claimed association independently.",
            True,
        )
    if issue_type == "NORMATIVE_REASONING":
        return TribunalQuestionPlan(
            issue_type,
            "The caption makes an evaluative or normative claim.",
            "What exact visible action, wording, target, and treatment are shown?",
            "State the evaluative claim and the behavior that would instantiate it, without imagining the image.",
            "Keep visual facts separate from the normative inference and verify both premises.",
            False,
        )
    if issue_type == "BACKGROUND_KNOWLEDGE":
        return TribunalQuestionPlan(
            issue_type,
            "The relation cannot be resolved from pixels alone.",
            f"What exact visible entity, action, text, or relationship involving {subject} is shown?",
            "State the minimum general background fact required to interpret the caption.",
            "Verify visual and background premises independently before combining them.",
            False,
        )
    if issue_type == "CONFLICTING_VERIFIED_EVIDENCE":
        return TribunalQuestionPlan(
            issue_type,
            "Verified evidence exists in opposing directions.",
            f"Which current visible state of {subject} resolves the conflicting records?",
            "Which exact caption condition distinguishes support from conflict?",
            "Reconfirm the disputed evidence using an independent method.",
            True,
        )
    if issue_type == "INSUFFICIENT_VISUAL_OBSERVATION":
        return TribunalQuestionPlan(
            issue_type,
            "The initial visual record is incomplete.",
            f"What directly visible entity and state correspond to {subject}?",
            "Identify the minimum observable condition needed to test the caption.",
            "Recover only the missing observation using a targeted inspection.",
            True,
        )
    return TribunalQuestionPlan(
        issue_type,
        "The cross-modal semantic relation is unresolved.",
        f"What exact visible state or relation is shown for {subject}?",
        "What exact proposition and polarity must the image support or conflict with?",
        "Verify the subject-bound relation without using missing evidence as conflict.",
        True,
    )


def compile_visual_question(comparison, mediation=None):
    """Return one validated visual-only question with a safe fallback.

    Mediator text is advisory.  It is never sent to Agent 1 until the same
    controller that validates direct visual questions has accepted it.
    """
    mediation = mediation or {}
    candidates = list(mediation.get("agent1_questions", []) or [])
    candidates += list(mediation.get("verification_requests", []) or [])
    for candidate in candidates:
        question = " ".join(str(candidate or "").split()).strip()
        question_type = AtomicVisualQuestionController.infer_question_type(
            question
        )
        valid, _ = AtomicVisualQuestionController.validate_question(
            question, question_type
        )
        if valid:
            return question

    fallback = build_question_plan(comparison or {}).agent1_question
    fallback_type = AtomicVisualQuestionController.infer_question_type(fallback)
    valid, _ = AtomicVisualQuestionController.validate_question(
        fallback, fallback_type
    )
    if valid:
        return fallback
    return "What directly visible state or action involves the main subject?"
