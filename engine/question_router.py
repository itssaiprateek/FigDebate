"""Universal, issue-specific tribunal question routing."""

from dataclasses import asdict, dataclass

from engine.reasoning_schema import (
    normalize_figurative_mechanism,
    normalize_structural_type,
)
from engine.pragmatic_schema import detect_pragmatic_schema


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
    "PRAGMATIC_SCOPE",
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
    if detect_pragmatic_schema(contract.get("source_caption")) != "NONE":
        return "PRAGMATIC_SCOPE"
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


# Source identity is not a question-selection policy. These templates contain
# no extracted subject/property guesses and cannot demand a literal idiom object.
_PLANS = {
    "CLAIM_CONTRACT_INVALID": (
        "The source representation must be checked before directional reasoning.",
        "", "Restate the complete original caption, preserving participants, negation and qualifiers.",
        "Validate source preservation before assessing the image relation.", False),
    "TEMPORAL_CAUSAL_SEQUENCE": (
        "The claim depends on order or an attributed cause.",
        "Describe the visible event sequence, keeping each actor, action and outcome attached to its panel.",
        "Which event order or causal relationship does the complete caption assert?",
        "Compare observed order with asserted order; do not infer causation from sequence alone.", True),
    "QUOTED_STATEMENT_AND_REACTION": (
        "A displayed statement and a reaction have different roles.",
        "Describe the visible response to the displayed message, identifying its speaker and target.",
        "Whose assertion or reaction does the caption express? Preserve any quotation scope.",
        "Check the message, reaction and target without treating a quoted belief as fact.", True),
    "AFFECTIVE_SCENE": (
        "The claim depends on a depicted expression or evaluation.",
        "Describe the visible facial expressions and posture of the pictured participants.",
        "What emotional or evaluative property does the caption assert, including its degree and scope?",
        "Check the contextual connection; appearance alone need not establish a hidden history.", True),
    "METAPHOR_MAPPING": (
        "A depicted situation may instantiate a figurative property.",
        "Describe the visible actions and physical states of the pictured entities.",
        "What property does the caption attribute under a conventional figurative reading? Preserve qualifiers.",
        "Check the licensed connection between depicted and asserted properties; topic overlap is insufficient.", True),
    "SARCASM_POLARITY": (
        "The expressed evaluation and implied attitude may differ.",
        "Describe the visible situation and the participants' actions.",
        "What does the caption assert, and what wording permits a different speaker attitude?",
        "Assess the expressed claim in context; sarcasm alone does not determine the relation.", True),
    "OCR_BINDING": (
        "Visible wording needs its speaker, object or panel attachment.",
        "Transcribe readable text with its speaker, object or panel attachment.",
        "Which caption roles need to be matched to the displayed wording?",
        "Verify text-to-role attachment before combining the claims.", True),
    "COMPARISON_OR_OUTCOME": (
        "The claim depends on a comparison or displayed outcome.",
        "Describe the visible action and displayed outcome for each pictured participant, retaining its panel and time wording.",
        "What comparison does the complete caption assert, and between which participants?",
        "Compare actual outcomes under the same scope; keep wishes and predictions separate.", True),
    "DIRECTIONAL_RELATION": (
        "A proposed relation needs a grounded comparison.",
        "Describe the directly visible actions and states of the pictured participants.",
        "What complete proposition does the caption assert? Preserve its polarity and qualifiers.",
        "Compare the assertion with observations; lack of evidence is not contrary evidence.", True),
    "SYMBOL_ATTACHMENT": (
        "A depicted symbol needs a physical attachment.",
        "Describe visible symbols with their location, attachment and physical condition.",
        "What property does the caption assert without assuming a symbol is present?",
        "Check the contextual association separately from the visible symbol.", True),
    "NORMATIVE_REASONING": (
        "The claim includes an evaluation.",
        "Describe the visible action and its recipient.",
        "What evaluation does the caption make and about which action?",
        "Separate observed facts from the evaluative inference.", False),
    "PRAGMATIC_SCOPE": (
        "A conventional construction affects the claim's scope.",
        "Transcribe readable wording with its speaker or object attachment.",
        "What proposition does the construction express, preserving its subject and scope?",
        "Check the contextual reading against the original assertion.", True),
    "BACKGROUND_KNOWLEDGE": (
        "A necessary premise may be unavailable in the image.",
        "", "Which necessary background premise cannot be established from the caption alone?",
        "Determine whether the unavailable premise is necessary; retain uncertainty if it is.", False),
    "CONFLICTING_VERIFIED_EVIDENCE": (
        "Opposing evidence needs a scope check.",
        "Describe the visible states with their participant, panel and time attachments.",
        "Which scope or qualifier distinguishes the competing interpretations of the caption?",
        "Check whether the opposing records describe the same subject and state.", True),
    "INSUFFICIENT_VISUAL_OBSERVATION": (
        "No deciding observation has been established.",
        "Describe the directly visible entities and their actions.",
        "Which necessary part of the complete caption needs observable evidence?",
        "Keep uncertainty unless an available observation resolves the necessary condition.", True),
    "SEMANTIC_RELATION": (
        "The image-to-caption relationship remains unresolved.",
        "Describe the directly visible relationships among the pictured entities.",
        "Explain the complete expressed caption, preserving roles, negation and qualifiers.",
        "Assess a context-licensed interpretation of the complete source; do not invent a literal referent.", True),
}


def build_question_plan(comparison):
    issue = classify_issue(comparison or {})
    if issue not in _PLANS:
        raise ValueError(f"Unknown question issue: {issue}")
    plan = TribunalQuestionPlan(issue, *_PLANS[issue])
    graph = ((comparison or {}).get("claim_contract") or {}).get("claim_graph") or {}
    if graph.get("representation") == "UNDECOMPOSED_SOURCE" and issue != "CLAIM_CONTRACT_INVALID":
        # Preserve coverage when the upstream issue tag is only a hypothesis.
        # Specific missing-fact requests from a hearing still take precedence.
        plan = TribunalQuestionPlan(issue, plan.issue,
            "Report visible text with its object or panel bindings, and visible actions, states and relationships. "
            "Distinguish observations from unclear details.",
            plan.agent2_question, plan.verification_request, plan.visually_resolvable)
    return plan


def compile_visual_question(comparison, mediation=None):
    """Select a visual request, never replace a rejected request with another task.

    Legacy text callers retain this adapter; every origin uses the same route.
    Empty means no valid visual work and must not cause a model call.
    """
    from engine.review_routing import route_question
    mediation = mediation or {}
    candidates = [(q, "VISUAL_PREMISE") for q in mediation.get("agent1_questions", []) or []]
    candidates += [(q, "COUNTER_INTERPRETATION") for q in mediation.get("verification_requests", []) or []]
    if not candidates:
        candidates = [(build_question_plan(comparison or {}).agent1_question, "VISUAL_PREMISE")]
    for candidate, target in candidates:
        text = " ".join(str(candidate or "").split())
        if route_question(text, target)[0] == "visual":
            return text
    return ""
