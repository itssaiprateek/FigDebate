from models.vision_model import Qwen3VLVisionModel
from models.language_model import MistralModel
from copy import deepcopy
import time
import re

from agents.visual_grounding import VisualGroundingAgent
from agents.claim_extraction import ClaimExtractionAgent

from arbiter.arbiter import Arbiter
from comparators.evidence_comparator import compare

from engine.feedback_loop import FeedbackLoop
from engine.evidence_verifier import AtomicEvidenceVerifier, merge_verified_evidence
from engine.evidence_ledger import (
    add_cross_agent_verified_relation,
    add_targeted_verifier_evidence,
    add_visual_reinspection_evidence,
    add_visual_witness_evidence,
    attach_evidence_audit,
    audit_decision,
    build_evidence_ledger,
    evidence_lifecycle_summary,
    is_active_evidence,
)
from engine.gpu_manager import GPUManager
from engine.review_board import attach_final_review, review_revision
from engine.question_router import build_question_plan, compile_visual_question
from engine.relation_semantics import is_missing_evidence_only
from engine.decision_trace import (
    append_decision_checkpoint,
    attach_decision_trace,
)

class DebateEngine:

    def __init__(self, enable_round3=False):

        self.feedback_loop = FeedbackLoop()
        self.enable_round3 = enable_round3
        self.nli_verifier = None

        print("[Debate] Ready.")

    # -------------------------------------------------------
    # Decide whether debate is required
    # -------------------------------------------------------

    def should_debate(
        self,
        decision,
        comparison=None,
    ):

        return self.debate_trigger_reason(decision, comparison) is not None

    @staticmethod
    def debate_assessment(decision, comparison=None):
        if not decision.get("_final_decision_valid", decision.get("label") in {"ENTAILS", "CONTRADICTS"}):
            # The primary Arbiter output is malformed, so debate has no
            # evidence-based decision to review.
            return {"trigger": False, "reason": None, "level": 0, "score": 0, "signals": []}

        if not comparison:
            return {"trigger": False, "reason": None, "level": 0, "score": 0, "signals": []}

        status = comparison.get("required_evidence_status")
        label = decision.get("label")
        has_support = bool(comparison.get("supporting_evidence"))
        has_conflict = bool(comparison.get("contradicting_evidence"))
        has_support_candidate = bool(
            comparison.get("relation_support_candidates")
        )
        has_conflict_candidate = bool(
            comparison.get("relation_conflict_candidates")
        )
        has_grounded_anchors = bool(comparison.get("grounded_anchor_evidence"))
        signals = []
        score = 0

        symbolic_relation_review = bool(
            comparison.get("has_symbolic_evidence", False)
            and (
                comparison.get("claim_relation", {}) or {}
            ).get("relation_family") == "sentiment"
        )
        if (
            comparison.get("relation_binding_required", False)
            and not comparison.get("relation_binding_observed", False)
            and not symbolic_relation_review
        ):
            pair_eligible = comparison.get(
                "region_pair_verifier_eligible", True
            )
            return {
                "trigger": True,
                "reason": (
                    "unresolved_text_layout_binding"
                    if pair_eligible
                    else "unresolved_text_relation_semantics"
                ),
                "level": 2,
                "score": 5,
                "signals": [
                    "unresolved_text_layout_binding"
                    if pair_eligible
                    else "unresolved_text_relation_semantics"
                ],
            }

        if (
            status == "INSUFFICIENT_VISUAL_EVIDENCE"
            or not comparison.get("visual_schema_complete", True)
        ):
            return {
                "trigger": True,
                "reason": "insufficient_visual_evidence",
                "level": 2,
                "score": 5,
                "signals": ["insufficient_visual_evidence"],
            }

        # Debate is an evidence-disagreement review, not a one-directional
        # label flipper. It can challenge either an unsupported CONTRADICTS
        # decision or an unsupported ENTAILS decision, including a recovered
        # binary-format decision, but only when the comparator has a concrete
        # direct signal to review.
        if status == "SUPPORTED" and has_support and label == "CONTRADICTS":
            return {"trigger": True, "reason": "comparator_support_disagrees_with_contradicts", "level": 1, "score": 5, "signals": ["direct_evidence_disagreement"]}
        if status == "CONFLICTING" and has_conflict and label == "ENTAILS":
            return {"trigger": True, "reason": "comparator_conflict_disagrees_with_entails", "level": 1, "score": 5, "signals": ["direct_evidence_disagreement"]}
        if status == "MIXED_VERIFIED_EVIDENCE" and has_support and has_conflict:
            return {"trigger": True, "reason": "mixed_verified_evidence_requires_review", "level": 1, "score": 5, "signals": ["mixed_verified_evidence"]}
        if status == "MIXED_RELATION_CANDIDATES" and (
            has_support_candidate and has_conflict_candidate
        ):
            return {
                "trigger": True,
                "reason": "mixed_relation_candidates_require_verification",
                "level": 2,
                "score": 5,
                "signals": ["mixed_relation_candidates"],
            }
        if status in {"SUPPORT_CANDIDATE", "CONFLICT_CANDIDATE"}:
            return {
                "trigger": True,
                "reason": "directional_candidate_requires_verification",
                "level": 2,
                "score": 4,
                "signals": ["directional_candidate_requires_verification"],
            }
        if (
            status == "GROUNDED_REVIEW_REQUIRED"
            and has_grounded_anchors
            and decision.get("debate_needed", False)
        ):
            score += 2
            signals.append("grounded_anchors_low_confidence")
        if (
            status == "SEMANTIC_REVIEW_REQUIRED"
            and comparison.get("has_visual_relations", False)
            and decision.get("_binary_resolution_raw_confidence", 1.0) < 0.55
        ):
            score += 2
            signals.append("visual_relation_close_semantic_decision")
        if (
            status == "SEMANTIC_REVIEW_REQUIRED"
            and comparison.get("has_visible_text", False)
            and comparison.get("has_visual_relations", False)
            and not has_support
            and not has_conflict
        ):
            score += 2
            signals.append("unresolved_text_relation_semantics")

        relation = comparison.get("claim_relation", {}) or {}
        candidates = comparison.get("structured_relation_candidates", []) or []
        verification = comparison.get("atomic_evidence_verification", {}) or {}
        if relation.get("resolved") and not verification.get("verified_count", 0):
            score += 1
            signals.append("structured_claim_without_verified_direction")
        if candidates and not verification.get("verified_count", 0):
            score += 2
            signals.append("uncorroborated_structured_relation")
        if (
            relation.get("resolved")
            and comparison.get("has_symbolic_evidence", False)
            and not has_support
            and not has_conflict
        ):
            score += 2
            signals.append("figurative_symbol_requires_visual_reinspection")

        assessment_text = " ".join(
            str(decision.get(key, ""))
            for key in ("_arbiter_assessment", "explanation")
        ).lower()
        if status in {"GROUNDED_REVIEW_REQUIRED", "SEMANTIC_REVIEW_REQUIRED"} and any(
            phrase in assessment_text
            for phrase in ("insufficient", "does not directly", "unclear relation")
        ):
            score += 1
            signals.append("arbiter_relation_insufficient")

        feedback_warning = comparison.get("feedback_warning")
        if feedback_warning:
            score += 1
            signals.append("verified_memory_warning")

        if score < 2:
            return {"trigger": False, "reason": None, "level": 0, "score": score, "signals": signals}
        level2_signals = {
            "uncorroborated_structured_relation",
            "unresolved_text_relation_semantics",
            "figurative_symbol_requires_visual_reinspection",
            "insufficient_visual_evidence",
        }
        level = 2 if level2_signals.intersection(signals) else 1
        reason = signals[0]
        return {"trigger": True, "reason": reason, "level": level, "score": score, "signals": signals}

    @classmethod
    def debate_trigger_reason(cls, decision, comparison=None):
        return cls.debate_assessment(decision, comparison).get("reason")

    @staticmethod
    def _cached_evidence_critique(evidence_ledger):
        """Build an independent Level-1 recommendation from verified evidence."""
        by_relation = {}
        for relation in ("SUPPORT", "CONFLICT"):
            by_relation[relation] = [
                item for item in (evidence_ledger or [])
                if item.get("grounded", False)
                and is_active_evidence(item)
                and item.get("relation") == relation
                and (
                    item.get("decision_grade", False)
                    or item.get("verification", {}).get(
                        "decision_grade", False
                    )
                )
            ]
        support = by_relation["SUPPORT"]
        conflict = by_relation["CONFLICT"]
        if bool(support) == bool(conflict):
            return {
                "stance": "UNRESOLVED",
                "recommendation": "ABSTAIN",
                "reason": (
                    "Decision-grade support and conflict are both present."
                    if support else
                    "No decision-grade directional evidence is available."
                ),
                "specific_evidence": False,
                "method": "independent_cached_evidence_review",
                "review_method": "independent_cached_evidence_review",
                "_seconds": 0.0,
            }
        item = (support or conflict)[0]
        recommendation = "ENTAILS" if support else "CONTRADICTS"
        return {
            "stance": "UNRESOLVED",
            "recommendation": recommendation,
            "reason": f"[{item['id']}] {item['text']}",
            "specific_evidence": True,
            "method": "independent_cached_evidence_review",
            "review_method": "independent_cached_evidence_review",
            "_seconds": 0.0,
        }

    @staticmethod
    def _stance_for_recommendation(recommendation, original_label):
        recommendation = str(recommendation or "").upper()
        if recommendation not in {"ENTAILS", "CONTRADICTS"}:
            return "UNRESOLVED"
        return "ENDORSE" if recommendation == original_label else "CHALLENGE"

    @staticmethod
    def _advocate_summary(evidence_ledger):
        def entries(relation):
            return [
                {"id": item.get("id"), "text": item.get("text")}
                for item in (evidence_ledger or [])
                if item.get("grounded", False)
                and is_active_evidence(item)
                and item.get("relation") == relation
                and (
                    item.get("decision_grade", False)
                    or item.get("verification", {}).get(
                        "decision_grade", False
                    )
                )
            ]
        return {
            "entails": entries("SUPPORT"),
            "contradicts": entries("CONFLICT"),
            "neutral_or_anchor": [
                {"id": item.get("id"), "text": item.get("text")}
                for item in (evidence_ledger or [])
                if item.get("grounded", False)
                and item.get("relation") in {"NEUTRAL", "ANCHOR"}
            ],
        }

    @staticmethod
    def _comparison_with_review_evidence(comparison, evidence_ledger):
        """Expose new current-image evidence to the debate Arbiter by ID."""
        output = deepcopy(comparison or {})
        catalog = list(output.get("grounded_evidence_catalog", []) or [])
        catalog_ids = {item.get("id") for item in catalog}
        support = list(output.get("supporting_evidence", []) or [])
        conflict = list(output.get("contradicting_evidence", []) or [])

        for item in evidence_ledger or []:
            if (
                not item.get("grounded", False)
                or not item.get("id")
                or not is_active_evidence(item)
            ):
                continue
            if item["id"] not in catalog_ids:
                catalog.append({
                    "id": item["id"],
                    "source": item.get("source"),
                    "type": item.get("type"),
                    "text": item.get("text", ""),
                    "relation": item.get("relation"),
                    "decision_grade": bool(
                        item.get("decision_grade", False)
                        or item.get("verification", {}).get(
                            "decision_grade", False
                        )
                    ),
                    "verification_method": item.get(
                        "verification_method"
                    ),
                    "lifecycle_status": item.get(
                        "lifecycle_status", "ACTIVE"
                    ),
                })
                catalog_ids.add(item["id"])
            if not (
                item.get("decision_grade", False)
                or item.get("verification", {}).get(
                    "decision_grade", False
                )
            ):
                continue
            text = f"[VISUAL][{item['id']}] {item.get('text', '')}"
            if item.get("relation") == "SUPPORT" and text not in support:
                support.append(text)
            elif item.get("relation") == "CONFLICT" and text not in conflict:
                conflict.append(text)

        output["grounded_evidence_catalog"] = catalog
        output["supporting_evidence"] = support
        output["contradicting_evidence"] = conflict
        output["supporting_points"] = support
        output["conflicting_points"] = conflict
        if support and conflict:
            output["required_evidence_status"] = "MIXED_VERIFIED_EVIDENCE"
            output["recommendation"] = "REVIEW_MIXED_EVIDENCE"
        elif support:
            output["required_evidence_status"] = "SUPPORTED"
            output["recommendation"] = "LEAN_ENTAILS"
        elif conflict:
            output["required_evidence_status"] = "CONFLICTING"
            output["recommendation"] = "LEAN_CONTRADICTS"
        return output

    @staticmethod
    def _has_direct_evidence(items):
        return any(
            str(item).strip()
            and "none" not in str(item).lower()
            and "[missing]" not in str(item).lower()
            for item in (items or [])
        )

    @staticmethod
    def _apply_unopposed_visual_evidence(decision, evidence_ledger, critique):
        """Align a proposal with uniquely directional reinspection evidence."""
        output = dict(decision or {})
        output["_visual_evidence_consensus_applied"] = False
        if not (
            isinstance(critique, dict)
            and critique.get("_format_valid", False)
            and critique.get("specific_evidence", False)
        ):
            return output
        recommendation = str(critique.get("recommendation", "")).upper()
        relation = {"ENTAILS": "SUPPORT", "CONTRADICTS": "CONFLICT"}.get(
            recommendation
        )
        if not relation:
            return output

        by_relation = {"SUPPORT": [], "CONFLICT": []}
        for item in evidence_ledger or []:
            item_relation = item.get("relation")
            decision_grade = bool(
                item.get("decision_grade", False)
                or item.get("verification", {}).get("decision_grade", False)
            )
            if (
                item_relation in by_relation
                and item.get("grounded", False)
                and is_active_evidence(item)
                and decision_grade
                and item.get("id")
            ):
                by_relation[item_relation].append(item["id"])
        opposite = "CONFLICT" if relation == "SUPPORT" else "SUPPORT"
        if not by_relation[relation] or by_relation[opposite]:
            return output

        output["_semantic_proposed_label"] = output.get("label")
        output["label"] = recommendation
        output["decision_method"] = "visual_reinspection_consensus"
        output["confidence"] = min(
            0.72, max(0.60, float(output.get("confidence") or 0.60))
        )
        output["_model_cited_evidence_ids"] = list(by_relation[relation])
        output["explanation"] = (
            "Independent current-image reinspection supplied unopposed "
            f"decision-grade {relation.lower()} evidence: "
            + str(critique.get("reason", ""))
        )
        output["_visual_evidence_consensus_applied"] = True
        return output

    @staticmethod
    def _enforce_revision_requirements(
        original_decision, revised_decision, agent1_critique,
        agent2_critique, debate_level,
    ):
        """Prevent a linguistic-only or malformed-evidence debate flip."""
        output = dict(revised_decision or {})
        targeted = output.get("_targeted_region_verification", {}) or {}
        response_status = (agent1_critique or {}).get("response_status")
        legacy_visual_valid = bool(
            (agent1_critique or {}).get("_format_valid", False)
            and (agent1_critique or {}).get("specific_evidence", False)
            and response_status is None
        )
        visual_ready = bool(
            response_status == "VALID_DIRECTIONAL_ANSWER"
            or legacy_visual_valid
            or targeted.get("decision_grade", False)
        )
        deficiencies = []
        if int(debate_level) >= 2 and not visual_ready:
            if response_status in {
                "FORMAT_FAILURE", "TRUNCATED_RESPONSE", "INVALID_ENUM",
                "INCONSISTENT_FIELDS",
            }:
                deficiencies.append("AGENT1_FORMAT_FAILURE")
            elif response_status == "QUESTION_NOT_ANSWERED":
                deficiencies.append("QUESTION_NOT_ANSWERED")
            elif response_status == "SEMANTICALLY_UNRESOLVED":
                deficiencies.append("INSUFFICIENT_DIRECTIONAL_EVIDENCE")
            else:
                deficiencies.append("MISSING_VISUAL_OBSERVATION")
        if not (agent2_critique or {}).get("requirements_valid", True):
            deficiencies.append("CLAIM_REQUIREMENTS_INVALID")
        if not deficiencies:
            return output
        output.setdefault("_unconstrained_proposed_label", output.get("label"))
        output.setdefault(
            "_raw_debate_proposed_label",
            output.get("_unconstrained_proposed_label"),
        )
        output["label"] = (original_decision or {}).get("label")
        output["confidence"] = (original_decision or {}).get(
            "confidence", output.get("confidence", 0.5)
        )
        output["decision_method"] = "no_visual_revision"
        output["_revision_status"] = "NO_VISUAL_REVISION"
        output["_relation_status"] = "INSUFFICIENT"
        output["_deficiencies"] = list(dict.fromkeys(
            deficiencies + list(output.get("_deficiencies", []) or [])
        ))
        output["_model_cited_evidence_ids"] = []
        output["explanation"] = (
            "No directional debate revision was permitted because the required "
            "independent visual or claim evidence was not valid."
        )
        return output

    @staticmethod
    def _is_absence_based(reason):
        return is_missing_evidence_only(reason)

    @staticmethod
    def _accept_revision(
        original_decision,
        revised_decision,
        comparison,
        evidence_ledger=None,
        mediation=None,
    ):
        """Compatibility wrapper around the deterministic review board."""
        accepted, reason, audit = review_revision(
            original_decision,
            revised_decision,
            evidence_ledger or [],
            visual_review=revised_decision.get(
                "_debate_agent1_critique", {}
            ),
            claim_contract=(comparison or {}).get("claim_contract", {}),
            mediation=mediation,
        )
        # Preserve the complete deterministic audit without changing this
        # long-standing two-value compatibility interface.
        revised_decision["_revision_acceptance_audit"] = audit
        return accepted, reason

    def run_debate(
        self,
        image,
        caption,
        visual_output,
        language_output,
        comparison,
        decision,
        evidence_ledger=None,
        metadata=None,
        debate_assessment=None,
    ):
        debate_assessment = debate_assessment or {}
        results = self.run_debate_batch([{
            "key": "single",
            "image": image,
            "caption": caption,
            "visual_output": visual_output,
            "language_output": language_output,
            "comparison": comparison,
            "decision": decision,
            "evidence_ledger": evidence_ledger or [],
            "debate_level": debate_assessment.get("level", 2),
            "debate_score": debate_assessment.get("score", 0),
            "debate_signals": debate_assessment.get("signals", []),
            "mediation_plan": (metadata or {}).get("mediation_plan", {}),
        }])
        revised_decision = results["single"]
        return revised_decision, revised_decision["_debate"]["rounds"]

    def run_debate_batch(self, cases, language_runtime=None):
        """Run one debate revision per case while reusing each loaded model."""
        if not cases:
            self.last_batch_timing = {}
            return {}

        results = {}
        agent1_critiques = {}
        recovered_visual_outputs = {}
        recovered_comparisons = {}
        recovered_evidence_verifications = {}
        vision_load_seconds = 0.0
        language_load_seconds = 0.0

        level2_cases = [
            case for case in cases
            if int(case.get("debate_level", 2)) >= 2
        ]
        visual_review_cases = [
            case for case in cases
            if case in level2_cases or case.get("force_visual_review", False)
        ]
        cached_review_cases = [
            case for case in cases if case not in visual_review_cases
        ]
        for case in cached_review_cases:
            agent1_critiques[case["key"]] = self._cached_evidence_critique(
                case.get("evidence_ledger", [])
            )

        print(
            "\nLoading Qwen3-VL for targeted debate batch..."
            if visual_review_cases
            else "\nSkipping Qwen3-VL: Level 1 debate uses cached evidence."
        )
        vision_runtime = None
        agent1 = None
        if visual_review_cases:
            try:
                load_started = time.time()
                vision_runtime = Qwen3VLVisionModel()
                vision_load_seconds = time.time() - load_started
                agent1 = VisualGroundingAgent(vision_runtime)
                for case in visual_review_cases:
                    debate_signals = list(
                        case.get("debate_signals", []) or []
                    )
                    # A tribunal follow-up already has a compact, image-bound
                    # atomic question. Re-running the broad claim recovery
                    # before answering it repeats a full visual pass, consumes
                    # VRAM/time, and does not add independent evidence.
                    if (
                        int(case.get("debate_level", 2)) >= 2
                        and "tribunal_follow_up" not in debate_signals
                        and not case.get("tribunal_hearing", False)
                    ):
                        recovery_reason = next(iter(
                            debate_signals or [
                                "targeted_grounding_recovery"
                            ]
                        ))
                        recovered_visual = agent1.recover_for_claim(
                            case["image"],
                            case.get("visual_output", {}),
                            (case.get("comparison", {}) or {}).get(
                                "claim_relation", {}
                            ),
                            recovery_reason=recovery_reason,
                        )
                    else:
                        recovered_visual = {}
                    if recovered_visual.get("_targeted_recovery_success"):
                        superseded_ledger = deepcopy(
                            case.get("evidence_ledger", []) or []
                        )
                        for item in superseded_ledger:
                            if is_active_evidence(item):
                                item["lifecycle_status"] = "SUPERSEDED"
                        case["superseded_evidence_ledger"] = superseded_ledger
                        previous_comparison = case.get("comparison", {}) or {}
                        recovered_comparison = compare(
                            recovered_visual,
                            case.get("language_output", {}),
                            caption=case.get("caption", ""),
                        )
                        if previous_comparison.get("feedback_warning"):
                            recovered_comparison["feedback_warning"] = (
                                previous_comparison["feedback_warning"]
                            )
                        recovered_comparison["recovery_trigger_reason"] = (
                            recovery_reason
                        )
                        recovered_comparison[
                            "targeted_region_review_recommended"
                        ] = bool(
                            recovered_comparison.get(
                                "region_pair_verifier_eligible", False
                            )
                        )
                        recovered_ledger = build_evidence_ledger(
                            recovered_visual,
                            case.get("language_output", {}),
                            recovered_comparison,
                        )
                        if self.nli_verifier is not None:
                            recovered_verifier = AtomicEvidenceVerifier(
                                nli_verifier=self.nli_verifier
                            )
                            recovered_ledger, recovered_verification = (
                                recovered_verifier.verify(
                                    recovered_ledger,
                                    case.get("language_output", {}),
                                    recovered_comparison,
                                )
                            )
                            recovered_comparison = merge_verified_evidence(
                                recovered_comparison,
                                recovered_ledger,
                                recovered_verification,
                            )
                            recovered_evidence_verifications[case["key"]] = (
                                recovered_verification
                            )
                        case["visual_output"] = recovered_visual
                        case["comparison"] = recovered_comparison
                        case["evidence_ledger"] = recovered_ledger
                        recovered_visual_outputs[case["key"]] = recovered_visual
                        recovered_comparisons[case["key"]] = recovered_comparison
                    critique_started = time.time()
                    challenge_prompt = self.build_agent1_challenge_prompt(
                            case["visual_output"],
                            case["decision"],
                            case.get("comparison"),
                            case.get("mediation_plan"),
                        )
                    if case.get("tribunal_hearing", False):
                        challenge_prompt += (
                            "\nTRIBUNAL_VISUAL_WITNESS_ONLY: Report the direct "
                            "observation only. Do not assign its semantic relation "
                            "to the caption; the tribunal performs that step.\n"
                        )
                    critique = agent1.critique(
                        case["image"],
                        challenge_prompt,
                    )
                    critique["_seconds"] = time.time() - critique_started
                    agent1_critiques[case["key"]] = critique
            finally:
                if agent1 is not None:
                    del agent1
                if vision_runtime is not None:
                    del vision_runtime
                GPUManager.clear()

        owns_language_runtime = language_runtime is None
        print(
            "\nLoading Mistral for debate batch..."
            if owns_language_runtime
            else "\nReusing the loaded Mistral for Level 1 debate."
        )
        mistral = None
        agent2 = (
            language_runtime.get("agent2") if language_runtime else None
        )
        arbiter = (
            language_runtime.get("arbiter") if language_runtime else None
        )
        try:
            if owns_language_runtime:
                load_started = time.time()
                mistral = MistralModel()
                language_load_seconds = time.time() - load_started
                agent2 = ClaimExtractionAgent(mistral.model, mistral.tokenizer)
                arbiter = Arbiter(
                    mistral.model,
                    mistral.tokenizer,
                    nli_verifier=self.nli_verifier,
                )

            for case in cases:
                agent1_critique = agent1_critiques[case["key"]]
                agent1_critique["stance"] = self._stance_for_recommendation(
                    agent1_critique.get("recommendation"),
                    case["decision"].get("label"),
                )
                original_ledger = case.get("evidence_ledger", []) or []
                review_ledger = add_visual_witness_evidence(
                    original_ledger, agent1_critique
                )
                if (
                    int(case.get("debate_level", 2)) >= 2
                    and agent1_critique.get("review_method")
                    != "independent_cached_evidence_review"
                ):
                    review_ledger = add_visual_reinspection_evidence(
                        review_ledger,
                        agent1_critique,
                        case.get("comparison", {}),
                    )
                review_comparison = self._comparison_with_review_evidence(
                    case.get("comparison", {}), review_ledger
                )
                mediation_plan = case.get("mediation_plan", {}) or {}
                if mediation_plan.get("_usable", False):
                    review_comparison["mediation_questions"] = list(dict.fromkeys(
                        list(mediation_plan.get("disputed_issues", []) or [])
                        + list(mediation_plan.get("verification_requests", []) or [])
                    ))
                advocate_summary = self._advocate_summary(
                    review_ledger
                )
                agent1_critique["advocates"] = advocate_summary
                critique_started = time.time()
                agent2_critique = agent2.critique(
                    case["caption"],
                    self.build_agent2_challenge_prompt(
                        case["language_output"], case["decision"],
                        mediation_plan,
                    ),
                )
                agent2_critique["_seconds"] = time.time() - critique_started

                review_ledger, cross_agent_verification = (
                    add_cross_agent_verified_relation(
                        review_ledger,
                        agent1_critique,
                        agent2_critique,
                        (case.get("comparison") or {}).get(
                            "claim_contract", {}
                        ),
                    )
                )
                if case.get("tribunal_hearing", False):
                    # In tribunal mode the agents are witnesses, not a hidden
                    # first decision court. Preserve the current decision and
                    # pass the typed testimony and any independently promoted
                    # relation to the tribunal.
                    decision = attach_evidence_audit(
                        dict(case["decision"]), review_ledger
                    )
                    decision = attach_final_review(
                        decision,
                        review_ledger,
                        (case.get("comparison") or {}).get(
                            "claim_contract", {}
                        ),
                    )
                    decision_trace = append_decision_checkpoint(
                        case["decision"].get("_decision_trace", []),
                        "tribunal_witness_hearing_recorded",
                        decision,
                        ledger=review_ledger,
                        metadata={
                            "agent1_response_status": agent1_critique.get(
                                "response_status"
                            ),
                            "agent2_requirements_valid": agent2_critique.get(
                                "requirements_valid", True
                            ),
                        },
                    )
                    decision = attach_decision_trace(decision, decision_trace)
                    inference_seconds = (
                        agent1_critique["_seconds"]
                        + agent2_critique["_seconds"]
                    )
                    decision["_debate"] = {
                        "architecture": "tribunal_targeted_hearing",
                        "agent1_critique": agent1_critique,
                        "agent2_critique": agent2_critique,
                        "rounds": 1,
                        "level": 2,
                        "need_score": case.get("debate_score", 0),
                        "need_signals": case.get("debate_signals", []),
                        "inference_seconds": round(inference_seconds, 4),
                        "revision_accepted": False,
                        "revision_acceptance_reason": "tribunal_controls_resolution",
                        "revision_acceptance_audit": {},
                        "proposed_label": "",
                        "unconstrained_proposed_label": "",
                        "agent1_response_status": agent1_critique.get(
                            "response_status"
                        ),
                        "agent2_requirements_valid": agent2_critique.get(
                            "requirements_valid", True
                        ),
                        "cross_agent_verification": cross_agent_verification,
                        "relation_status": "HEARING_PENDING_TRIBUNAL",
                        "deficiencies": [],
                        "evidence_lifecycle": evidence_lifecycle_summary(
                            review_ledger
                        ),
                        "review_status": "hearing_recorded",
                        "original_evidence_audit": audit_decision(
                            case["decision"], original_ledger
                        ),
                        "proposed_evidence_audit": decision.get(
                            "_evidence_audit", {}
                        ),
                        "evidence_ledger_before": original_ledger,
                        "evidence_ledger_after": review_ledger,
                        "proposed_decision": {},
                        "mediation": mediation_plan,
                    }
                    decision["_evidence_ledger"] = review_ledger
                    results[case["key"]] = decision
                    continue

                # The Arbiter must see the newly verified relation during its
                # reasoning pass, not only after it has already selected a
                # label.  Mediation questions remain advisory metadata.
                review_comparison = self._comparison_with_review_evidence(
                    case.get("comparison", {}), review_ledger
                )
                if mediation_plan.get("_usable", False):
                    review_comparison["mediation_questions"] = list(
                        dict.fromkeys(
                            list(mediation_plan.get("disputed_issues", []) or [])
                            + list(
                                mediation_plan.get(
                                    "verification_requests", []
                                ) or []
                            )
                        )
                    )
                advocate_summary = self._advocate_summary(review_ledger)
                agent1_critique["advocates"] = advocate_summary

                revised_decision = arbiter.analyze(
                    case["caption"],
                    case["visual_output"],
                    case["language_output"],
                    review_comparison,
                    agent1_critique=agent1_critique,
                    agent2_critique=agent2_critique,
                    previous_decision=case["decision"],
                )
                revised_decision["_raw_debate_proposed_label"] = (
                    revised_decision.get(
                        "_unconstrained_proposed_label",
                        revised_decision.get("label"),
                    )
                )
                decision_trace = append_decision_checkpoint(
                    case["decision"].get("_decision_trace", []),
                    "raw_debate_proposal",
                    revised_decision,
                    ledger=review_ledger,
                )
                revised_decision["_debate_agent1_critique"] = agent1_critique
                revised_ledger = add_targeted_verifier_evidence(
                    review_ledger,
                    agent1_critique,
                    revised_decision,
                )
                revised_decision = self._enforce_revision_requirements(
                    case["decision"], revised_decision, agent1_critique,
                    agent2_critique, case.get("debate_level", 2),
                )
                decision_trace = append_decision_checkpoint(
                    decision_trace,
                    "evidence_constrained_proposal",
                    revised_decision,
                    ledger=revised_ledger,
                )
                targeted_verification = revised_decision.get(
                    "_targeted_region_verification", {}
                ) or {}
                revised_decision = self._apply_unopposed_visual_evidence(
                    revised_decision,
                    revised_ledger,
                    agent1_critique,
                )
                revised_decision = attach_evidence_audit(
                    revised_decision,
                    revised_ledger,
                )
                if (
                    targeted_verification.get("decision_grade", False)
                    and agent1_critique.get("recommendation") == "ABSTAIN"
                ):
                    agent1_critique["recommendation"] = revised_decision.get(
                        "label"
                    )
                    agent1_critique["stance"] = self._stance_for_recommendation(
                        revised_decision.get("label"),
                        case["decision"].get("label"),
                    )
                revised_decision["_debate_agent1_critique"] = agent1_critique
                accepted, reason = self._accept_revision(
                    case["decision"],
                    revised_decision,
                    review_comparison,
                    revised_ledger,
                    mediation=mediation_plan,
                )
                decision = (
                    revised_decision
                    if accepted
                    else attach_evidence_audit(
                        dict(case["decision"]), revised_ledger
                    )
                )
                decision = attach_final_review(
                    decision,
                    revised_ledger,
                    (case.get("comparison") or {}).get("claim_contract", {}),
                )
                decision_trace = append_decision_checkpoint(
                    decision_trace,
                    "debate_review_accepted" if accepted else "debate_review_preserved",
                    decision,
                    ledger=revised_ledger,
                    metadata={
                        "accepted": accepted,
                        "reason": reason,
                        "failed_invariant": revised_decision.get(
                            "_revision_acceptance_audit", {}
                        ).get("failed_invariant", ""),
                    },
                )
                decision = attach_decision_trace(decision, decision_trace)
                inference_seconds = (
                    agent1_critique["_seconds"]
                    + agent2_critique["_seconds"]
                    + revised_decision.get("_timing", {}).get("total_seconds", 0.0)
                    + float(
                        recovered_visual_outputs.get(
                            case["key"], {}
                        ).get("_targeted_recovery_seconds", 0.0)
                    )
                )
                decision["_debate"] = {
                    "agent1_critique": agent1_critique,
                    "agent2_critique": agent2_critique,
                    "rounds": 2,
                    "level": int(case.get("debate_level", 2)),
                    "need_score": case.get("debate_score", 0),
                    "need_signals": case.get("debate_signals", []),
                    "advocates": advocate_summary,
                    "inference_seconds": round(inference_seconds, 4),
                    "grounding_recovery_seconds": float(
                        recovered_visual_outputs.get(
                            case["key"], {}
                        ).get("_targeted_recovery_seconds", 0.0)
                    ),
                    "revision_accepted": accepted,
                    "revision_acceptance_reason": reason,
                    "revision_acceptance_audit": revised_decision.get(
                        "_revision_acceptance_audit", {}
                    ),
                    "proposed_label": revised_decision.get("label"),
                    "unconstrained_proposed_label": revised_decision.get(
                        "_raw_debate_proposed_label",
                        revised_decision.get("label"),
                    ),
                    "agent1_response_status": agent1_critique.get(
                        "response_status"
                    ),
                    "agent2_requirements_valid": agent2_critique.get(
                        "requirements_valid", True
                    ),
                    "cross_agent_verification": cross_agent_verification,
                    "relation_status": revised_decision.get(
                        "_relation_status"
                    ),
                    "deficiencies": revised_decision.get(
                        "_deficiencies", []
                    ),
                    "evidence_lifecycle": evidence_lifecycle_summary(
                        list(case.get("superseded_evidence_ledger", []) or [])
                        + list(revised_ledger or [])
                    ),
                    "superseded_evidence_ledger": case.get(
                        "superseded_evidence_ledger", []
                    ),
                    "review_status": (
                        "accepted" if accepted
                        else "no_change" if reason == "unchanged"
                        else "rejected"
                    ),
                    "original_evidence_audit": audit_decision(
                        case["decision"], original_ledger
                    ),
                    "proposed_evidence_audit": revised_decision.get(
                        "_evidence_audit", {}
                    ),
                    "evidence_ledger_before": original_ledger,
                    "evidence_ledger_after": revised_ledger,
                    "proposed_decision": dict(revised_decision),
                    "mediation": mediation_plan,
                    "recovered_visual_output": recovered_visual_outputs.get(
                        case["key"]
                    ),
                    "recovered_comparison": recovered_comparisons.get(
                        case["key"]
                    ),
                    "recovered_evidence_verification": (
                        recovered_evidence_verifications.get(case["key"])
                    ),
                }
                results[case["key"]] = decision
        finally:
            if owns_language_runtime:
                if agent2 is not None:
                    del agent2
                if arbiter is not None:
                    del arbiter
                if mistral is not None:
                    del mistral
                GPUManager.clear()

        self.last_batch_timing = {
            "vision_model_load_seconds": round(vision_load_seconds, 4),
            "language_model_load_seconds": round(language_load_seconds, 4),
        }
        return results

    # -------------------------------------------------------
    # Existing feedback loop
    # -------------------------------------------------------

    def generate_feedback(
        self,
        visual_output,
        language_output,
        comparison,
        decision,
        ground_truth=None,
        phenomenon=None,
        metadata=None,
        apply_calibration=False,
        evidence_ledger=None,
    ):

        return self.feedback_loop.generate_feedback(

            visual_output,

            language_output,

            comparison,

            decision,

            ground_truth=ground_truth,

            phenomenon=phenomenon,

            metadata=metadata,

            apply_calibration=apply_calibration,

            evidence_ledger=evidence_ledger,

        )

    def generate_feedback_batch(self, history):

        return self.feedback_loop.generate_feedback_batch(history)

    # -------------------------------------------------------
    # Prompt for Agent 1 (Visual Grounding)
    # -------------------------------------------------------

    def build_agent1_challenge_prompt(
        self,
        visual_output,
        decision,
        comparison=None,
        mediation=None,
    ):

        relation_focus = ""
        if (
            comparison
            and comparison.get("required_evidence_status")
            == "INSUFFICIENT_VISUAL_EVIDENCE"
        ):
            relation_focus = """
Review mode: INSUFFICIENT_VISUAL_EVIDENCE
The first grounding record was empty or malformed. Reconstruct the literal
scene, inspect all text-bearing regions, and state one complete entity-to-state
or entity-to-text observation. Do not infer the missing observation from the
caption. Use ABSTAIN if the image remains unreadable.
"""
        elif (
            comparison
            and comparison.get("region_pair_verifier_eligible", True)
            and (
                not comparison.get("relation_binding_observed", False)
                or comparison.get(
                    "targeted_region_review_recommended", False
                )
            )
        ):
            relation_focus = """
Review mode: UNRESOLVED_TEXT_LAYOUT_BINDING
Disputed issue: OCR phrases were detected but were not reliably assigned to
the compared panels or objects. Reinspect the image and state the exact
left/right, top/bottom, or object-to-text bindings. Include both sides in the
Reason. A statement that there is merely "no clear relation" is insufficient.
"""
        elif comparison and comparison.get("structured_relation_candidates"):
            claim_relation = comparison.get("claim_relation", {}) or {}
            candidates = comparison.get("structured_relation_candidates", []) or []
            candidate_lines = "\n".join(
                f"- Proposed {item.get('proposed_relation')}: {item.get('text')} "
                f"(cues: {', '.join(item.get('matched_cues', []))})"
                for item in candidates[:4]
            )
            relation_focus = f"""
Review mode: UNCORROBORATED_STRUCTURED_RELATION
Caption claim: {claim_relation.get('claim_text', '')}
Relation family: {claim_relation.get('relation_family', 'unresolved')}
Candidate current-image relations:
{candidate_lines}

Reinspect whether the nominated state is actually visible and applies to the
caption subject. Reject a nomination based only on a shared word. State the
exact observed state and its subject in the Reason.
"""
        elif (
            comparison
            and comparison.get("required_evidence_status")
            == "SEMANTIC_REVIEW_REQUIRED"
            and comparison.get("has_visible_text", False)
            and not (
                comparison.get("has_symbolic_evidence", False)
                and (
                    comparison.get("claim_relation", {}) or {}
                ).get("relation_family") == "sentiment"
            )
        ):
            relation_focus = """
Review mode: UNRESOLVED_TEXT_RELATION_SEMANTICS
Re-read every identity or category label exactly as printed and bind it to the
correct person, object, or region. Then state the visible relationship between
those labeled elements. Do not replace a printed label with clothing or infer
a relation from outside knowledge. A literal caption may be instantiated by a
meme whose people or objects are explicitly labeled with the caption roles.
Judge that visible role mapping rather than requiring the literal-world object.
"""
        elif (
            comparison
            and comparison.get("relation_binding_required", False)
            and not comparison.get("relation_binding_observed", False)
            and not (
                comparison.get("has_symbolic_evidence", False)
                and (
                    comparison.get("claim_relation", {}) or {}
                ).get("relation_family") == "sentiment"
            )
        ):
            relation_focus = """
Review mode: UNRESOLVED_TEXT_RELATION_SEMANTICS
Re-read every identity or category label exactly as printed and bind it to the
correct person, object, or region. Then state the visible relationship between
those labeled elements. Review the complete image; do not assume it has a
two-column comparison layout.
"""
        elif comparison and comparison.get("has_symbolic_evidence", False):
            relation_focus = """
Review mode: FIGURATIVE_SYMBOL_REINSPECTION
Name the exact visible symbol, the subject or object it is attached to, and
its conventional association. State whether that observed association matches
the asserted property, expected state, or opposite state. Explicitly describe
the symbol's visible condition (for example whole, damaged, bright, dark) and
its attachment. Use ABSTAIN if the association is ambiguous or is not visibly
attached to the caption subject.
"""
        feedback_warning = (comparison or {}).get("feedback_warning", {}) or {}
        diagnostic_questions = feedback_warning.get(
            "diagnostic_questions", []
        ) or []
        if diagnostic_questions:
            relation_focus += "\nProcedural memory asks you to check:\n" + "\n".join(
                f"- {item}" for item in diagnostic_questions[:3]
            )
            relation_focus += (
                "\nThese are review questions, not evidence and not a label vote.\n"
            )
        plan = mediation or {}
        atomic_question = ""
        if plan.get("_usable", False):
            atomic_question = compile_visual_question(
                comparison or {}, plan
            )

        if not atomic_question and diagnostic_questions:
            atomic_question = str(diagnostic_questions[0])
        if not atomic_question:
            atomic_question = build_question_plan(
                comparison or {}
            ).agent1_question
        if not atomic_question:
            if "TEXT_LAYOUT_BINDING" in relation_focus:
                atomic_question = (
                    "Which exact text belongs to each visible panel or object?"
                )
            elif "TEXT_RELATION_SEMANTICS" in relation_focus:
                atomic_question = (
                    "Which visible entities do the printed labels identify, "
                    "and what relation is visibly shown between them?"
                )
            elif "FIGURATIVE_SYMBOL" in relation_focus:
                atomic_question = (
                    "What visible symbol is attached to the claim subject, "
                    "and what exact condition is that symbol in?"
                )
            else:
                atomic_question = (
                    "What exact visible state or relation is shown for the "
                    "caption subject?"
                )

        claim = (comparison or {}).get("claim_relation", {}) or {}

        def compact_items(name, values, limit=5):
            items = [
                " ".join(str(item).split())
                for item in (values or [])
                if str(item).strip()
            ]
            return f"{name}: " + ("; ".join(items[:limit]) or "None")

        visual_context = "\n".join((
            "Prior caption-blind grounding (advisory; verify against image):",
            f"Scene: {' '.join(str(visual_output.get('visual_description', '')).split())[:500] or 'None'}",
            compact_items("Objects", visual_output.get("objects", [])),
            compact_items("Visible text", visual_output.get("visible_text", [])),
            compact_items("Facts", visual_output.get("visual_facts", [])),
            compact_items("Relations", visual_output.get("visual_relations", [])),
            compact_items(
                "Possible visual metaphors",
                visual_output.get("possible_visual_metaphors", []),
            ),
            f"Symbolic tone: {' '.join(str(visual_output.get('symbolic_tone', '')).split())[:240] or 'None'}",
        ))
        return f"""
You are the independent Visual Evidence Reviewer.

You are not shown another agent's label. Reinspect the image and evaluate the
caption claim using current-image evidence only.

Question ID: agent1_{(comparison or {}).get('required_evidence_status', 'review').lower()}
Review question: {atomic_question}

Original caption: {claim.get('claim_text', 'Unavailable')}
Claim subject: {claim.get('subject', 'Unavailable')}
Asserted property: {claim.get('asserted_property', 'Unavailable')}
Relation family: {claim.get('relation_family', 'Unavailable')}
Expected visual state: {claim.get('expected_visual_state', 'Unavailable')}
Expected state cues: {', '.join(claim.get('expected_visual_cues', []) or []) or 'Unavailable'}
Opposite visual state: {claim.get('opposite_visual_state', 'Unavailable')}
Opposite state cues: {', '.join(claim.get('opposite_visual_cues', []) or []) or 'Unavailable'}

Visual Analysis

{visual_context}

{relation_focus}

Answer only the Review question from the current image. Missing evidence is
unclear, never contradiction.
"""

    # -------------------------------------------------------
    # Prompt for Agent 2 (Linguistic)
    # -------------------------------------------------------

    def build_agent2_challenge_prompt(
        self,
        language_output,
        decision,
        mediation=None,
    ):
        contract = language_output.get("claim_contract", {}) or {}

        def compact(value, limit=280):
            return " ".join(str(value or "None").split())[:limit]

        fields = (
            ("Original caption", language_output.get("original_caption")),
            ("Caption proposition", language_output.get("caption_proposition")),
            ("Claim subject", language_output.get("claim_subject")),
            ("Claim predicate", language_output.get("claim_predicate")),
            ("Claim object", language_output.get("claim_object")),
            ("Claim source", language_output.get("claim_source")),
            ("Claim target", language_output.get("claim_target")),
            ("Asserted property", language_output.get("asserted_property")),
            ("Caption polarity", language_output.get("caption_polarity")),
            ("Intended meaning", language_output.get("intended_meaning")),
            ("Relation family", language_output.get("relation_family")),
            ("Expected visual state", language_output.get("expected_visual_state")),
            ("Opposite visual state", language_output.get("opposite_visual_state")),
            ("Audit warnings", ", ".join(contract.get("warnings", []) or []) or "None"),
        )
        prompt = "\n".join(
            f"{name}: {compact(value)}" for name, value in fields
        )
        plan = mediation or {}
        if plan.get("_usable", False):
            questions = list(plan.get("agent2_questions", []) or [])[:4]
            if questions:
                prompt += (
                    "\nMediator claim checks (advisory questions, never evidence or a label):\n"
                    + "\n".join(f"- {item}" for item in questions)
                    + "\nResolve only the caption meaning, polarity, and relation."
                )
        return prompt

    # -------------------------------------------------------
    # Round 3 Trigger
    # -------------------------------------------------------

    def agents_disagree(
        self,
        agent1_response,
        agent2_response,
    ):

        return (

            agent1_response["stance"]

            !=

            agent2_response["stance"]

        )

    # -------------------------------------------------------
    # Round 3 Prompt
    # -------------------------------------------------------

    def build_round3_prompt(
        self,
        agent1_response,
        agent2_response,
    ):

        return f"""
Agent 1 Response

{agent1_response}

Agent 2 Response

{agent2_response}

The two agents disagree.

Reanalyse ONLY the disputed issue.

Do not repeat your previous analysis.

Focus only on resolving the disagreement.

Return your answer in exactly this format:

Stance:
ENDORSE
or
CHALLENGE

Reason:
Explain whether Agent 1's criticism changes your opinion.
"""
