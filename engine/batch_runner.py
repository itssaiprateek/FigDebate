import time

from agents.visual_grounding import VisualGroundingAgent
from agents.claim_extraction import ClaimExtractionAgent
from agents.multimodal_judge import (
    MultimodalJudgeAgent,
    MultimodalMediatorAgent,
    TribunalMediatorAgent,
)
from arbiter.arbiter import Arbiter
from comparators.evidence_comparator import compare
from engine.debate import DebateEngine
from engine.evidence_ledger import attach_evidence_audit, build_evidence_ledger
from engine.evidence_verifier import AtomicEvidenceVerifier, merge_verified_evidence
from engine.gpu_manager import GPUManager
from engine.relation_schema import attach_claim_relation
from engine.review_board import attach_final_review
from engine.decision_trace import (
    append_decision_checkpoint,
    attach_decision_trace,
)
from engine.judge_review import (
    JUDGE_MODES,
    JUDGE_SCOPES,
    apply_judge_review,
    judge_feedback_candidate,
    judge_request_reasons,
)
from models.judge_model import QwenJudgeModel
from models.language_model import MistralModel
from models.vision_model import Qwen3VLVisionModel
from engine.tribunal import (
    apply_tribunal_resolution,
    followup_plan,
    new_tribunal_session,
    record_tribunal_round,
)


class StagewiseRunner:
    """Run FigDebate with optional verified feedback collection or application."""

    def __init__(
        self,
        feedback_mode="disabled",
        feedback_log_path=None,
        verified_feedback_path=None,
        debate_mode="enabled",
        evidence_mode="enabled",
        judge_mode="disabled",
        judge_scope="escalated",
    ):
        if feedback_mode not in {"disabled", "collect", "calibrate", "verified"}:
            raise ValueError(f"Unknown feedback mode: {feedback_mode}")
        if debate_mode not in {"enabled", "disabled"}:
            raise ValueError(f"Unknown debate mode: {debate_mode}")
        if evidence_mode not in {"enabled", "disabled"}:
            raise ValueError(f"Unknown evidence mode: {evidence_mode}")
        if judge_mode not in JUDGE_MODES:
            raise ValueError(f"Unknown judge mode: {judge_mode}")
        if judge_scope not in JUDGE_SCOPES:
            raise ValueError(f"Unknown judge scope: {judge_scope}")
        self.debate = DebateEngine()
        self.feedback_mode = feedback_mode
        self.debate_mode = debate_mode
        self.evidence_mode = evidence_mode
        self.judge_mode = judge_mode
        self.judge_scope = judge_scope
        self.run_timing = {}
        self.feedback_events = []
        if feedback_log_path:
            self.debate.feedback_loop.log_file = feedback_log_path
        if feedback_mode == "verified":
            if not verified_feedback_path:
                raise ValueError("Verified feedback mode requires a feedback file.")
            self.debate.feedback_loop.load_verified_examples(verified_feedback_path)

    @staticmethod
    def _agent1_feedback_instruction():
        return """Re-analyze the image using only directly observable visual evidence.
Avoid unsupported symbolic interpretations and do not invent objects."""

    @staticmethod
    def _agent2_feedback_instruction():
        return """Re-analyze the caption without imagining the image. Distinguish
humor, sarcasm, metaphor, and a literal caption; the dataset phenomenon may be visual."""

    @staticmethod
    def _arbiter_feedback_instruction():
        return """Make a binary decision from grounded visual evidence and the caption meaning.
Do not treat missing support as contradiction or broad thematic similarity as proof."""

    def _feedback_context(self, batch_index):
        loop = self.debate.feedback_loop
        agent1_size = len(loop.agent1_memory)
        agent2_size = len(loop.agent2_memory)
        arbiter_size = len(loop.arbiter_memory)
        return {
            "mode": self.feedback_mode,
            "enabled": self.feedback_mode != "disabled",
            "batch_index": batch_index,
            "memory_active": False,
            "agent1_memory_size_before": agent1_size,
            "agent2_memory_size_before": agent2_size,
            "arbiter_memory_size_before": arbiter_size,
            "available_rule_count": agent1_size + agent2_size + arbiter_size,
            "matched_rule_ids": [],
        }

    @staticmethod
    def _base_result(
        visual_output,
        language_output,
        comparison,
        decision,
        evidence_ledger,
        evidence_verification=None,
    ):
        initial_decision = attach_evidence_audit(
            decision.get("_primary_decision", decision), evidence_ledger
        )
        arbiter_timing = decision.get("_timing", {})
        timing = {
            "agent1_seconds": visual_output.get("_generation_seconds", 0.0),
            "agent2_seconds": language_output.get("_generation_seconds", 0.0),
            "comparator_seconds": 0.0,
            "evidence_verifier_seconds": (
                (evidence_verification or {}).get("seconds", 0.0)
            ),
            "feedback_review_seconds": 0.0,
            "arbiter_primary_seconds": arbiter_timing.get("primary_seconds", 0.0),
            "citation_retry_seconds": arbiter_timing.get(
                "citation_retry_seconds", 0.0
            ),
            "format_retry_seconds": arbiter_timing.get("format_retry_seconds", 0.0),
            "binary_resolution_seconds": arbiter_timing.get("binary_resolution_seconds", 0.0),
            "debate_seconds": 0.0,
            "judge_seconds": 0.0,
            "mediator_seconds": 0.0,
        }
        return {
            "visual_output": visual_output,
            "language_output": language_output,
            "comparison": comparison,
            "evidence_ledger": evidence_ledger,
            "evidence_verification": evidence_verification or {},
            "initial_decision": initial_decision,
            "decision": decision,
            "debate_triggered": False,
            "debate_rounds": 0,
            "debate_details": {},
            "round1_confidence": initial_decision.get("confidence"),
            "round2_confidence": None,
            "timing": timing,
        }

    @staticmethod
    def _finish_timing(result):
        result["timing"]["sample_inference_seconds"] = sum(
            value
            for key, value in result["timing"].items()
            if key != "sample_inference_seconds"
        )

    @staticmethod
    def _build_debate_cases(samples, results):
        return [
            {
                "key": sample["index"],
                "image": sample["image"],
                "caption": sample["caption"],
                "visual_output": results[sample["index"]]["visual_output"],
                "language_output": results[sample["index"]]["language_output"],
                "comparison": results[sample["index"]]["comparison"],
                "decision": results[sample["index"]]["decision"],
                "evidence_ledger": results[sample["index"]]["evidence_ledger"],
                "debate_level": results[sample["index"]].get("debate_level", 1),
                "debate_score": results[sample["index"]].get(
                    "debate_need_score", 0
                ),
                "debate_signals": results[sample["index"]].get(
                    "debate_need_signals", []
                ),
                "mediation_plan": results[sample["index"]].get(
                    "mediation_plan", {}
                ),
                "force_visual_review": results[sample["index"]].get(
                    "force_visual_review", False
                ),
            }
            for sample in samples
        ]

    @staticmethod
    def _merge_debate_results(samples, results, debate_results):
        for sample in samples:
            result = results[sample["index"]]
            decision = debate_results[sample["index"]]
            result["decision"] = decision
            result["evidence_ledger"] = decision.get(
                "_evidence_ledger", result["evidence_ledger"]
            )
            result["debate_triggered"] = True
            result["debate_rounds"] = decision["_debate"]["rounds"]
            result["debate_details"] = decision.get("_debate", {})
            recovered_visual = result["debate_details"].get(
                "recovered_visual_output"
            )
            recovered_comparison = result["debate_details"].get(
                "recovered_comparison"
            )
            if recovered_visual:
                result["visual_output"] = recovered_visual
            if recovered_comparison:
                result["comparison"] = recovered_comparison
            recovered_verification = result["debate_details"].get(
                "recovered_evidence_verification"
            )
            if recovered_verification:
                result["evidence_verification"] = recovered_verification
            result["round2_confidence"] = decision.get("confidence")
            result["timing"]["debate_seconds"] = decision["_debate"].get(
                "inference_seconds", 0.0
            )

    def _apply_feedback(self, batch, results, context):
        if self.feedback_mode not in {"collect", "calibrate"}:
            for sample in batch:
                result = results[sample["index"]]
                base_context = result.get("_feedback_context", context)
                reliability_updates = []
                if self.feedback_mode == "verified" and base_context.get(
                    "matched_rule_ids"
                ):
                    reliability_updates = (
                        self.debate.feedback_loop.record_reliability_outcome(
                            base_context.get("matched_rule_ids", []),
                            result.get("pre_feedback_decision", {}).get("label"),
                            result.get("decision", {}).get("label"),
                            sample["raw"].get("label"),
                        )
                    )
                results[sample["index"]]["feedback"] = {
                    **base_context,
                    "update_applied": False,
                    "candidate_recorded": False,
                    "failure_type": None,
                    "feedback_target_agent": (
                        "arbiter" if base_context.get("matched_rule_ids") else None
                    ),
                    "agent1_memory_size_after": base_context.get(
                        "agent1_memory_size_before", 0
                    ),
                    "agent2_memory_size_after": base_context.get(
                        "agent2_memory_size_before", 0
                    ),
                    "arbiter_memory_size_after": base_context.get(
                        "arbiter_memory_size_before", 0
                    ),
                    "role": "procedural_review_routing_only",
                    "reliability_updates": reliability_updates,
                    "feedback_baseline_label": result.get(
                        "pre_feedback_decision", {}
                    ).get("label"),
                    "feedback_post_review_label": result.get(
                        "decision", {}
                    ).get("label"),
                    "feedback_decision_changed": result.get(
                        "pre_feedback_decision", {}
                    ).get("label") != result.get("decision", {}).get("label"),
                }
                if self.feedback_mode == "verified":
                    self.feedback_events.append({
                        "sample_id": sample["raw"]["id"],
                        "sample_index": sample["index"],
                        "batch_index": context["batch_index"],
                        **results[sample["index"]]["feedback"],
                    })
            return

        for sample in batch:
            result = results[sample["index"]]
            base_context = result.get("_feedback_context", context)
            metadata = {
                "sample_id": sample["raw"]["id"],
                "sample_index": sample["index"],
                "batch_index": context["batch_index"],
            }
            event = self.debate.generate_feedback(
                result["visual_output"],
                result["language_output"],
                result["comparison"],
                result["decision"],
                ground_truth=sample["raw"]["label"],
                phenomenon=sample["raw"].get("phenomenon"),
                metadata=metadata,
                apply_calibration=self.feedback_mode == "calibrate",
                evidence_ledger=result.get("evidence_ledger", []),
            )
            feedback_record = {
                **base_context,
                "update_applied": event["update_applied"],
                "candidate_recorded": event["candidate_recorded"],
                "failure_type": event["failure_type"],
                "feedback_target_agent": event.get("target_agent"),
                "agent1_memory_size_after": event["agent1_memory_size"],
                "agent2_memory_size_after": event["agent2_memory_size"],
                "arbiter_memory_size_after": event["arbiter_memory_size"],
            }
            result["feedback"] = feedback_record
            self.feedback_events.append({**metadata, **feedback_record})

    def _run_tribunal_review_round(
        self, samples, results, round_number
    ):
        """Run one mediator review after both agents have answered."""
        followups = []
        load_seconds = 0.0
        if not samples:
            return followups, load_seconds
        judge_runtime = None
        reviewer = None
        try:
            load_started = time.time()
            judge_runtime = QwenJudgeModel()
            load_seconds = time.time() - load_started
            reviewer = TribunalMediatorAgent(judge_runtime)
            for sample in samples:
                result = results[sample["index"]]
                debate = result.get("debate_details", {}) or {}
                review = reviewer.review(
                    sample["image"],
                    sample["caption"],
                    result.get("visual_output", {}),
                    result.get("language_output", {}),
                    result.get("comparison", {}),
                    result.get("evidence_ledger", []),
                    debate,
                    round_number=round_number,
                )
                result["timing"]["mediator_seconds"] = round(
                    float(result["timing"].get("mediator_seconds", 0.0))
                    + float(review.get("_generation_seconds", 0.0)),
                    4,
                )
                judge = result.setdefault("judge", {})
                session = judge.get("tribunal_session") or new_tribunal_session(
                    judge.get("mediation", {})
                )
                session = record_tribunal_round(session, review, debate)
                judge["tribunal_session"] = session
                judge.setdefault("tribunal_reviews", []).append(review)

                if session.get("state") == "FOLLOW_UP_REQUIRED":
                    plan = followup_plan(review)
                    if plan.get("_usable", False):
                        result["mediation_plan"] = plan
                        result["debate_level"] = 2
                        result["debate_need_signals"] = [
                            "tribunal_follow_up"
                        ]
                        followups.append(sample)
                        judge["status"] = "tribunal_follow_up_planned"
                        continue

                resolved, verified_ledger, resolution = (
                    apply_tribunal_resolution(
                        result.get("decision", {}),
                        review,
                        result.get("evidence_ledger", []),
                        result.get("language_output", {}).get(
                            "claim_contract", {}
                        ),
                        agent2_requirements_valid=debate.get(
                            "agent2_requirements_valid", True
                        ),
                        agent1_critique=debate.get("agent1_critique", {}),
                        agent2_critique=debate.get("agent2_critique", {}),
                    )
                )
                result["decision"] = resolved
                result["evidence_ledger"] = verified_ledger
                judge["tribunal_resolution"] = resolution
                session = dict(judge.get("tribunal_session", {}) or {})
                prior_state = session.get("state")
                if resolution.get("accepted"):
                    session["state"] = "RESOLVED"
                elif review.get("status") == "ABSTAIN" or prior_state == "ABSTAINED":
                    session["state"] = "ABSTAINED"
                else:
                    session["state"] = "PRESERVED"
                session["stop_reason"] = (
                    session.get("stop_reason")
                    if prior_state == "ABSTAINED" and session.get("stop_reason")
                    else resolution.get("reason", "")
                )
                judge["tribunal_session"] = session
                judge["status"] = (
                    "tribunal_revision_accepted"
                    if resolution.get("accepted")
                    else "tribunal_abstained"
                    if session.get("state") == "ABSTAINED"
                    else "tribunal_revision_rejected"
                )
        finally:
            if reviewer is not None:
                del reviewer
            if judge_runtime is not None:
                del judge_runtime
            GPUManager.clear()
        return followups, load_seconds

    def run_samples(self, samples, on_result):
        run_started = time.time()
        load_totals = {
            "vision_model_load_seconds": 0.0,
            "language_model_load_seconds": 0.0,
            "debate_vision_model_load_seconds": 0.0,
            "debate_language_model_load_seconds": 0.0,
            "evidence_verifier_model_load_seconds": 0.0,
            "judge_model_load_seconds": 0.0,
        }
        # Calibrated memory is collected only after inference. Verified memory
        # is immutable and matched per case, so results do not depend on sample order.
        batches = [samples]
        for batch_index, batch in enumerate(batches, start=1):
            context = self._feedback_context(batch_index)
            visual_outputs = {}
            print(f"\nSTAGE 1: visual grounding (batch {batch_index})")
            vision_runtime = None
            agent1 = None
            try:
                load_started = time.time()
                vision_runtime = Qwen3VLVisionModel()
                load_totals["vision_model_load_seconds"] += time.time() - load_started
                agent1 = VisualGroundingAgent(vision_runtime)
                for sample in batch:
                    visual_outputs[sample["index"]] = agent1.analyze(
                        sample["image"], feedback=None
                    )
                    # Preserve decoded evidence in system RAM while releasing
                    # only temporary CUDA allocations before the next image.
                    vision_runtime.release_generation_memory()
            finally:
                if agent1 is not None:
                    del agent1
                if vision_runtime is not None:
                    del vision_runtime
                GPUManager.clear()

            results = {}
            debate_candidates = []
            print(f"\nSTAGE 2: language, comparison, and initial decision (batch {batch_index})")
            mistral = None
            agent2 = None
            arbiter = None
            evidence_verifier = None
            try:
                load_started = time.time()
                mistral = MistralModel()
                load_totals["language_model_load_seconds"] += time.time() - load_started
                agent2 = ClaimExtractionAgent(mistral.model, mistral.tokenizer)
                if self.evidence_mode == "enabled":
                    verifier_load_started = time.time()
                    evidence_verifier = AtomicEvidenceVerifier()
                    load_totals["evidence_verifier_model_load_seconds"] += (
                        time.time() - verifier_load_started
                    )
                    self.debate.nli_verifier = evidence_verifier.nli
                arbiter = Arbiter(
                    mistral.model,
                    mistral.tokenizer,
                    nli_verifier=(
                        evidence_verifier.nli if evidence_verifier else None
                    ),
                )
                for sample in batch:
                    visual_output = visual_outputs[sample["index"]]
                    language_output = agent2.analyze(
                        sample["caption"], feedback=None
                    )
                    language_output = attach_claim_relation(
                        language_output, sample["caption"]
                    )
                    comparison_started = time.time()
                    comparison = compare(
                        visual_output, language_output, caption=sample["caption"]
                    )
                    comparison_seconds = time.time() - comparison_started
                    evidence_ledger = build_evidence_ledger(
                        visual_output, language_output, comparison
                    )
                    if evidence_verifier:
                        evidence_ledger, evidence_verification = evidence_verifier.verify(
                            evidence_ledger,
                            language_output,
                            comparison,
                        )
                        comparison = merge_verified_evidence(
                            comparison,
                            evidence_ledger,
                            evidence_verification,
                        )
                    else:
                        evidence_verification = {
                            "mode": "disabled_ablation",
                            "candidate_count": 0,
                            "verified_count": 0,
                            "support_count": 0,
                            "conflict_count": 0,
                            "neutral_count": 0,
                            "seconds": 0.0,
                        }
                    baseline_decision = arbiter.analyze(
                        sample["caption"], visual_output, language_output, comparison,
                        feedback=None,
                    )
                    baseline_decision = attach_evidence_audit(
                        baseline_decision, evidence_ledger
                    )
                    baseline_decision = attach_final_review(
                        baseline_decision,
                        evidence_ledger,
                        language_output.get("claim_contract", {}),
                    )
                    baseline_trace = append_decision_checkpoint(
                        [], "initial_arbiter", baseline_decision,
                        ledger=evidence_ledger,
                    )
                    baseline_decision = attach_decision_trace(
                        baseline_decision, baseline_trace
                    )
                    decision = baseline_decision
                    feedback_candidate = None
                    feedback_review_seconds = 0.0
                    feedback_revision_accepted = False
                    feedback_revision_reason = "feedback_not_requested"
                    matched_rules = []
                    matched_rule_ids = []
                    matched_rule_scores = {}
                    feedback_case = {
                        "visual_output": visual_output,
                        "language_output": language_output,
                        "comparison": comparison,
                        "decision": baseline_decision,
                        "evidence_ledger": evidence_ledger,
                    }
                    if self.feedback_mode == "verified":
                        loop = self.debate.feedback_loop
                        matched_rules = loop.matching_rules("arbiter", feedback_case)
                        matched_rule_ids = loop.matching_rule_ids(
                            "arbiter", feedback_case
                        )
                        matched_rule_scores = loop.matching_rule_scores(
                            "arbiter", feedback_case
                        )
                        if matched_rules:
                            comparison = dict(comparison)
                            comparison["feedback_warning"] = {
                                "memory_ids": matched_rule_ids,
                                "failure_patterns": sorted({
                                    item.get(
                                        "failure_mechanism",
                                        item.get("failure_type", "reviewed_error"),
                                    )
                                    for item in matched_rules
                                }),
                                "diagnostic_questions": [
                                    item.get("diagnostic_question", "")
                                    for item in matched_rules
                                    if item.get("diagnostic_question")
                                ],
                                "repair_actions": [
                                    item.get("repair_action", item.get("example", ""))
                                    for item in matched_rules
                                    if item.get("repair_action", item.get("example"))
                                ],
                                "role": "procedural_review_routing_only",
                            }
                            feedback_revision_reason = (
                                "procedural_memory_routed_to_review"
                            )
                        else:
                            feedback_revision_reason = (
                                "no_matching_procedural_memory"
                            )
                    result = self._base_result(
                        visual_output,
                        language_output,
                        comparison,
                        decision,
                        evidence_ledger,
                        evidence_verification,
                    )
                    result["pre_feedback_decision"] = baseline_decision
                    result["feedback_candidate_decision"] = feedback_candidate
                    baseline_timing = baseline_decision.get("_timing", {}) or {}
                    result["timing"]["arbiter_primary_seconds"] = baseline_timing.get(
                        "primary_seconds", 0.0
                    )
                    result["timing"]["format_retry_seconds"] = baseline_timing.get(
                        "format_retry_seconds", 0.0
                    )
                    result["timing"]["citation_retry_seconds"] = (
                        baseline_timing.get("citation_retry_seconds", 0.0)
                    )
                    result["timing"]["binary_resolution_seconds"] = (
                        baseline_timing.get("binary_resolution_seconds", 0.0)
                    )
                    result["_feedback_context"] = {
                        **context,
                        "memory_active": bool(matched_rule_ids),
                        "matched_rule_ids": matched_rule_ids,
                        "matched_rule_scores": matched_rule_scores,
                        "feedback_revision_accepted": feedback_revision_accepted,
                        "feedback_revision_reason": feedback_revision_reason,
                        "feedback_candidate_label": (
                            feedback_candidate.get("label")
                            if feedback_candidate else None
                        ),
                        "feedback_baseline_label": baseline_decision.get("label"),
                        "feedback_post_review_label": decision.get("label"),
                    }
                    result["timing"]["comparator_seconds"] = comparison_seconds
                    result["timing"]["feedback_review_seconds"] = (
                        feedback_review_seconds
                    )
                    results[sample["index"]] = result
                    debate_assessment = self.debate.debate_assessment(
                        decision, comparison
                    )
                    trigger_reason = debate_assessment.get("reason")
                    result["debate_trigger_reason"] = trigger_reason
                    result["debate_level"] = debate_assessment.get("level", 0)
                    result["debate_need_score"] = debate_assessment.get("score", 0)
                    result["debate_need_signals"] = debate_assessment.get("signals", [])
                    if debate_assessment.get("trigger") and self.debate_mode == "enabled":
                        debate_candidates.append(sample)

                level1_candidates = [
                    sample for sample in debate_candidates
                    if results[sample["index"]].get("debate_level", 1) == 1
                ]
                if level1_candidates and self.judge_mode not in {
                    "mediated", "tribunal"
                }:
                    print(
                        f"\nSTAGE 2B: Level 1 debate with loaded Mistral "
                        f"({len(level1_candidates)} cases)"
                    )
                    level1_results = self.debate.run_debate_batch(
                        self._build_debate_cases(level1_candidates, results),
                        language_runtime={"agent2": agent2, "arbiter": arbiter},
                    )
                    self._merge_debate_results(
                        level1_candidates, results, level1_results
                    )
                    debate_candidates = [
                        sample for sample in debate_candidates
                        if sample not in level1_candidates
                    ]
            finally:
                if agent2 is not None:
                    del agent2
                if arbiter is not None:
                    del arbiter
                if mistral is not None:
                    del mistral
                GPUManager.clear()

            if self.judge_mode in {"mediated", "tribunal"}:
                candidate_indexes = {
                    sample["index"] for sample in debate_candidates
                }
                for sample in batch:
                    result = results[sample["index"]]
                    reasons = judge_request_reasons(result, self.judge_scope)
                    requested = bool(reasons)
                    # Tribunal eligibility is independent from the ordinary
                    # debate router.  Invalid claim contracts and ungrounded
                    # decisions must still receive agent questions rather
                    # than silently skipping the tribunal.
                    if (
                        requested
                        and sample["index"] not in candidate_indexes
                        and self.debate_mode == "enabled"
                    ):
                        debate_candidates.append(sample)
                        candidate_indexes.add(sample["index"])
                        result["debate_level"] = 2
                        result["debate_need_signals"] = list(dict.fromkeys(
                            list(result.get("debate_need_signals", []) or [])
                            + ["tribunal_escalation"]
                        ))
                        result["debate_trigger_reason"] = (
                            result.get("debate_trigger_reason")
                            or "tribunal_requires_agent_responses"
                        )
                    results[sample["index"]]["judge"] = {
                        "mode": self.judge_mode,
                        "scope": self.judge_scope,
                        "requested": requested,
                        "trigger_reasons": reasons,
                        "status": "pending" if requested else "not_escalated",
                    }
                if debate_candidates:
                    print(
                        f"\nSTAGE 2C: label-blind debate mediation "
                        f"({len(debate_candidates)} cases)"
                    )
                    judge_runtime = None
                    mediator_agent = None
                    try:
                        load_started = time.time()
                        judge_runtime = QwenJudgeModel()
                        load_totals["judge_model_load_seconds"] += (
                            time.time() - load_started
                        )
                        mediator_agent = MultimodalMediatorAgent(judge_runtime)
                        for sample in debate_candidates:
                            result = results[sample["index"]]
                            mediation = mediator_agent.analyze(
                                sample["image"],
                                sample["caption"],
                                result["visual_output"],
                                result["language_output"],
                                result["comparison"],
                                result["evidence_ledger"],
                            )
                            result["mediation_plan"] = mediation
                            result["timing"]["mediator_seconds"] = mediation.get(
                                "_generation_seconds", 0.0
                            )
                            result["judge"]["mediation"] = mediation
                            if (
                                self.judge_mode == "tribunal"
                                or (
                                    mediation.get("_usable", False)
                                    and mediation.get("agent1_questions")
                                )
                            ):
                                # A tribunal visual question must be answered
                                # from the current image even when ordinary
                                # debate routing selected lightweight Level 1.
                                result["force_visual_review"] = True
                            if not mediation.get("_format_valid", False):
                                result["judge"]["status"] = "invalid_mediation_contract"
                            elif mediation.get("_invalid_evidence_ids"):
                                result["judge"]["status"] = (
                                    "invalid_mediation_evidence_ids"
                                )
                            elif mediation.get("status") == "ABSTAIN":
                                result["judge"]["status"] = "mediation_abstained"
                            else:
                                result["judge"]["status"] = "mediation_planned"
                    finally:
                        if mediator_agent is not None:
                            del mediator_agent
                        if judge_runtime is not None:
                            del judge_runtime
                        GPUManager.clear()

            if debate_candidates:
                stage_name = (
                    "tribunal-guided debate revisions"
                    if self.judge_mode == "tribunal"
                    else "mediated debate revisions"
                    if self.judge_mode == "mediated"
                    else "Level 2 debate revisions"
                )
                print(f"\nSTAGE 3: {stage_name} (batch {batch_index})")
                debate_cases = self._build_debate_cases(
                    debate_candidates, results
                )
                debate_results = self.debate.run_debate_batch(debate_cases)
                load_totals["debate_vision_model_load_seconds"] += self.debate.last_batch_timing.get(
                    "vision_model_load_seconds", 0.0
                )
                load_totals["debate_language_model_load_seconds"] += self.debate.last_batch_timing.get(
                    "language_model_load_seconds", 0.0
                )
                self._merge_debate_results(
                    debate_candidates, results, debate_results
                )
                if self.judge_mode in {"mediated", "tribunal"}:
                    for sample in debate_candidates:
                        result = results[sample["index"]]
                        debate = result.get("debate_details", {}) or {}
                        accepted = bool(debate.get("revision_accepted", False))
                        result["judge"]["mediation_review"] = {
                            "accepted": accepted,
                            "changed_decision": accepted and (
                                result.get("initial_decision", {}).get("label")
                                != result.get("decision", {}).get("label")
                            ),
                            "previous_label": result.get(
                                "initial_decision", {}
                            ).get("label"),
                            "proposed_label": debate.get("proposed_label"),
                            "reason": debate.get("revision_acceptance_reason", ""),
                        }
                        if result["judge"].get("status") == "mediation_planned":
                            result["judge"]["status"] = (
                                "mediated_revision_accepted"
                                if accepted else "mediated_revision_rejected"
                            )

            if self.judge_mode == "tribunal" and debate_candidates:
                print(
                    f"\nSTAGE 4: tribunal reviews both agent responses "
                    f"({len(debate_candidates)} cases)"
                )
                followups, tribunal_load = self._run_tribunal_review_round(
                    debate_candidates, results, 1
                )
                load_totals["judge_model_load_seconds"] += tribunal_load
                if followups:
                    print(
                        f"\nSTAGE 5: bounded tribunal follow-up "
                        f"({len(followups)} cases)"
                    )
                    prior_debate_seconds = {
                        sample["index"]: results[sample["index"]]["timing"].get(
                            "debate_seconds", 0.0
                        )
                        for sample in followups
                    }
                    followup_results = self.debate.run_debate_batch(
                        self._build_debate_cases(followups, results)
                    )
                    load_totals["debate_vision_model_load_seconds"] += (
                        self.debate.last_batch_timing.get(
                            "vision_model_load_seconds", 0.0
                        )
                    )
                    load_totals["debate_language_model_load_seconds"] += (
                        self.debate.last_batch_timing.get(
                            "language_model_load_seconds", 0.0
                        )
                    )
                    self._merge_debate_results(
                        followups, results, followup_results
                    )
                    for sample in followups:
                        result = results[sample["index"]]
                        result["timing"]["debate_seconds"] = round(
                            prior_debate_seconds[sample["index"]]
                            + result["timing"].get("debate_seconds", 0.0),
                            4,
                        )
                    print(
                        f"\nSTAGE 6: final tribunal review "
                        f"({len(followups)} cases)"
                    )
                    _, final_load = self._run_tribunal_review_round(
                        followups, results, 2
                    )
                    load_totals["judge_model_load_seconds"] += final_load

            if self.judge_mode in {"shadow", "appellate"}:
                judge_candidates = []
                for sample in batch:
                    result = results[sample["index"]]
                    reasons = judge_request_reasons(result, self.judge_scope)
                    result["judge"] = {
                        "mode": self.judge_mode,
                        "scope": self.judge_scope,
                        "requested": bool(reasons),
                        "trigger_reasons": reasons,
                        "status": "pending" if reasons else "not_escalated",
                    }
                    if reasons:
                        judge_candidates.append(sample)

                if judge_candidates:
                    print(
                        f"\nSTAGE 4: independent multimodal judge "
                        f"({len(judge_candidates)} cases, {self.judge_mode} mode)"
                    )
                    judge_runtime = None
                    judge_agent = None
                    try:
                        load_started = time.time()
                        judge_runtime = QwenJudgeModel()
                        load_totals["judge_model_load_seconds"] += (
                            time.time() - load_started
                        )
                        judge_agent = MultimodalJudgeAgent(judge_runtime)
                        for sample in judge_candidates:
                            result = results[sample["index"]]
                            judgment = judge_agent.analyze(
                                sample["image"],
                                sample["caption"],
                                result["visual_output"],
                                result["language_output"],
                                result["comparison"],
                                result["evidence_ledger"],
                                result.get("debate_details", {}),
                            )
                            reviewed_decision, appellate_review = apply_judge_review(
                                result["decision"],
                                judgment,
                                result["evidence_ledger"],
                                result["language_output"].get(
                                    "claim_contract", {}
                                ),
                                mode=self.judge_mode,
                            )
                            result["decision"] = reviewed_decision
                            result["timing"]["judge_seconds"] = judgment.get(
                                "_generation_seconds", 0.0
                            )
                            if not judgment.get("_format_valid", False):
                                judge_status = "invalid_contract"
                            elif judgment.get("verdict") == "ABSTAIN":
                                judge_status = "abstained"
                            elif appellate_review.get("accepted"):
                                judge_status = "appellate_revision_accepted"
                            elif self.judge_mode == "shadow":
                                judge_status = "shadow_completed"
                            else:
                                judge_status = "appellate_revision_rejected"
                            result["judge"].update({
                                "status": judge_status,
                                "judgment": judgment,
                                "appellate_review": appellate_review,
                                "feedback_candidate": judge_feedback_candidate(
                                    judgment,
                                    appellate_review.get("previous_label"),
                                ),
                            })
                    finally:
                        if judge_agent is not None:
                            del judge_agent
                        if judge_runtime is not None:
                            del judge_runtime
                        GPUManager.clear()

            self._apply_feedback(batch, results, context)
            for sample in batch:
                result = results[sample["index"]]
                self._finish_timing(result)
                on_result(
                    sample["index"], sample["raw"], result,
                    result["timing"]["sample_inference_seconds"],
                )

        self.run_timing = {
            **{key: round(value, 4) for key, value in load_totals.items()},
            "feedback_mode": self.feedback_mode,
            "debate_mode": self.debate_mode,
            "evidence_mode": self.evidence_mode,
            "judge_mode": self.judge_mode,
            "judge_scope": self.judge_scope,
            "judge_requested_samples": sum(
                bool(result.get("judge", {}).get("requested"))
                for result in results.values()
            ),
            "judge_accepted_revisions": sum(
                bool(
                    (
                        result.get("judge", {}).get(
                            "tribunal_resolution", {}
                        )
                        if self.judge_mode == "tribunal"
                        else result.get("judge", {}).get(
                            "mediation_review", {}
                        )
                        if self.judge_mode == "mediated"
                        else result.get("judge", {}).get("appellate_review", {})
                    ).get("accepted")
                )
                for result in results.values()
            ),
            "mediated_tiebreak_revisions": sum(
                str(
                    result.get("judge", {})
                    .get("mediation_review", {})
                    .get("reason", "")
                ).startswith("accepted_mediated_verified_tiebreak:")
                for result in results.values()
            ),
            "feedback_updates": sum(
                event.get("update_applied", False)
                for event in self.feedback_events
            ),
            "feedback_events": len(self.feedback_events),
            "feedback_candidates": sum(
                event.get("candidate_recorded", False)
                for event in self.feedback_events
            ),
            "feedback_matched_samples": sum(
                bool(result.get("feedback", {}).get("matched_rule_ids"))
                for result in results.values()
            ),
            "wall_clock_seconds": round(time.time() - run_started, 4),
        }
        return self.run_timing

    def export_feedback_examples(self):
        return self.debate.feedback_loop.export_examples()
