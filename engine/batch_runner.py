import time
from copy import deepcopy

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
    new_tribunal_session,
    record_tribunal_round,
    followup_plan,
)
from engine.pre_hearing import build_pre_hearing_audit
from engine.runtime_profile import resolve_runtime_profile
from engine.reproducibility import seed_stage
from engine.stage_checkpoint import StageCheckpointStore


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
        hardware_profile="8gb",
        semantic_bridge_mode="disabled",
        stage_checkpoint_dir=None,
        resume_stages=False,
        global_seed=42,
        checkpoint_fingerprint="",
        candidate_mode="independent",
        readonly_stage_sources=(),
        control_mode="none",
        batch_size=32,
        tribunal_repair_mode="disabled",
        tribunal_audit_mode="baseline",
    ):
        if feedback_mode not in {"disabled", "collect", "calibrate", "verified", "precedent", "integrated"}:
            raise ValueError(f"Unknown feedback mode: {feedback_mode}")
        if debate_mode not in {"enabled", "disabled"}:
            raise ValueError(f"Unknown debate mode: {debate_mode}")
        if evidence_mode not in {"enabled", "disabled"}:
            raise ValueError(f"Unknown evidence mode: {evidence_mode}")
        if judge_mode not in JUDGE_MODES:
            raise ValueError(f"Unknown judge mode: {judge_mode}")
        if judge_scope not in JUDGE_SCOPES:
            raise ValueError(f"Unknown judge scope: {judge_scope}")
        if candidate_mode not in {"independent", "disabled"}:
            raise ValueError(f"Unknown candidate mode: {candidate_mode}")
        self.candidate_mode = candidate_mode
        if control_mode not in {"none", "independent_vote", "conventional_debate"}:
            raise ValueError("Unknown control mode")
        if control_mode != "none" and (debate_mode != "disabled" or judge_mode != "disabled"
                or semantic_bridge_mode != "disabled" or feedback_mode != "disabled" or candidate_mode != "independent"):
            raise ValueError("Simpler controls require independent candidates and disabled hearing/judge/bridge/feedback")
        self.control_mode = control_mode
        self.debate = DebateEngine()
        self.feedback_mode = feedback_mode
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.batch_size = batch_size
        self.debate_mode = debate_mode
        self.evidence_mode = evidence_mode
        self.judge_mode = judge_mode
        self.judge_scope = judge_scope
        self.hardware_profile = resolve_runtime_profile(hardware_profile)
        from engine.tribunal_repair import validate_options
        validate_options(tribunal_repair_mode, tribunal_audit_mode, judge_mode,
                         debate_mode, self.hardware_profile.tribunal_protocol, feedback_mode)
        self.tribunal_repair_mode = tribunal_repair_mode
        self.tribunal_audit_mode = tribunal_audit_mode
        if feedback_mode == 'integrated' and (judge_mode != 'tribunal' or debate_mode == 'disabled'):
            raise ValueError('Integrated repair requires tribunal mode and enabled hearings')
        if feedback_mode == 'integrated' and self.hardware_profile.tribunal_protocol != 'evidence-review-5.0':
            raise ValueError('Integrated repair requires the explicit paper-8gb-review5 candidate profile')
        if feedback_mode == 'precedent' and self.hardware_profile.tribunal_protocol == 'evidence-review-5.0':
            raise ValueError('V5 replaces standalone precedent retries. Use --feedback-mode integrated, or disabled for the ablation.')
        if semantic_bridge_mode not in {"disabled", "shadow", "corroborated"}:
            raise ValueError(f"Unknown semantic bridge mode: {semantic_bridge_mode}")
        self.semantic_bridge_mode = semantic_bridge_mode
        self.global_seed = int(global_seed)
        self.debate.hardware_profile = self.hardware_profile.name
        self.debate.global_seed = self.global_seed
        self.stage_checkpoints = StageCheckpointStore(
            stage_checkpoint_dir, enabled=resume_stages,
            fingerprint=checkpoint_fingerprint,
            readonly_sources=readonly_stage_sources,
        )
        self.run_timing = {}
        self.feedback_events = []
        self.tribunal_precedents = None
        if feedback_mode == "precedent":
            if judge_mode != "tribunal" or not verified_feedback_path:
                raise ValueError("Precedent feedback requires tribunal mode and a frozen library file")
            from engine.tribunal_feedback import TribunalPrecedents
            self.tribunal_precedents = TribunalPrecedents(verified_feedback_path)
        if feedback_log_path:
            self.debate.feedback_loop.log_file = feedback_log_path
        if feedback_mode == "verified":
            if not verified_feedback_path:
                raise ValueError("Verified feedback mode requires a feedback file.")
            self.debate.feedback_loop.load_verified_examples(verified_feedback_path)

    def _seed_sample(self, sample, stage, attempt=0):
        sample_id = sample.get("raw", {}).get("id", sample.get("index"))
        return seed_stage(
            getattr(self, "global_seed", 42), sample_id, stage, attempt
        )

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
                "sample_id": sample.get("raw", {}).get("id", sample["index"]),
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
                "tribunal_hearing": results[sample["index"]].get(
                    "tribunal_hearing", False
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
            previous_hearing = result.get("debate_details", {}) or {}
            history = deepcopy(previous_hearing.get("hearing_history", []))
            if previous_hearing:
                history.append({key: deepcopy(previous_hearing[key]) for key in (
                    "agent1_critique", "agent2_critique", "mediation", "review_status") if key in previous_hearing})
            result["debate_details"] = decision.get("_debate", {})
            result["debate_details"]["hearing_history"] = history
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
            result["timing"]["debate_seconds"] = result["timing"].get("debate_seconds", 0.0) + decision["_debate"].get(
                "inference_seconds", 0.0
            )

    def _apply_feedback(self, batch, results, context):
        if self.feedback_mode not in {"collect", "calibrate"}:
            for sample in batch:
                result = results[sample["index"]]
                base_context = result.get("_feedback_context", context)
                reliability_updates = []
                # Verified memory is frozen across ALL batches. Gold-based
                # reliability updates here would alter retrieval for later cases.
                # Corrections/harms are measured by the post-run evaluator instead.
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
                if self.feedback_mode == "precedent":
                    reviews = result.get("judge", {}).get("tribunal_reviews", [])
                    matched = sorted({p for r in reviews for p in r.get("_retrieved_precedent_ids", [])})
                    result["feedback"].update(feedback_target_agent="tribunal", role="reasoning_guidance_only",
                        matched_rule_ids=matched, memory_active=bool(matched),
                        library_sha256=self.tribunal_precedents.sha256,
                        verifier_receives_precedents=False, attribution="requires_paired_ablation")
                if getattr(self, 'feedback_mode', 'disabled') == 'integrated':
                    reviews = result.get('judge', {}).get('tribunal_reviews', [])
                    attempted = any(r.get('_integrated_feedback', {}).get('attempted') for r in reviews)
                    result['feedback'].update(mode='integrated', feedback_target_agent='tribunal',
                        role='one_diagnostic_repair_hearing', memory_active=False,
                        repair_attempted=attempted, attribution='requires_paired_ablation', online_update=False)
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
        """Run the single supervisor review after both agents have answered."""
        if not hasattr(self, "stage_checkpoints"):
            self.stage_checkpoints = StageCheckpointStore()
        load_seconds = 0.0
        if not samples:
            return load_seconds
        pending_samples = []
        for sample in samples:
            cached = self.stage_checkpoints.load(
                f"tribunal_round_{round_number}", sample
            )
            if cached is None:
                pending_samples.append(sample)
                continue
            results[sample["index"]] = cached
            print(
                f"[resume] restored tribunal round {round_number} for "
                f"{sample.get('raw', {}).get('id', sample['index'])}"
            )
        if not pending_samples:
            return load_seconds
        samples = pending_samples
        judge_runtime = None
        reviewer = None
        try:
            load_started = time.time()
            profile_name = getattr(
                getattr(self, "hardware_profile", None), "name", "8gb"
            )
            load_error = None
            try:
                judge_runtime = QwenJudgeModel(hardware_profile=profile_name)
                judge_runtime.tribunal_audit_mode = getattr(self, 'tribunal_audit_mode', 'baseline')
                if getattr(self, "_tribunal_cost_history", None):
                    judge_runtime._review_cost_samples = deepcopy(self._tribunal_cost_history)
                reviewer = TribunalMediatorAgent(judge_runtime)
            except (RuntimeError, ValueError, OSError) as error:
                load_error = error
            load_seconds = time.time() - load_started
            for sample in samples:
                result = results[sample["index"]]
                sample_id = sample.get("raw", {}).get("id", sample["index"])
                print(f"[sample={sample_id}][model=judge][round={round_number}] start")
                from engine.judge_telemetry import mark_phase
                mark_phase(f"tribunal_round_{round_number}:{sample_id}")
                self._seed_sample(sample, "tribunal_review", round_number)
                debate = result.get("debate_details", {}) or {}
                from engine.review_outcome import failed_review
                precedent_kwargs = {}
                prior_guidance = any(r.get("_precedent_feedback", {}).get("attempted")
                                     for r in result.get("judge", {}).get("tribunal_reviews", []))
                if getattr(self, "tribunal_precedents", None):
                    precedent_kwargs["precedents"] = ([] if prior_guidance else
                        self.tribunal_precedents.retrieve(result.get("language_output", {}), max_items=2))
                review = failed_review(load_error, "model_load", load_seconds / len(samples)) if load_error else reviewer.review(
                    sample["image"],
                    sample["caption"],
                    result.get("visual_output", {}),
                    result.get("language_output", {}),
                    result.get("comparison", {}),
                    result.get("evidence_ledger", []),
                    debate,
                    round_number=round_number,
                    current_decision=result.get("decision", {}),
                    pre_hearing=result.get("pre_hearing", {}),
                    _verification_repair=(result.get('judge', {}).get('verification_followup_plan') or {}).get('repair_context') if round_number == 2 else None,
                    prior_review=(result.get('judge', {}).get('tribunal_reviews') or [None])[-1] if round_number == 2 else None,
                    **precedent_kwargs,
                )
                if getattr(self, 'feedback_mode', 'disabled') == 'integrated':
                    review['_integrated_feedback'] = {'attempted': round_number == 2,
                        'policy': 'one_diagnostic_repair_hearing',
                        'repair_reasons': (result.get('judge', {}).get('verification_followup_plan') or {}).get('repair_reasons', []),
                        'execution_status': review.get('_execution_status'), 'format_valid': review.get('_format_valid')}
                review['_tribunal_repair'] = {
                    'mode': getattr(self, 'tribunal_repair_mode', 'disabled'),
                    'attempted': bool(round_number == 2 and (result.get('judge', {}).get('verification_followup_plan') or {}).get('repair_context')),
                    'feedback_mode': getattr(self, 'feedback_mode', 'disabled')}
                if precedent_kwargs:
                    if prior_guidance:
                        review["_precedent_feedback"] = {"attempted": False, "selected_guided_review": False,
                            "reason": "already_attempted_this_case", "policy": "one_guided_attempt_per_case"}
                    review["_precedent_library_sha256"] = self.tribunal_precedents.sha256
                    review["_retrieved_precedent_ids"] = [p["id"] for p in precedent_kwargs["precedents"]]
                    self.feedback_events.append({"sample_id": sample_id, "round": round_number,
                        "mode": "precedent", "matched_rule_ids": review["_retrieved_precedent_ids"],
                        "library_sha256": self.tribunal_precedents.sha256,
                        "feedback_audit": review.get("_precedent_feedback", {}), "online_update": False})
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
                        semantic_bridge_mode=getattr(
                            self, "semantic_bridge_mode", "disabled"
                        ),
                        language_output=result.get("language_output", {}),
                        source_caption=sample["caption"],
                    )
                )
                result["decision"] = resolved
                if getattr(self, 'feedback_mode', 'disabled') == 'integrated' and round_number == 1:
                    result['pre_feedback_decision'] = deepcopy(resolved)
                if getattr(self, 'feedback_mode', 'disabled') == 'integrated' and round_number == 2:
                    self.feedback_events.append({'sample_id': sample_id, 'mode': 'integrated',
                        'repair_attempted': True, 'accepted': bool(resolution.get('accepted')),
                        'causal_attribution': 'requires_paired_ablation'})
                result["evidence_ledger"] = verified_ledger
                judge["tribunal_resolution"] = resolution
                judge["execution_status"] = resolution.get("execution_status", "NOT_RUN")
                judge["case_dossier"] = {
                    "schema_version": review.get("_case_dossier_schema"),
                    "supervisor_checkpoint": "POST_TARGETED_HEARING",
                    "persistent_memory": "serialized_case_dossier",
                    "model_residency": "stage_local",
                }
                feedback_candidate = judge_feedback_candidate(
                    {
                        "_format_valid": review.get("_format_valid", False),
                        "verdict": review.get("provisional_verdict"),
                    },
                    resolution.get("previous_label"),
                )
                feedback_candidate.update({
                    "issue": review.get("issue", ""),
                    "evidence_ids": list(
                        review.get("_valid_evidence_ids", []) or []
                    ),
                    "tribunal_resolution_accepted": bool(
                        resolution.get("accepted", False)
                    ),
                    "review_reason": resolution.get("reason", ""),
                    "verification_required": True,
                })
                judge["feedback_candidate"] = feedback_candidate
                session = dict(judge.get("tribunal_session", {}) or {})
                prior_state = session.get("state")
                if resolution.get("execution_status") == "FAILED" or resolution.get('terminal_outcome') == 'EXECUTION_INTERRUPTED_OR_FAILED':
                    session["state"] = "EXECUTION_FAILED"
                elif resolution.get("context_status", "NOT_CHECKED") not in {"VALID", "NOT_CHECKED"}:
                    session["state"] = resolution["context_status"]
                elif resolution.get("schema_status") != "VALID":
                    session["state"] = "INVALID_OUTPUT"
                elif prior_state == "FOLLOW_UP_REQUIRED":
                    session["state"] = "FOLLOW_UP_REQUIRED"
                elif resolution.get("accepted"):
                    session["state"] = "RESOLVED"
                elif (
                    review.get("status") in {"ABSTAIN", "FOLLOW_UP"}
                    or prior_state == "ABSTAINED"
                ):
                    session["state"] = "ABSTAINED"
                else:
                    session["state"] = "PRESERVED"
                if review.get('_protocol') == 'evidence-review-5.0' and round_number < session.get('max_rounds', 2):
                    from engine.tribunal_repair import scheduled_repair
                    plan = scheduled_repair(review, debate, resolution,
                        mode=getattr(self, 'tribunal_repair_mode', 'disabled'),
                        feedback_mode=getattr(self, 'feedback_mode', 'disabled'),
                        round_number=round_number, max_rounds=session.get('max_rounds', 2))
                    if plan.get('_usable'):
                        judge['verification_followup_plan'] = plan
                        session['state'] = 'FOLLOW_UP_REQUIRED'
                    elif session['state'] == 'FOLLOW_UP_REQUIRED':
                        session['state'] = 'ABSTAINED' if review.get('relation') == 'UNRESOLVED' else 'PRESERVED'
                elif (session["state"] == "PRESERVED" and round_number < session.get("max_rounds", 2)
                        and not resolution.get("confirmation_valid") and self.debate_mode != "disabled"):
                    from engine.tribunal import repair_followup_plan
                    repair_plan = repair_followup_plan(review, result.get("comparison", {}), debate)
                    if repair_plan.get("_usable"):
                        judge["verification_followup_plan"] = repair_plan
                        session["state"] = "FOLLOW_UP_REQUIRED"
                session["stop_reason"] = (
                    session.get("stop_reason")
                    if prior_state == "ABSTAINED" and session.get("stop_reason")
                    else resolution.get("reason", "")
                )
                judge["tribunal_session"] = session
                judge["status"] = (
                    "tribunal_execution_failed"
                    if session.get("state") == "EXECUTION_FAILED"
                    else "tribunal_context_blocked"
                    if resolution.get("context_status", "NOT_CHECKED") not in {"VALID", "NOT_CHECKED"}
                    else "tribunal_invalid_output"
                    if session.get("state") == "INVALID_OUTPUT"
                    else "tribunal_revision_accepted"
                    if resolution.get("accepted")
                    else "tribunal_abstained"
                    if session.get("state") == "ABSTAINED"
                    else "tribunal_revision_rejected"
                )
                self.stage_checkpoints.save(
                    f"tribunal_round_{round_number}", sample, result
                )
        finally:
            if judge_runtime is not None:
                self._tribunal_cost_history = deepcopy(getattr(judge_runtime, "_review_cost_samples", {}))
            if reviewer is not None:
                del reviewer
            if judge_runtime is not None:
                del judge_runtime
            GPUManager.clear()
        return load_seconds

    def _run_tribunal_followups(self, tribunal_candidates, results):
        """Execute the single scheduled repair using the production witness/review path."""
        load_totals = {"judge_model_load_seconds": 0., "debate_vision_model_load_seconds": 0., "debate_language_model_load_seconds": 0.}
        # Follow-up is an actual new witness hearing, only on explicit
        # unresolved questions. No forced opposing answer is fabricated.
        followup_samples = []
        previous_evidence = {}
        from engine.deliberation import deliberation_signature
        for sample in tribunal_candidates:
            if self.debate_mode == "disabled":
                continue
            result = results[sample["index"]]
            judge = result.get("judge", {})
            if judge.get("tribunal_session", {}).get("state") != "FOLLOW_UP_REQUIRED":
                continue
            review = (judge.get("tribunal_reviews") or [{}])[-1]
            plan = judge.get("verification_followup_plan") or followup_plan(review, result.get("comparison", {}))
            if not plan.get("_usable"):
                judge["tribunal_session"].update(
                    state="PRESERVED", stop_reason=review.get("_follow_up_budget_status") or review.get("_follow_up_question_status") or "no_actionable_followup")
                continue
            previous_evidence[sample["index"]] = deliberation_signature(result)
            plan["hearing_round"] = 2
            result["mediation_plan"] = plan
            result["force_visual_review"] = bool(plan.get("agent1_questions"))
            result["tribunal_hearing"] = True
            followup_samples.append(sample)
        if followup_samples:
            from engine.judge_telemetry import mark_phase
            mark_phase("followup_witnesses")
            witness_samples = [sample for sample in followup_samples
                if any(results[sample["index"]]["mediation_plan"].get(k)
                       for k in ("agent1_questions", "agent2_questions"))]
            if witness_samples:
                followup_results = self.debate.run_debate_batch(
                    self._build_debate_cases(witness_samples, results))
                self._merge_debate_results(witness_samples, results, followup_results)
            else:
                self.debate.last_batch_timing = {}
            for sample in followup_samples:
                result = results[sample["index"]]
                questions = result["mediation_plan"].get("verification_requests", [])
                if questions:
                    from engine.review_routing import new_semantic_questions
                    previous_questions = result.get('debate_details', {}).get('tribunal_semantic_questions', {}).get('questions', [])
                    questions = new_semantic_questions(questions, previous_questions)
                    result.setdefault("debate_details", {})["tribunal_semantic_questions"] = {
                        "questions": questions or previous_questions, "role": "tribunal_only",
                        "new_resolving_check": bool(questions),
                        "is_new_visual_evidence": False}

            for key in ("vision_model_load_seconds", "language_model_load_seconds"):
                load_totals["debate_" + key] += self.debate.last_batch_timing.get(key, 0.0)
            ready = []
            for sample in followup_samples:
                result = results[sample["index"]]
                changed = deliberation_signature(result) != previous_evidence[sample["index"]]
                semantic_requested = bool(result.get("debate_details", {}).get("tribunal_semantic_questions", {}).get('new_resolving_check'))
                result["judge"]["tribunal_session"].setdefault("hearing_transitions", []).append({
                    "round": 2, "before": previous_evidence[sample["index"]],
                    "after": deliberation_signature(result), "reconsidered": changed or semantic_requested,
                    "repair_reasons": list((result["judge"].get("verification_followup_plan") or {}).get("repair_reasons", [])),
                    "agent2_response": result.get("debate_details", {}).get("agent2_critique", {})})
                if changed or semantic_requested:
                    ready.append(sample)
                else:
                    result["judge"]["tribunal_session"].update(
                        state="PRESERVED", stop_reason="no_new_admissible_evidence")
            load_totals["judge_model_load_seconds"] += self._run_tribunal_review_round(ready, results, 2)
        return load_totals

    def run_samples(self, samples, on_result):
        if getattr(self, "tribunal_precedents", None):
            import hashlib
            self.tribunal_precedents.assert_disjoint([
                dict(s["raw"], caption_sha256=hashlib.sha256(s["caption"].encode()).hexdigest()) for s in samples])
        from engine.runtime_accounting import begin_accounting, sample_accounting
        begin_accounting()
        run_started = time.time()
        load_totals = {
            "vision_model_load_seconds": 0.0,
            "language_model_load_seconds": 0.0,
            "debate_vision_model_load_seconds": 0.0,
            "debate_language_model_load_seconds": 0.0,
            "evidence_verifier_model_load_seconds": 0.0,
            "judge_model_load_seconds": 0.0,
        }
        run_counts = dict(judge_requested_samples=0, judge_accepted_revisions=0,
                          mediated_tiebreak_revisions=0, feedback_matched_samples=0)
        # Calibrated memory is collected only after inference. Verified memory
        # is immutable and matched per case, so results do not depend on sample order.
        from engine.bounded_batches import decoded_batches
        batches = decoded_batches(samples, getattr(self, 'batch_size', 32))
        for batch_index, batch in enumerate(batches, start=1):
            context = self._feedback_context(batch_index)
            visual_outputs = {}
            print(f"\nSTAGE 1: visual grounding (batch {batch_index})")
            visual_pending = []
            for sample in batch:
                cached = self.stage_checkpoints.load("visual_grounding", sample)
                if cached is None:
                    visual_pending.append(sample)
                    continue
                visual_outputs[sample["index"]] = cached
                print(
                    "[resume] restored visual grounding for "
                    f"{sample.get('raw', {}).get('id', sample['index'])}"
                )
            vision_runtime = None
            agent1 = None
            try:
                if visual_pending:
                    load_started = time.time()
                    profile_name = getattr(
                        getattr(self, "hardware_profile", None), "name", "8gb"
                    )
                    vision_runtime = Qwen3VLVisionModel(
                        hardware_profile=profile_name
                    )
                    load_totals["vision_model_load_seconds"] += time.time() - load_started
                    agent1 = VisualGroundingAgent(vision_runtime)
                for sample in visual_pending:
                    sample_id = sample.get("raw", {}).get("id", sample["index"])
                    width, height = getattr(sample["image"], "size", (None, None))
                    print(
                        f"[sample={sample_id}][model=agent1] start "
                        f"image={width}x{height} profile={self.hardware_profile.name}"
                    )
                    self._seed_sample(sample, "visual_grounding")
                    visual_outputs[sample["index"]] = agent1.analyze(sample["image"], feedback=None)
                    self.stage_checkpoints.save(
                        "visual_grounding", sample,
                        visual_outputs[sample["index"]],
                    )
                    # Preserve decoded evidence in system RAM while releasing
                    # only temporary CUDA allocations before the next image.
                    vision_runtime.release_generation_memory()
                if (self.judge_mode == "tribunal" or self.control_mode != "none") and self.candidate_mode == "independent":
                    from engine.candidate_cases import candidate_case
                    for sample in batch:
                        candidate = self.stage_checkpoints.load("visual_candidate", sample)
                        if candidate is None:
                            if vision_runtime is None:
                                started = time.time()
                                vision_runtime = Qwen3VLVisionModel(hardware_profile=self.hardware_profile.name)
                                load_totals["vision_model_load_seconds"] += time.time() - started
                            self._seed_sample(sample, "visual_candidate")
                            candidate = candidate_case(vision_runtime, sample["image"], sample["caption"],
                                                       model_family="Qwen3-VL")
                            self.stage_checkpoints.save("visual_candidate", sample, candidate)
                        visual_outputs[sample["index"]]["_independent_candidate"] = candidate
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
                    cached = self.stage_checkpoints.load("initial_reasoning", sample)
                    if cached is not None:
                        results[sample["index"]] = cached
                        if (
                            cached.get("debate_level", 0) > 0
                            and self.debate_mode == "enabled"
                        ):
                            debate_candidates.append(sample)
                        print(
                            "[resume] restored initial reasoning for "
                            f"{sample.get('raw', {}).get('id', sample['index'])}"
                        )
                        continue
                    visual_output = visual_outputs[sample["index"]]
                    sample_id = sample.get("raw", {}).get("id", sample["index"])
                    print(f"[sample={sample_id}][model=agent2] start")
                    self._seed_sample(sample, "initial_reasoning")
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
                    result["reproducibility"] = {
                        "global_seed": self.global_seed,
                        "sample_seed": seed_stage(
                            self.global_seed, sample_id, "sample_identity"
                        ),
                    }
                    result["runtime_profile"] = self.hardware_profile.as_dict()
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
                    result["pre_hearing"] = build_pre_hearing_audit(
                        decision, comparison, debate_assessment
                    )
                    if debate_assessment.get("trigger") and self.debate_mode == "enabled":
                        debate_candidates.append(sample)
                    baseline_cache = deepcopy(result)
                    baseline_cache["visual_output"].pop("_independent_candidate", None)
                    self.stage_checkpoints.save("initial_reasoning", sample, baseline_cache)

                if (self.judge_mode == "tribunal" or self.control_mode != "none") and self.candidate_mode == "independent":
                    from engine.candidate_cases import candidate_case, TextCandidateRuntime, candidate_dispute
                    for sample in batch:
                        result = results[sample["index"]]
                        result["visual_output"]["_independent_candidate"] = visual_outputs[sample["index"]].get("_independent_candidate")
                        candidate = self.stage_checkpoints.load("text_candidate", sample)
                        if candidate is None:
                            self._seed_sample(sample, "text_candidate")
                            observations = [item.get("text", "") for item in result["evidence_ledger"]
                                            if item.get("source") == "agent1" and item.get("grounded")]
                            candidate = candidate_case(
                                TextCandidateRuntime(mistral.model, mistral.tokenizer), None,
                                sample["caption"], observations, model_family="Mistral")
                            self.stage_checkpoints.save("text_candidate", sample, candidate)
                        result["language_output"]["_independent_candidate"] = candidate
                        dispute = candidate_dispute(result["visual_output"], result["language_output"])
                        result["candidate_dispute"] = dispute
                        if dispute["decisive_questions"] and (dispute["genuine_disagreement"] or dispute["unresolved"]):
                            result.setdefault("pre_hearing", {})["requires_live_hearing"] = True
                        result["timing"]["candidate_seconds"] = sum(
                            float((output.get("_independent_candidate") or {}).get("_generation_seconds", 0.0))
                            for output in (result["visual_output"], result["language_output"]))

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

            if self.control_mode != "none":
                from engine.control_conditions import run_control_batch
                control_loads = run_control_batch(batch, results, self.control_mode,
                    self.hardware_profile.name, self.global_seed)
                for key, value in control_loads.items():
                    load_totals[key] += value
            if self.judge_mode in {"mediated", "tribunal"}:
                tribunal_candidates = []
                if self.judge_mode == "tribunal":
                    debate_candidates = [
                        sample for sample in debate_candidates
                        if results[sample["index"]].get(
                            "pre_hearing", {}
                        ).get("requires_live_hearing", False)
                    ]
                candidate_indexes = {
                    sample["index"] for sample in debate_candidates
                }
                for sample in batch:
                    result = results[sample["index"]]
                    reasons = judge_request_reasons(result, self.judge_scope)
                    requested = bool(reasons)
                    if self.judge_mode == "tribunal" and requested:
                        tribunal_candidates.append(sample)
                    # Tribunal eligibility is independent from the ordinary
                    # debate router.  Invalid claim contracts and ungrounded
                    # decisions must still receive agent questions rather
                    # than silently skipping the tribunal.
                    if (
                        requested
                        and sample["index"] not in candidate_indexes
                        and self.debate_mode == "enabled"
                        and (
                            self.judge_mode != "tribunal"
                            or result.get("pre_hearing", {}).get(
                                "requires_live_hearing", False
                            )
                        )
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
                    result["tribunal_hearing"] = bool(
                        self.judge_mode == "tribunal"
                        and sample["index"] in candidate_indexes
                    )
                    results[sample["index"]]["judge"] = {
                        "mode": self.judge_mode,
                        "scope": self.judge_scope,
                        "requested": requested,
                        "trigger_reasons": reasons,
                        "status": (
                            "pending" if requested else "not_escalated"
                        ),
                    }
                if debate_candidates:
                    print(
                        f"\nSTAGE 2C: label-blind hearing plan "
                        f"({len(debate_candidates)} cases)"
                    )
                    if self.judge_mode == "tribunal":
                        for sample in debate_candidates:
                            result = results[sample["index"]]
                            plan = result.get("pre_hearing", {}).get(
                                "question_plan", {}
                            )
                            mediation = {
                                "status": "MEDIATE",
                                "provisional_verdict": "ABSTAIN",
                                "confidence": 0.5,
                                "disputed_issues": [plan.get("issue", "")],
                                "agent1_questions": [
                                    plan.get("agent1_question", "")
                                ],
                                "agent2_questions": [
                                    plan.get("agent2_question", "")
                                ],
                                "verification_requests": [
                                    plan.get("verification_request", "")
                                ],
                                "reason": "deterministic_pre_hearing_plan",
                                "_valid_evidence_ids": [],
                                "_invalid_evidence_ids": [],
                                "_format_valid": True,
                                "_usable": bool(plan.get("agent1_question")),
                                "_generation_seconds": 0.0,
                            }
                            from engine.candidate_cases import candidate_hearing_plan
                            mediation = candidate_hearing_plan(mediation, result.get("candidate_dispute", {}))
                            mediation["hearing_round"] = 1
                            result["mediation_plan"] = mediation
                            result["judge"]["mediation"] = mediation
                            result["judge"]["status"] = "hearing_planned"
                            result["force_visual_review"] = True
                    else:
                        judge_runtime = None
                        mediator_agent = None
                        try:
                            load_started = time.time()
                            judge_runtime = QwenJudgeModel(
                                hardware_profile=self.hardware_profile.name
                            )
                            load_totals["judge_model_load_seconds"] += (
                                time.time() - load_started
                            )
                            mediator_agent = MultimodalMediatorAgent(judge_runtime)
                            for sample in debate_candidates:
                                result = results[sample["index"]]
                                self._seed_sample(sample, "mediation_plan")
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
                                    mediation.get("_usable", False)
                                    and mediation.get("agent1_questions")
                                ):
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

            if self.judge_mode == "tribunal" and tribunal_candidates:
                print(
                    f"\nSTAGE 4: checkpointed supervisor review "
                    f"({len(tribunal_candidates)} cases)"
                )
                tribunal_load = self._run_tribunal_review_round(
                    tribunal_candidates, results, 1
                )
                load_totals["judge_model_load_seconds"] += tribunal_load

                for key, value in self._run_tribunal_followups(tribunal_candidates, results).items():
                    load_totals[key] += value

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
                        judge_runtime = QwenJudgeModel(
                            hardware_profile=self.hardware_profile.name
                        )
                        load_totals["judge_model_load_seconds"] += (
                            time.time() - load_started
                        )
                        judge_agent = MultimodalJudgeAgent(judge_runtime)
                        for sample in judge_candidates:
                            result = results[sample["index"]]
                            self._seed_sample(sample, "appellate_judge")
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
                from engine.final_artifact import final_artifact
                result["final_artifact"] = final_artifact(sample["caption"], result)
                result["runtime_accounting"] = sample_accounting(sample.get("raw", {}).get("id", sample["index"]))
                self.stage_checkpoints.save('final_result', sample, result)
                on_result(
                    sample["index"], sample["raw"], result,
                    result["timing"]["sample_inference_seconds"],
                )
                judge = result.get('judge', {})
                review_key = 'tribunal_resolution' if self.judge_mode == 'tribunal' else 'mediation_review' if self.judge_mode == 'mediated' else 'appellate_review'
                run_counts['judge_requested_samples'] += bool(judge.get('requested'))
                run_counts['judge_accepted_revisions'] += bool(judge.get(review_key, {}).get('accepted'))
                run_counts['mediated_tiebreak_revisions'] += str(judge.get('mediation_review', {}).get('reason', '')).startswith('accepted_mediated_verified_tiebreak:')
                run_counts['feedback_matched_samples'] += bool(result.get('feedback', {}).get('matched_rule_ids'))

        self.run_timing = {
            **{key: round(value, 4) for key, value in load_totals.items()},
            "feedback_mode": self.feedback_mode,
            "debate_mode": self.debate_mode,
            "evidence_mode": self.evidence_mode,
            "judge_mode": self.judge_mode,
            "judge_scope": self.judge_scope,
            "feedback_updates": sum(
                event.get("update_applied", False)
                for event in self.feedback_events
            ),
            "feedback_events": len(self.feedback_events),
            "feedback_candidates": sum(
                event.get("candidate_recorded", False)
                for event in self.feedback_events
            ),
            "wall_clock_seconds": round(time.time() - run_started, 4),
        }
        self.run_timing["stage_reuse_events"] = deepcopy(self.stage_checkpoints.reuse_events)
        self.run_timing.update(run_counts, batch_size=getattr(self, 'batch_size', 32))
        return self.run_timing

    def export_feedback_examples(self):
        return self.debate.feedback_loop.export_examples()
