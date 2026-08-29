import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from importlib import metadata

from dataset.loaders import decode_image, load_split
from engine.batch_runner import StagewiseRunner
from engine.evidence_verifier import AtomicEvidenceVerifier
from engine.feedback_loop import FeedbackLoop
from engine.run_integrity import validate_resume_config
from engine.sampling import select_records
from evaluation.evaluate_predictions import evaluate_predictions
from figdebate import FigDebate
from models.judge_model import JUDGE_MODEL_ID, JUDGE_MODEL_REVISION
from models.vision_model import VISION_MODEL_ID, VISION_MODEL_REVISION


LANGUAGE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
LANGUAGE_MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
EVIDENCE_LEDGER_VERSION = "11.0"


FIELDNAMES = [
    "sample", "id", "dataset_source", "phenomenon", "ground_truth",
    "reference_explanation", "prediction",
    "initial_prediction", "final_prediction", "primary_prediction",
    "primary_confidence", "primary_decision_valid", "decision_method",
    "format_retry_used", "citation_retry_used",
    "binary_resolution_used", "binary_resolution_entails_score",
    "binary_resolution_contradicts_score", "binary_resolution_raw_confidence", "round1_confidence",
    "semantic_entails_score", "semantic_contradicts_score",
    "semantic_neutral_score", "semantic_verifier_model",
    "semantic_verifier_revision", "semantic_raw_confidence",
    "semantic_evidence_quality",
    "round2_confidence", "final_confidence", "debate_triggered",
    "debate_rounds", "decision_changed_after_debate",
    "debate_revision_accepted", "debate_revision_reason",
    "debate_failed_invariant", "debate_acceptance_checks",
    "debate_proposed_label",
    "debate_unconstrained_proposed_label", "debate_proposed_decision_method",
    "debate_relation_status", "debate_deficiencies",
    "debate_review_status", "debate_trigger_reason",
    "debate_level", "debate_need_score", "debate_need_signals",
    "agent1_critique_stance", "agent1_critique_recommendation",
    "agent1_critique_reason",
    "agent1_critique_method", "agent1_critique_format_valid",
    "agent1_critique_format_retry_used",
    "agent1_critique_format_retry_success",
    "agent1_critique_region_ocr_candidates",
    "agent1_critique_observed_entity", "agent1_critique_observed_state",
    "agent1_critique_image_region", "agent1_critique_claim_relation",
    "agent1_critique_response_status", "agent1_critique_question_id",
    "agent1_critique_parser_errors", "agent1_generation_diagnostics",
    "agent1_region_pairs",
    "agent2_critique_stance", "agent2_critique_reason",
    "agent2_critique_format_valid", "agent2_support_requirement",
    "agent2_conflict_requirement", "agent2_figurative_mechanism",
    "agent2_critique_ambiguity", "agent2_requirements_source",
    "agent2_requirements_valid", "agent2_requirement_errors",
    "visual_evidence_consensus_applied",
    "initial_forced_label", "forced_label", "retry_attempted",
    "retry_failed", "final_decision_valid", "figurative_type_predicted",
    "figurative_type_was_guessed", "figurative_type_source",
    "figurative_type_retry_attempted", "figurative_type_retry_failed", "comparator_score",
    "figurative_type_resolution_confidence", "figurative_type_resolution_scores",
    "claim_retry_attempted", "claim_retry_success", "claim_retry_seconds",
    "comparator_recommendation", "final_reason", "arbiter_visual_support",
    "arbiter_contradictions", "arbiter_missing_evidence",
    "comparator_evidence_status", "comparator_shared_terms",
    "comparator_pre_verification_status",
    "comparator_evidence_quality",
    "comparator_claim_direction", "comparator_relation_binding_required",
    "comparator_relation_binding_observed", "comparator_has_symbolic_evidence",
    "comparator_has_symbolic_object_candidate", "comparator_has_text_surface",
    "comparator_text_surface_without_ocr",
    "comparator_direct_support_count", "comparator_direct_conflict_count",
    "comparator_grounded_anchor_count", "agent1_visible_text",
    "agent1_visible_text_count", "agent1_visual_fact_count",
    "agent1_visual_relation_count", "agent1_symbolic_tone",
    "agent1_visual_metaphor_count", "agent1_schema_complete",
    "agent1_schema_format_valid", "agent1_factual_grounding_present",
    "agent1_ocr_usable", "agent1_relation_binding_present",
    "agent1_schema_issues", "agent1_targeted_recovery_attempted",
    "agent1_targeted_recovery_success", "agent1_targeted_recovery_reason",
    "agent1_targeted_recovery_seconds",
    "agent1_schema_retry_attempted", "agent1_schema_retry_success",
    "agent1_visual_confidence", "agent2_linguistic_cue",
    "agent2_polarity_reversal", "agent2_language_confidence",
    "agent2_caption_proposition", "arbiter_evidence_assessment",
    "claim_relation_family", "claim_relation_polarity",
    "claim_relation_predicate", "claim_relation_resolved",
    "claim_contract_valid", "claim_contract_proposition_preserved",
    "claim_contract_entity_frame_preserved", "claim_contract_warnings",
    "claim_relation_pair_valid", "claim_reasoning_requirement",
    "claim_safe_for_automatic_directional_reasoning",
    "structured_relation_candidate_count",
    "review_board_status", "review_board_binary_valid",
    "review_board_directionally_grounded", "review_board_source_grounded",
    "review_board_confidence_cap_applied",
    "review_board_confidence_before", "review_board_confidence_after",
    "judge_mode", "judge_scope", "judge_requested", "judge_status",
    "pre_judge_prediction",
    "judge_trigger_reasons", "judge_verdict", "judge_confidence",
    "judge_format_valid", "judge_format_error", "judge_evidence_ids",
    "judge_invalid_evidence_ids", "judge_visual_observations", "judge_reason",
    "judge_agreed_with_pre_judge_decision", "judge_revision_accepted",
    "judge_revision_reason", "judge_changed_decision", "judge_model",
    "judge_model_revision", "judge_feedback_candidate_recorded",
    "judge_feedback_role", "judge_feedback_memory_update_applied",
    "mediator_status", "mediator_provisional_verdict", "mediator_confidence",
    "mediator_format_valid", "mediator_format_error", "mediator_evidence_ids",
    "mediator_invalid_evidence_ids", "mediator_disputed_issues",
    "mediator_agent1_questions", "mediator_agent2_questions",
    "mediator_verification_requests", "mediator_reason", "mediator_usable",
    "mediator_tiebreak_used",
    "tribunal_state", "tribunal_round_count", "tribunal_stop_reason",
    "tribunal_review_status", "tribunal_revision_accepted",
    "tribunal_revision_reason", "tribunal_corroboration_reason",
    "tribunal_acceptance_checks", "tribunal_verified_evidence_id",
    "agent1_seconds", "agent2_seconds", "comparator_seconds",
    "arbiter_primary_seconds", "citation_retry_seconds", "format_retry_seconds", "binary_resolution_seconds", "debate_seconds", "judge_seconds", "mediator_seconds",
    "feedback_mode", "feedback_enabled", "feedback_batch", "feedback_memory_active",
    "feedback_update_applied",
    "feedback_candidate_recorded",
    "feedback_failure_type", "feedback_agent1_memory_before",
    "feedback_agent2_memory_before", "feedback_agent1_memory_after",
    "feedback_agent2_memory_after", "feedback_arbiter_memory_before",
    "feedback_arbiter_memory_after", "feedback_target_agent",
    "feedback_matched_rule_ids", "feedback_matched_rule_count",
    "feedback_available_rule_count",
    "feedback_matched_rule_scores", "feedback_baseline_prediction",
    "feedback_candidate_prediction", "feedback_post_review_prediction",
    "feedback_revision_accepted", "feedback_revision_reason",
    "feedback_decision_changed", "feedback_role",
    "feedback_reliability_updates",
    "evidence_ledger_count", "evidence_support_count", "evidence_conflict_count",
    "evidence_anchor_count", "debate_visual_evidence_count",
    "debate_visual_evidence_ids", "debate_visual_witness_count",
    "debate_visual_witness_ids", "evidence_level_summary",
    "evidence_ledger_json",
    "initial_evidence_status", "initial_evidence_valid", "initial_cited_evidence_ids",
    "initial_source_evidence_valid", "initial_source_cited_evidence_ids",
    "final_evidence_status", "final_evidence_valid", "final_cited_evidence_ids",
    "evidence_lifecycle_summary", "arbiter_relation_status",
    "arbiter_revision_status", "arbiter_deficiencies",
    "final_source_evidence_valid", "final_source_cited_evidence_ids",
    "debate_proposed_evidence_status", "debate_proposed_evidence_valid",
    "debate_proposed_cited_evidence_ids",
    "debate_proposed_source_evidence_valid",
    "debate_proposed_source_cited_evidence_ids",
    "evidence_verifier_candidate_count", "evidence_verifier_verified_count",
    "evidence_verifier_support_count", "evidence_verifier_conflict_count",
    "evidence_verifier_neutral_count", "evidence_verifier_model",
    "evidence_verifier_revision", "evidence_verifier_seconds",
    "targeted_region_verification_method",
    "targeted_region_verification_decision_grade",
    "targeted_region_verification_reason",
    "decision_trace_json",
    "feedback_review_seconds",
    "runtime_seconds", "correct",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the unified FigDebate pipeline on a named dataset split."
    )
    parser.add_argument(
        "--dataset-split",
        choices=("vflute_train_dev50", "vflute_val", "vflute_test"),
        default="vflute_train_dev50",
        help=(
            "Dataset split to run. vflute_train_dev50 is for development only; "
            "use vflute_val for tuning and keep vflute_test untouched for final reporting."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-strategy",
        choices=("stratified", "random", "prefix"),
        default="stratified",
        help=(
            "Subset policy. Stratified balances phenomenon/label groups; "
            "prefix is retained only for reproducing historical runs."
        ),
    )
    parser.add_argument("--run-dir", help="Experiment directory; required with --resume.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=("stagewise", "sequential"),
        default="stagewise",
        help="stagewise reuses each model across samples; sequential is a smoke-test fallback.",
    )
    parser.add_argument(
        "--feedback-mode",
        choices=("disabled", "collect", "calibrate", "verified"),
        default="disabled",
        help=(
            "collect logs error candidates; calibrate builds gold-label rules from a development split; "
            "verified applies an immutable feedback file before a held-out run."
        ),
    )
    parser.add_argument(
        "--verified-feedback-file",
        help="JSON file of human-reviewed prompt examples; required in verified mode.",
    )
    parser.add_argument(
        "--debate-mode",
        choices=("enabled", "disabled"),
        default="enabled",
        help="Keep enabled for FigDebate; disabled is only for a controlled ablation run.",
    )
    parser.add_argument(
        "--evidence-mode",
        choices=("enabled", "disabled"),
        default="enabled",
        help="Keep enabled for FigDebate; disabled is only for evidence ablation.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=("disabled", "shadow", "appellate", "mediated", "tribunal"),
        default="disabled",
        help=(
            "disabled preserves the established pipeline; shadow logs an independent "
            "Qwen verdict without changing predictions; appellate allows only revisions "
            "accepted by the deterministic evidence gate; mediated preserves the "
            "one-pass planner; tribunal adds bounded post-response review and one "
            "optional targeted follow-up round."
        ),
    )
    parser.add_argument(
        "--judge-scope",
        choices=("escalated", "all"),
        default="escalated",
        help=(
            "For shadow/appellate, use auditable uncertainty signals or every "
            "sample. Mediated and tribunal modes also escalate qualifying "
            "cases into the validated agent-response path when the ordinary "
            "debate router did not select them."
        ),
    )
    return parser.parse_args()


def set_reproducibility(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def dataset_selection_checksum(records):
    payload = [
        {
            "id": item.get("id"),
            "label": item.get("label"),
            "phenomenon": item.get("phenomenon"),
            "caption": item.get("caption"),
        }
        for item in records
    ]
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_checksum(path):
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline_source_checksum():
    project_root = os.path.dirname(os.path.abspath(__file__))
    paths = (
        "agents/visual_grounding.py", "agents/visual_adapter.py",
        "agents/claim_extraction.py",
        "arbiter/arbiter.py", "comparators/evidence_comparator.py",
        "engine/batch_runner.py", "engine/orchestrator.py", "figdebate.py",
        "engine/debate.py", "engine/evidence_ledger.py",
        "engine/evidence_verifier.py", "models/nli_model.py",
        "engine/claim_contract.py", "engine/relation_schema.py",
        "engine/region_verifier.py", "engine/review_board.py",
        "engine/judge_review.py", "agents/multimodal_judge.py",
        "engine/decision_trace.py", "engine/question_router.py",
        "engine/tribunal.py",
        "engine/sampling.py", "engine/run_integrity.py",
        "engine/feedback_loop.py", "evaluation/build_feedback_memory.py",
        "evaluation/audit_evidence_provenance.py",
        "evaluation/evaluate_predictions.py", "evaluation/metrics_core.py",
        "models/vision_model.py", "models/language_model.py",
        "models/prepare_vision_model.py",
        "models/judge_model.py", "models/hub_source.py", "utils/judge_parser.py",
        "utils/visual_parser.py", "utils/claim_parser.py",
        "utils/arbiter_parser.py", "utils/decision_scoring.py",
        "run_figdebate.py",
    )
    digest = hashlib.sha256()
    for relative_path in paths:
        digest.update(relative_path.encode("utf-8"))
        with open(os.path.join(project_root, relative_path), "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def runtime_environment():
    packages = {}
    for package in (
        "torch", "transformers", "bitsandbytes", "datasets", "pandas",
        "scikit-learn", "Pillow",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    environment = {"packages": packages}
    try:
        import torch

        environment.update({
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        })
    except ImportError:
        environment.update({
            "torch_cuda_version": None,
            "cuda_available": False,
            "gpu_name": None,
        })
    return environment


def resolve_run_dir(args):
    if args.resume and not args.run_dir:
        raise ValueError("--resume requires an explicit --run-dir.")
    if args.run_dir:
        return os.path.abspath(args.run_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.abspath(os.path.join("outputs", f"run_{stamp}"))


def load_records(path):
    records = {}
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("id"):
                records[record["id"]] = record
    return records


def text_list(value):
    return " | ".join(str(item) for item in (value or []))


def build_record(index, raw, result, elapsed):
    decision = result["decision"]
    initial = result.get("initial_decision", decision)
    primary = initial
    language = result.get("language_output", {})
    comparison = result.get("comparison", {})
    visual = result.get("visual_output", {})
    debate = result.get("debate_details", {}) or {}
    agent1_critique = debate.get("agent1_critique", {}) or {}
    agent2_critique = debate.get("agent2_critique", {}) or {}
    prediction = decision.get("label")
    valid = bool(decision.get("_final_decision_valid", False))
    timing = result.get("timing", {}) or {}
    feedback = result.get("feedback", {}) or {}
    resolution_scores = decision.get("_binary_resolution_scores", {}) or {}
    ledger = result.get("evidence_ledger", decision.get("_evidence_ledger", [])) or []
    initial_audit = initial.get("_evidence_audit", {}) or {}
    final_audit = decision.get("_evidence_audit", {}) or {}
    proposed_audit = debate.get("proposed_evidence_audit", {}) or {}
    debate_acceptance_audit = debate.get(
        "revision_acceptance_audit", {}
    ) or {}
    matched_rule_ids = feedback.get("matched_rule_ids", []) or []
    evidence_verification = result.get("evidence_verification", {}) or {}
    pre_feedback = result.get("pre_feedback_decision", initial) or initial
    feedback_candidate = result.get("feedback_candidate_decision") or {}
    claim_contract = language.get("claim_contract", {}) or {}
    review_board = decision.get("_review_board", {}) or {}
    targeted_verification = decision.get(
        "_targeted_region_verification", {}
    ) or {}
    judge = result.get("judge", {}) or {}
    judgment = judge.get("judgment", {}) or {}
    tribunal_reviews = judge.get("tribunal_reviews", []) or []
    tribunal_review = tribunal_reviews[-1] if tribunal_reviews else {}
    tribunal_session = judge.get("tribunal_session", {}) or {}
    tribunal_resolution = judge.get("tribunal_resolution", {}) or {}
    mediation = judge.get("mediation", result.get("mediation_plan", {})) or {}
    judge_review = (
        judge.get("appellate_review", {})
        or tribunal_resolution
        or judge.get("mediation_review", {})
        or {}
    )
    judge_feedback = judge.get("feedback_candidate", {}) or {}
    debate_visual_evidence = [
        item for item in ledger
        if item.get("source") == "debate_visual_reinspection"
        and item.get("decision_grade", False)
    ]
    debate_visual_witnesses = [
        item for item in ledger
        if item.get("source") == "debate_visual_witness"
    ]
    evidence_level_summary = {}
    for item in ledger:
        level = item.get("evidence_level", "LEGACY_UNTYPED")
        evidence_level_summary[level] = evidence_level_summary.get(level, 0) + 1
    return {
        "sample": index, "id": raw["id"],
        "dataset_source": raw.get("source"),
        "phenomenon": raw.get("phenomenon"),
        "ground_truth": raw["label"], "prediction": prediction,
        "reference_explanation": raw.get("explanation", ""),
        "initial_prediction": initial.get("label"), "final_prediction": prediction,
        "primary_prediction": primary.get("label"),
        "primary_confidence": primary.get("confidence"),
        "primary_decision_valid": primary.get("_final_decision_valid", False),
        "decision_method": decision.get("decision_method", "primary"),
        "format_retry_used": decision.get("_format_retry_used", False),
        "citation_retry_used": decision.get("_citation_retry_used", False),
        "binary_resolution_used": decision.get("_binary_resolution_used", False),
        "binary_resolution_entails_score": resolution_scores.get("ENTAILS"),
        "binary_resolution_contradicts_score": resolution_scores.get("CONTRADICTS"),
        "binary_resolution_raw_confidence": decision.get("_binary_resolution_raw_confidence"),
        "semantic_entails_score": resolution_scores.get("ENTAILS"),
        "semantic_contradicts_score": resolution_scores.get("CONTRADICTS"),
        "semantic_neutral_score": resolution_scores.get("NEUTRAL"),
        "semantic_verifier_model": resolution_scores.get("nli_model", ""),
        "semantic_verifier_revision": resolution_scores.get("nli_revision", ""),
        "semantic_raw_confidence": decision.get("_binary_resolution_raw_confidence"),
        "semantic_evidence_quality": decision.get("_evidence_quality"),
        "round1_confidence": result.get("round1_confidence"),
        "round2_confidence": result.get("round2_confidence"),
        "final_confidence": decision.get("confidence"),
        "debate_triggered": result.get("debate_triggered", False),
        "debate_rounds": result.get("debate_rounds", 0),
        "decision_changed_after_debate": bool(result.get("debate_triggered"))
        and initial.get("label") != prediction,
        "debate_revision_accepted": debate.get("revision_accepted"),
        "debate_revision_reason": debate.get("revision_acceptance_reason", ""),
        "debate_failed_invariant": debate_acceptance_audit.get(
            "failed_invariant", ""
        ),
        "debate_acceptance_checks": json.dumps(
            debate_acceptance_audit.get("acceptance_checks", []),
            ensure_ascii=True,
        ),
        "debate_proposed_label": debate.get("proposed_label", ""),
        "debate_unconstrained_proposed_label": debate.get(
            "unconstrained_proposed_label", ""
        ),
        "debate_proposed_decision_method": (
            debate.get("proposed_decision", {}) or {}
        ).get("decision_method", ""),
        "debate_review_status": debate.get("review_status", ""),
        "debate_relation_status": debate.get("relation_status", ""),
        "debate_deficiencies": text_list(debate.get("deficiencies", [])),
        "debate_trigger_reason": result.get("debate_trigger_reason", ""),
        "debate_level": debate.get("level", result.get("debate_level", 0)),
        "debate_need_score": debate.get(
            "need_score", result.get("debate_need_score", 0)
        ),
        "debate_need_signals": text_list(
            debate.get("need_signals", result.get("debate_need_signals", []))
        ),
        "agent1_critique_stance": agent1_critique.get("stance", ""),
        "agent1_critique_recommendation": agent1_critique.get(
            "recommendation", ""
        ),
        "agent1_critique_reason": agent1_critique.get("reason", ""),
        "agent1_critique_method": agent1_critique.get("review_method", ""),
        "agent1_critique_format_valid": agent1_critique.get(
            "_format_valid", bool(agent1_critique.get("reason"))
        ),
        "agent1_critique_format_retry_used": agent1_critique.get(
            "_format_retry_used", False
        ),
        "agent1_critique_format_retry_success": agent1_critique.get(
            "_format_retry_success", False
        ),
        "agent1_critique_region_ocr_candidates": text_list(
            agent1_critique.get("_region_ocr_candidates", [])
        ),
        "agent1_critique_observed_entity": agent1_critique.get(
            "observed_entity", ""
        ),
        "agent1_critique_observed_state": agent1_critique.get(
            "observed_state", ""
        ),
        "agent1_critique_image_region": agent1_critique.get(
            "image_region", ""
        ),
        "agent1_critique_claim_relation": agent1_critique.get(
            "claim_relation", ""
        ),
        "agent1_critique_response_status": agent1_critique.get(
            "response_status", ""
        ),
        "agent1_critique_question_id": agent1_critique.get("question_id", ""),
        "agent1_critique_parser_errors": text_list(
            agent1_critique.get("parser_errors", [])
        ),
        "agent1_generation_diagnostics": json.dumps(
            agent1_critique.get("generation_diagnostics", {}), sort_keys=True
        ),
        "agent1_region_pairs": json.dumps(
            agent1_critique.get("region_pairs", []), ensure_ascii=True
        ),
        "agent2_critique_stance": agent2_critique.get("stance", ""),
        "agent2_critique_reason": agent2_critique.get("reason", ""),
        "agent2_critique_format_valid": agent2_critique.get(
            "_format_valid", False
        ),
        "agent2_support_requirement": agent2_critique.get(
            "support_requirement", ""
        ),
        "agent2_conflict_requirement": agent2_critique.get(
            "conflict_requirement", ""
        ),
        "agent2_figurative_mechanism": agent2_critique.get(
            "figurative_mechanism", ""
        ),
        "agent2_critique_ambiguity": agent2_critique.get(
            "ambiguity", ""
        ),
        "agent2_requirements_source": agent2_critique.get(
            "requirements_source", ""
        ),
        "agent2_requirements_valid": agent2_critique.get(
            "requirements_valid", True
        ),
        "agent2_requirement_errors": text_list(
            agent2_critique.get("requirement_errors", [])
        ),
        "visual_evidence_consensus_applied": (
            debate.get("proposed_decision", {}) or {}
        ).get("_visual_evidence_consensus_applied", False),
        "initial_forced_label": initial.get("_label_was_forced", False),
        "forced_label": decision.get("_label_was_forced", False),
        "retry_attempted": decision.get("_retry_attempted", False),
        "retry_failed": decision.get("_retry_failed", False),
        "final_decision_valid": valid,
        "figurative_type_predicted": language.get("figurative_type"),
        "figurative_type_was_guessed": language.get("_figurative_type_was_guessed", False),
        "figurative_type_source": language.get("_figurative_type_source", "unresolved"),
        "figurative_type_retry_attempted": language.get("_figurative_type_retry_attempted", False),
        "figurative_type_retry_failed": language.get("_figurative_type_retry_failed", False),
        "figurative_type_resolution_confidence": language.get("_figurative_type_resolution_confidence"),
        "figurative_type_resolution_scores": json.dumps(
            language.get("_figurative_type_resolution_scores", {}), sort_keys=True
        ),
        "claim_retry_attempted": language.get("_claim_retry_attempted", False),
        "claim_retry_success": language.get("_claim_retry_success", False),
        "claim_retry_seconds": language.get("_claim_retry_seconds", 0.0),
        "comparator_score": comparison.get("alignment_score"),
        "comparator_recommendation": comparison.get("recommendation"),
        "comparator_evidence_status": comparison.get("required_evidence_status"),
        "comparator_pre_verification_status": comparison.get(
            "_pre_verification_status"
        ),
        "comparator_evidence_quality": comparison.get("evidence_quality"),
        "comparator_claim_direction": comparison.get("claim_direction"),
        "comparator_relation_binding_required": comparison.get("relation_binding_required", False),
        "comparator_relation_binding_observed": comparison.get("relation_binding_observed", False),
        "comparator_has_symbolic_evidence": comparison.get(
            "has_symbolic_evidence", False
        ),
        "comparator_has_symbolic_object_candidate": comparison.get(
            "has_symbolic_object_candidate", False
        ),
        "comparator_has_text_surface": comparison.get(
            "has_text_surface", False
        ),
        "comparator_text_surface_without_ocr": comparison.get(
            "text_surface_without_ocr", False
        ),
        "comparator_shared_terms": text_list(comparison.get("shared_terms")),
        "comparator_direct_support_count": comparison.get("direct_support_count", 0),
        "comparator_direct_conflict_count": comparison.get("direct_conflict_count", 0),
        "comparator_grounded_anchor_count": len(comparison.get("grounded_anchor_evidence", []) or []),
        "agent1_visible_text": text_list(visual.get("visible_text")),
        "agent1_visible_text_count": visual.get("visible_text_count", 0),
        "agent1_visual_fact_count": visual.get("visual_fact_count", 0),
        "agent1_visual_relation_count": visual.get("visual_relation_count", 0),
        "agent1_symbolic_tone": visual.get("symbolic_tone", "None"),
        "agent1_visual_metaphor_count": len(
            visual.get("possible_visual_metaphors", []) or []
        ),
        "agent1_schema_complete": visual.get("schema_complete", False),
        "agent1_schema_format_valid": visual.get(
            "schema_format_valid", False
        ),
        "agent1_factual_grounding_present": visual.get(
            "factual_grounding_present", False
        ),
        "agent1_ocr_usable": visual.get("ocr_usable", False),
        "agent1_relation_binding_present": visual.get(
            "relation_binding_present", False
        ),
        "agent1_schema_issues": text_list(visual.get("schema_issues")),
        "agent1_targeted_recovery_attempted": visual.get(
            "_targeted_recovery_attempted", False
        ),
        "agent1_targeted_recovery_success": visual.get(
            "_targeted_recovery_success", False
        ),
        "agent1_targeted_recovery_reason": visual.get(
            "_targeted_recovery_reason", ""
        ),
        "agent1_targeted_recovery_seconds": visual.get(
            "_targeted_recovery_seconds", 0.0
        ),
        "agent1_schema_retry_attempted": visual.get("_schema_retry_attempted", False),
        "agent1_schema_retry_success": visual.get("_schema_retry_success", False),
        "agent1_visual_confidence": visual.get("visual_confidence"),
        "agent2_linguistic_cue": language.get("linguistic_cue", ""),
        "agent2_polarity_reversal": language.get("polarity_reversal", ""),
        "agent2_language_confidence": language.get("language_confidence"),
        "agent2_caption_proposition": language.get("caption_proposition", ""),
        "claim_relation_family": (language.get("claim_relation", {}) or {}).get(
            "relation_family", "unresolved"
        ),
        "claim_relation_polarity": (language.get("claim_relation", {}) or {}).get(
            "polarity", "unresolved"
        ),
        "claim_relation_predicate": (language.get("claim_relation", {}) or {}).get(
            "predicate", "unresolved"
        ),
        "claim_relation_resolved": (language.get("claim_relation", {}) or {}).get(
            "resolved", False
        ),
        "claim_contract_valid": claim_contract.get(
            "safe_for_directional_reasoning", False
        ),
        "claim_contract_proposition_preserved": claim_contract.get(
            "proposition_preserved", False
        ),
        "claim_contract_entity_frame_preserved": claim_contract.get(
            "entity_frame_preserved", False
        ),
        "claim_contract_warnings": text_list(claim_contract.get("warnings")),
        "claim_relation_pair_valid": claim_contract.get(
            "relation_pair_valid", False
        ),
        "claim_reasoning_requirement": claim_contract.get(
            "reasoning_requirement", "visual"
        ),
        "claim_safe_for_automatic_directional_reasoning": claim_contract.get(
            "safe_for_automatic_directional_reasoning", False
        ),
        "structured_relation_candidate_count": len(
            comparison.get("structured_relation_candidates", []) or []
        ),
        "review_board_status": review_board.get("status"),
        "review_board_binary_valid": review_board.get("binary_valid", False),
        "review_board_directionally_grounded": review_board.get(
            "directionally_grounded", False
        ),
        "review_board_source_grounded": review_board.get(
            "source_grounded", False
        ),
        "review_board_confidence_cap_applied": review_board.get(
            "confidence_cap_applied", False
        ),
        "review_board_confidence_before": review_board.get(
            "confidence_before_review"
        ),
        "review_board_confidence_after": review_board.get(
            "confidence_after_review"
        ),
        "judge_mode": judge.get("mode", "disabled"),
        "judge_scope": judge.get("scope", "escalated"),
        "judge_requested": judge.get("requested", False),
        "judge_status": judge.get("status", "disabled"),
        "pre_judge_prediction": judge_review.get("previous_label", prediction),
        "judge_trigger_reasons": text_list(judge.get("trigger_reasons")),
        "judge_verdict": judgment.get(
            "verdict", tribunal_review.get("provisional_verdict", "")
        ),
        "judge_confidence": judgment.get(
            "confidence", tribunal_review.get("confidence")
        ),
        "judge_format_valid": judgment.get(
            "_format_valid", tribunal_review.get("_format_valid", False)
        ),
        "judge_format_error": judgment.get(
            "_format_error", tribunal_review.get("_format_error", "")
        ),
        "judge_evidence_ids": text_list(
            judgment.get(
                "_valid_evidence_ids",
                tribunal_review.get("_valid_evidence_ids"),
            )
        ),
        "judge_invalid_evidence_ids": text_list(
            judgment.get(
                "_invalid_evidence_ids",
                tribunal_review.get("_invalid_evidence_ids"),
            )
        ),
        "judge_visual_observations": text_list(
            judgment.get(
                "visual_observations",
                tribunal_review.get("visual_observations"),
            )
        ),
        "judge_reason": judgment.get(
            "reason", tribunal_review.get("reason", "")
        ),
        "judge_agreed_with_pre_judge_decision": (
            bool(judgment.get("_format_valid", False))
            and judgment.get("verdict") == judge_review.get("previous_label")
        ),
        "judge_revision_accepted": judge_review.get("accepted", False),
        "judge_revision_reason": judge_review.get("reason", ""),
        "judge_changed_decision": judge_review.get("changed_decision", False),
        "judge_model": judgment.get(
            "_model_id", tribunal_review.get("_model_id", "")
        ),
        "judge_model_revision": judgment.get(
            "_model_revision", tribunal_review.get("_model_revision", "")
        ),
        "judge_feedback_candidate_recorded": judge_feedback.get(
            "recorded", False
        ),
        "judge_feedback_role": judge_feedback.get("role", ""),
        "judge_feedback_memory_update_applied": judge_feedback.get(
            "memory_update_applied", False
        ),
        "mediator_status": mediation.get("status", ""),
        "mediator_provisional_verdict": mediation.get(
            "provisional_verdict", ""
        ),
        "mediator_confidence": mediation.get("confidence"),
        "mediator_format_valid": mediation.get("_format_valid", False),
        "mediator_format_error": mediation.get("_format_error", ""),
        "mediator_evidence_ids": text_list(
            mediation.get("_valid_evidence_ids")
        ),
        "mediator_invalid_evidence_ids": text_list(
            mediation.get("_invalid_evidence_ids")
        ),
        "mediator_disputed_issues": text_list(
            mediation.get("disputed_issues")
        ),
        "mediator_agent1_questions": text_list(
            mediation.get("agent1_questions")
        ),
        "mediator_agent2_questions": text_list(
            mediation.get("agent2_questions")
        ),
        "mediator_verification_requests": text_list(
            mediation.get("verification_requests")
        ),
        "mediator_reason": mediation.get("reason", ""),
        "mediator_usable": mediation.get("_usable", False),
        "mediator_tiebreak_used": str(
            judge_review.get("reason", "")
        ).startswith("accepted_mediated_verified_tiebreak:"),
        "tribunal_state": tribunal_session.get("state", ""),
        "tribunal_round_count": len(tribunal_session.get("rounds", []) or []),
        "tribunal_stop_reason": tribunal_session.get("stop_reason", ""),
        "tribunal_review_status": tribunal_review.get("status", ""),
        "tribunal_revision_accepted": tribunal_resolution.get(
            "accepted", False
        ),
        "tribunal_revision_reason": tribunal_resolution.get("reason", ""),
        "tribunal_corroboration_reason": (
            tribunal_resolution.get("corroboration", {}) or {}
        ).get("reason", ""),
        "tribunal_acceptance_checks": json.dumps(
            tribunal_resolution.get("acceptance_checks", []),
            ensure_ascii=True,
        ),
        "tribunal_verified_evidence_id": tribunal_resolution.get(
            "verified_evidence_id", ""
        ),
        "arbiter_evidence_assessment": decision.get("_arbiter_assessment", ""),
        "final_reason": decision.get("explanation", ""),
        "arbiter_visual_support": text_list(decision.get("visual_support")),
        "arbiter_contradictions": text_list(decision.get("contradictions")),
        "arbiter_missing_evidence": text_list(decision.get("missing_evidence")),
        "agent1_seconds": round(timing.get("agent1_seconds", 0.0), 4),
        "agent2_seconds": round(timing.get("agent2_seconds", 0.0), 4),
        "comparator_seconds": round(timing.get("comparator_seconds", 0.0), 4),
        "arbiter_primary_seconds": round(timing.get("arbiter_primary_seconds", 0.0), 4),
        "citation_retry_seconds": round(
            timing.get("citation_retry_seconds", 0.0), 4
        ),
        "format_retry_seconds": round(timing.get("format_retry_seconds", 0.0), 4),
        "binary_resolution_seconds": round(timing.get("binary_resolution_seconds", 0.0), 4),
        "debate_seconds": round(timing.get("debate_seconds", 0.0), 4),
        "judge_seconds": round(timing.get("judge_seconds", 0.0), 4),
        "mediator_seconds": round(timing.get("mediator_seconds", 0.0), 4),
        "feedback_mode": feedback.get("mode", "disabled"),
        "feedback_enabled": feedback.get("enabled", False),
        "feedback_batch": feedback.get("batch_index"),
        "feedback_memory_active": feedback.get("memory_active", False),
        "feedback_update_applied": feedback.get("update_applied", False),
        "feedback_candidate_recorded": feedback.get("candidate_recorded", False),
        "feedback_failure_type": feedback.get("failure_type"),
        "feedback_agent1_memory_before": feedback.get("agent1_memory_size_before", 0),
        "feedback_agent2_memory_before": feedback.get("agent2_memory_size_before", 0),
        "feedback_arbiter_memory_before": feedback.get("arbiter_memory_size_before", 0),
        "feedback_agent1_memory_after": feedback.get("agent1_memory_size_after", 0),
        "feedback_agent2_memory_after": feedback.get("agent2_memory_size_after", 0),
        "feedback_arbiter_memory_after": feedback.get("arbiter_memory_size_after", 0),
        "feedback_target_agent": feedback.get("feedback_target_agent"),
        "feedback_matched_rule_ids": text_list(matched_rule_ids),
        "feedback_matched_rule_count": len(matched_rule_ids),
        "feedback_available_rule_count": feedback.get("available_rule_count", 0),
        "feedback_matched_rule_scores": json.dumps(
            feedback.get("matched_rule_scores", {}), sort_keys=True
        ),
        "feedback_baseline_prediction": feedback.get(
            "feedback_baseline_label", pre_feedback.get("label")
        ),
        "feedback_candidate_prediction": feedback.get(
            "feedback_candidate_label", feedback_candidate.get("label")
        ),
        "feedback_post_review_prediction": feedback.get(
            "feedback_post_review_label", initial.get("label")
        ),
        "feedback_revision_accepted": feedback.get(
            "feedback_revision_accepted", False
        ),
        "feedback_revision_reason": feedback.get("feedback_revision_reason", ""),
        "feedback_decision_changed": feedback.get(
            "feedback_baseline_label", pre_feedback.get("label")
        ) != feedback.get("feedback_post_review_label", initial.get("label")),
        "feedback_role": feedback.get("role", ""),
        "feedback_reliability_updates": json.dumps(
            feedback.get("reliability_updates", []), sort_keys=True
        ),
        "evidence_ledger_count": len(ledger),
        "evidence_support_count": sum(item.get("relation") == "SUPPORT" for item in ledger),
        "evidence_conflict_count": sum(item.get("relation") == "CONFLICT" for item in ledger),
        "evidence_anchor_count": sum(item.get("relation") == "ANCHOR" for item in ledger),
        "debate_visual_evidence_count": len(debate_visual_evidence),
        "debate_visual_evidence_ids": text_list(
            item.get("id") for item in debate_visual_evidence
        ),
        "debate_visual_witness_count": len(debate_visual_witnesses),
        "debate_visual_witness_ids": text_list(
            item.get("id") for item in debate_visual_witnesses
        ),
        "evidence_level_summary": json.dumps(
            evidence_level_summary, sort_keys=True
        ),
        "evidence_ledger_json": json.dumps(ledger, ensure_ascii=True),
        "evidence_lifecycle_summary": json.dumps(
            debate.get("evidence_lifecycle", {}), sort_keys=True
        ),
        "arbiter_relation_status": decision.get("_relation_status", ""),
        "arbiter_revision_status": decision.get("_revision_status", ""),
        "arbiter_deficiencies": text_list(decision.get("_deficiencies", [])),
        "initial_evidence_status": initial_audit.get("status"),
        "initial_evidence_valid": initial_audit.get("valid", False),
        "initial_cited_evidence_ids": text_list(initial_audit.get("cited_evidence_ids")),
        "initial_source_evidence_valid": initial_audit.get("source_valid", False),
        "initial_source_cited_evidence_ids": text_list(
            initial_audit.get("source_cited_evidence_ids")
        ),
        "final_evidence_status": final_audit.get("status"),
        "final_evidence_valid": final_audit.get("valid", False),
        "final_cited_evidence_ids": text_list(final_audit.get("cited_evidence_ids")),
        "final_source_evidence_valid": final_audit.get("source_valid", False),
        "final_source_cited_evidence_ids": text_list(
            final_audit.get("source_cited_evidence_ids")
        ),
        "debate_proposed_evidence_status": proposed_audit.get("status"),
        "debate_proposed_evidence_valid": proposed_audit.get("valid", False),
        "debate_proposed_cited_evidence_ids": text_list(
            proposed_audit.get("cited_evidence_ids")
        ),
        "debate_proposed_source_evidence_valid": proposed_audit.get(
            "source_valid", False
        ),
        "debate_proposed_source_cited_evidence_ids": text_list(
            proposed_audit.get("source_cited_evidence_ids")
        ),
        "evidence_verifier_candidate_count": evidence_verification.get(
            "candidate_count", 0
        ),
        "evidence_verifier_verified_count": evidence_verification.get(
            "verified_count", 0
        ),
        "evidence_verifier_support_count": evidence_verification.get(
            "support_count", 0
        ),
        "evidence_verifier_conflict_count": evidence_verification.get(
            "conflict_count", 0
        ),
        "evidence_verifier_neutral_count": evidence_verification.get(
            "neutral_count", 0
        ),
        "evidence_verifier_model": evidence_verification.get("model", ""),
        "evidence_verifier_revision": evidence_verification.get("revision", ""),
        "evidence_verifier_seconds": round(
            timing.get("evidence_verifier_seconds", 0.0), 4
        ),
        "targeted_region_verification_method": targeted_verification.get(
            "method", ""
        ),
        "targeted_region_verification_decision_grade": targeted_verification.get(
            "decision_grade", False
        ),
        "targeted_region_verification_reason": targeted_verification.get(
            "reason", ""
        ),
        "decision_trace_json": json.dumps(
            decision.get("_decision_trace", []),
            ensure_ascii=True,
            sort_keys=True,
        ),
        "feedback_review_seconds": round(
            timing.get("feedback_review_seconds", 0.0), 4
        ),
        "runtime_seconds": round(timing.get("sample_inference_seconds", elapsed), 4),
        "correct": valid and prediction == raw["label"],
        "trace": {
            "visual_output": result.get("visual_output", {}),
            "language_output": language, "comparison": comparison,
            "initial_decision": initial, "final_decision": decision,
            "debate_details": debate,
            "evidence_ledger": ledger,
            "feedback": feedback,
            "pre_feedback_decision": pre_feedback,
            "feedback_candidate_decision": feedback_candidate,
            "evidence_verification": evidence_verification,
            "judge": judge,
        },
    }


def append_record(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_predictions(path, records):
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["sample"]):
            writer.writerow({field: record.get(field) for field in FIELDNAMES})
    os.replace(temp_path, path)


def write_debate_logs(run_dir, records):
    jsonl_path = os.path.join(run_dir, "debate_log.jsonl")
    csv_path = os.path.join(run_dir, "debate_log.csv")
    rows = []
    for record in sorted(records, key=lambda item: item["sample"]):
        if not record.get("debate_triggered"):
            continue
        initial_correct = record.get("initial_prediction") == record.get("ground_truth")
        final_correct = record.get("prediction") == record.get("ground_truth")
        outcome = "unchanged"
        if not initial_correct and final_correct:
            outcome = "corrected"
        elif initial_correct and not final_correct:
            outcome = "harmed"
        trace = record.get("trace", {}) or {}
        debate = trace.get("debate_details", {}) or {}
        rows.append({
            "sample": record.get("sample"),
            "id": record.get("id"),
            "phenomenon": record.get("phenomenon"),
            "ground_truth": record.get("ground_truth"),
            "trigger_reason": record.get("debate_trigger_reason"),
            "debate_level": record.get("debate_level"),
            "debate_need_score": record.get("debate_need_score"),
            "debate_need_signals": record.get("debate_need_signals"),
            "initial_label": record.get("initial_prediction"),
            "proposed_label": record.get("debate_proposed_label"),
            "final_label": record.get("prediction"),
            "revision_accepted": record.get("debate_revision_accepted"),
            "review_status": record.get("debate_review_status"),
            "acceptance_reason": record.get("debate_revision_reason"),
            "outcome": outcome,
            "initial_confidence": record.get("round1_confidence"),
            "final_confidence": record.get("final_confidence"),
            "debate_seconds": record.get("debate_seconds"),
            "agent1_critique": debate.get("agent1_critique", {}),
            "agent2_critique": debate.get("agent2_critique", {}),
            "advocates": debate.get("advocates", {}),
            "original_evidence_audit": debate.get("original_evidence_audit", {}),
            "proposed_evidence_audit": debate.get("proposed_evidence_audit", {}),
            "evidence_ledger_before": debate.get("evidence_ledger_before", []),
            "evidence_ledger_after": debate.get("evidence_ledger_after", []),
            "proposed_decision": debate.get("proposed_decision", {}),
        })

    temp_jsonl = f"{jsonl_path}.tmp"
    with open(temp_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=True)
            handle.write("\n")
    os.replace(temp_jsonl, jsonl_path)

    csv_fields = (
        "sample", "id", "phenomenon", "ground_truth", "trigger_reason",
        "debate_level", "debate_need_score", "debate_need_signals",
        "initial_label", "proposed_label", "final_label", "revision_accepted",
        "review_status", "acceptance_reason", "outcome", "initial_confidence",
        "final_confidence", "debate_seconds", "agent1_critique", "agent2_critique",
        "advocates",
        "original_evidence_audit", "proposed_evidence_audit",
        "evidence_ledger_before", "evidence_ledger_after", "proposed_decision",
    )
    temp_csv = f"{csv_path}.tmp"
    with open(temp_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row[key], ensure_ascii=True)
                if isinstance(row[key], (dict, list)) else row[key]
                for key in csv_fields
            })
    os.replace(temp_csv, csv_path)
    return jsonl_path, csv_path


def write_feedback_decision_logs(run_dir, records):
    jsonl_path = os.path.join(run_dir, "feedback_decision_log.jsonl")
    csv_path = os.path.join(run_dir, "feedback_decision_log.csv")
    rows = []
    for record in sorted(records, key=lambda item: item["sample"]):
        if not record.get("feedback_enabled"):
            continue
        baseline = record.get("feedback_baseline_prediction")
        post_review = record.get("feedback_post_review_prediction")
        gold = record.get("ground_truth")
        outcome = "unchanged"
        if baseline != gold and post_review == gold:
            outcome = "corrected"
        elif baseline == gold and post_review != gold:
            outcome = "harmed"
        trace = record.get("trace", {}) or {}
        rows.append({
            "sample": record.get("sample"),
            "id": record.get("id"),
            "phenomenon": record.get("phenomenon"),
            "ground_truth": gold,
            "matched_memory_ids": record.get("feedback_matched_rule_ids"),
            "matched_memory_scores": record.get("feedback_matched_rule_scores"),
            "baseline_label": baseline,
            "candidate_label": record.get("feedback_candidate_prediction"),
            "post_review_label": post_review,
            "revision_accepted": record.get("feedback_revision_accepted"),
            "revision_reason": record.get("feedback_revision_reason"),
            "outcome": outcome,
            "pre_feedback_decision": trace.get("pre_feedback_decision", {}),
            "feedback_candidate_decision": trace.get(
                "feedback_candidate_decision", {}
            ),
            "evidence_ledger": trace.get("evidence_ledger", []),
        })
    fields = (
        "sample", "id", "phenomenon", "ground_truth", "matched_memory_ids",
        "matched_memory_scores", "baseline_label", "candidate_label",
        "post_review_label", "revision_accepted", "revision_reason", "outcome",
        "pre_feedback_decision", "feedback_candidate_decision", "evidence_ledger",
    )
    temp_jsonl = f"{jsonl_path}.tmp"
    with open(temp_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=True)
            handle.write("\n")
    os.replace(temp_jsonl, jsonl_path)
    temp_csv = f"{csv_path}.tmp"
    with open(temp_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row[key], ensure_ascii=True)
                if isinstance(row[key], (dict, list)) else row[key]
                for key in fields
            })
    os.replace(temp_csv, csv_path)
    return jsonl_path, csv_path


def main():
    args = parse_args()
    set_reproducibility(args.seed)
    if args.feedback_mode == "verified" and not args.verified_feedback_file:
        raise ValueError("Verified feedback mode requires --verified-feedback-file.")
    if args.feedback_mode != "disabled" and args.execution_mode != "stagewise":
        raise ValueError(
            "Feedback modes require --execution-mode stagewise."
        )
    if args.evidence_mode == "disabled" and args.execution_mode != "stagewise":
        raise ValueError("Evidence ablation requires --execution-mode stagewise.")
    if args.judge_mode != "disabled" and args.execution_mode != "stagewise":
        raise ValueError("Judge modes require --execution-mode stagewise.")
    if args.feedback_mode == "calibrate" and args.dataset_split == "vflute_test":
        raise ValueError(
            "Calibration must never use vflute_test. Use vflute_train_dev50 or vflute_val."
        )
    run_dir = resolve_run_dir(args)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "paper_assets"), exist_ok=True)
    records_path = os.path.join(run_dir, "records.jsonl")
    predictions_path = os.path.join(run_dir, "predictions.csv")
    if os.path.exists(records_path) and not args.resume:
        raise ValueError(
            "The run directory already contains records.jsonl. Use --resume "
            "with the identical configuration or choose a new --run-dir."
        )
    existing = load_records(records_path) if args.resume else {}
    dataset = load_split(args.dataset_split)
    selected = select_records(
        dataset,
        args.num_samples,
        strategy=args.selection_strategy,
        seed=args.seed,
    )
    pending = [(index, raw) for index, raw in enumerate(selected) if raw["id"] not in existing]

    run_config = {
        "system_name": "FigDebate", "dataset": args.dataset_split,
        "requested_samples": len(selected), "completed_before_run": len(existing),
        "execution_mode": args.execution_mode,
        "debate_mode": args.debate_mode,
        "debate_enabled": args.debate_mode == "enabled",
        "evidence_mode": args.evidence_mode,
        "judge_mode": args.judge_mode,
        "judge_scope": args.judge_scope,
        "feedback_enabled": args.feedback_mode != "disabled",
        "feedback_mode": args.feedback_mode,
        "verified_feedback_file": args.verified_feedback_file,
        "verified_feedback_sha256": file_checksum(args.verified_feedback_file),
        "model_vision": VISION_MODEL_ID,
        "model_vision_revision": VISION_MODEL_REVISION,
        "model_language": LANGUAGE_MODEL_ID,
        "model_language_revision": LANGUAGE_MODEL_REVISION,
        "model_judge": JUDGE_MODEL_ID,
        "model_judge_revision": JUDGE_MODEL_REVISION,
        "seed": args.seed,
        "selection_strategy": args.selection_strategy,
        "dataset_selection_sha256": dataset_selection_checksum(selected),
        "pipeline_source_sha256": pipeline_source_checksum(),
        "evidence_ledger_version": EVIDENCE_LEDGER_VERSION,
        "evidence_verifier_thresholds": {
            "minimum_relation_probability": (
                AtomicEvidenceVerifier.MIN_RELATION_PROBABILITY
            ),
            "minimum_relation_margin": AtomicEvidenceVerifier.MIN_RELATION_MARGIN,
            "maximum_atomic_items": AtomicEvidenceVerifier.MAX_ATOMIC_ITEMS,
        },
        "evidence_policy": {
            "generic_nli": "diagnostic_candidate_only",
            "lexical_relation_matches": "candidate_only_until_independently_verified",
            "source_attribution": "current_sample_ledger_id_required",
            "directional_proof": (
                "verified_relation_from_deterministic_check_or_independent_multimodal_corroboration"
            ),
        },
        "debate_policy": {
            "routing": "auditable_need_score",
            "level_1": "independent_decision_grade_evidence_deliberation",
            "level_2": "adaptive_regrounding_then_independent_multimodal_reinspection",
            "revision_gate": "deterministic_review_board_requires_stronger_current_image_evidence",
        },
        "judge_policy": {
            "position": (
                "inside_debate_before_agent_reviews"
                if args.judge_mode == "mediated"
                else "bounded_mediator_before_between_and_after_agent_reviews"
                if args.judge_mode == "tribunal"
                else "after_existing_arbiter_and_debate"
            ),
            "primary_label_visible": False,
            "gold_label_visible": False,
            "shadow_changes_predictions": False,
            "appellate_revision_gate": (
                "strict_contract_plus_stronger_cited_decision_grade_current_image_evidence"
            ),
            "mediated_revision_gate": (
                "verified_directional_evidence_plus_narrow_independent_mediation_tiebreak"
            ),
            "mediator_provisional_label_visible_to_agents": False,
            "judge_generated_text_is_visual_proof": False,
            "tribunal_maximum_rounds": 2,
            "tribunal_follow_up": "one_neutral_atomic_question_per_agent",
            "tribunal_revision_gate": (
                "ordinary_review_board_plus_independent_current_image_verification"
            ),
        },
        "feedback_policy": {
            "retrieval": "strict_procedural_case_similarity",
            "role": "diagnostic_question_and_debate_routing_only",
            "online_label_updates": False,
            "gold_direction_stored": False,
        },
        "feedback_minimum_case_similarity": FeedbackLoop.MIN_CASE_SIMILARITY,
        "comparator": "evidence_comparator", "python_version": sys.version,
        "platform": platform.platform(),
        "runtime_environment": runtime_environment(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if args.resume:
        original_config = validate_resume_config(run_dir, run_config)
        run_config["original_timestamp"] = original_config.get("timestamp")
        run_config["resume_timestamp"] = run_config["timestamp"]
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    def record_result(index, raw, result, elapsed):
        record = build_record(index, raw, result, elapsed)
        existing[raw["id"]] = record
        append_record(records_path, record)
        print(f"Saved checkpoint for {raw['id']} ({elapsed:.2f} sec)")

    run_started = time.time()
    run_timing = {}
    if pending:
        if args.execution_mode == "stagewise":
            samples = [
                {"index": index, "raw": raw, "image": decode_image(raw["image_bytes"]), "caption": raw["caption"]}
                for index, raw in pending
            ]
            runner = StagewiseRunner(
                feedback_mode=args.feedback_mode,
                feedback_log_path=os.path.join(run_dir, "feedback_log.json"),
                verified_feedback_path=args.verified_feedback_file,
                debate_mode=args.debate_mode,
                evidence_mode=args.evidence_mode,
                judge_mode=args.judge_mode,
                judge_scope=args.judge_scope,
            )
            run_timing = runner.run_samples(samples, record_result) or {}
            if args.feedback_mode != "disabled":
                with open(
                    os.path.join(run_dir, "feedback_events.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(runner.feedback_events, handle, indent=2)
            if args.feedback_mode == "calibrate":
                with open(
                    os.path.join(run_dir, "calibrated_feedback.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(runner.export_feedback_examples(), handle, indent=2)
        else:
            system = FigDebate()
            for index, raw in pending:
                start = time.time()
                result = system.predict(decode_image(raw["image_bytes"]), raw["caption"])
                record_result(index, raw, result, time.time() - start)

    run_timing["wall_clock_seconds"] = round(time.time() - run_started, 4)
    with open(os.path.join(run_dir, "run_timing.json"), "w", encoding="utf-8") as handle:
        json.dump(run_timing, handle, indent=2)

    records = [existing[raw["id"]] for raw in selected if raw["id"] in existing]
    write_predictions(predictions_path, records)
    debate_jsonl, debate_csv = write_debate_logs(run_dir, records)
    feedback_jsonl, feedback_csv = write_feedback_decision_logs(run_dir, records)
    metrics = evaluate_predictions(predictions_path, run_dir)
    valid = sum(record["final_decision_valid"] for record in records)
    correct = sum(record["correct"] for record in records)
    print("\nFIGDEBATE RUN COMPLETE")
    print(f"Completed samples: {len(records)}/{len(selected)}")
    print(f"Valid decisions: {valid}")
    print(f"Correct decisions: {correct}")
    print(f"Run folder: {run_dir}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {os.path.join(run_dir, 'metrics_summary.txt')}")
    print(f"Debate log (full): {debate_jsonl}")
    print(f"Debate log (table): {debate_csv}")
    print(f"Feedback decision log (full): {feedback_jsonl}")
    print(f"Feedback decision log (table): {feedback_csv}")
    print(f"Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
