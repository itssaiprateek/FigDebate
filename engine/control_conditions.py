"""Explicit simpler controls using the same inputs/models, never tribunal gates.

The conventional control is a local two-agent, two-round peer debate, not a
claim of exact reproduction of any published system. Compute matching must
be demonstrated from actual accounting; neither control is called matched here.
"""
from collections import Counter
from copy import deepcopy
import time
from engine.candidate_cases import candidate_case, TextCandidateRuntime
from engine.gpu_manager import GPUManager
from engine.reproducibility import seed_stage


def public_case(record):
    if not record or not record.get("_format_valid") or record.get("_execution_status") != "SUCCEEDED":
        return {"status": "peer_unavailable"}
    return {key: record.get(key) for key in (
        "relation", "claim_reading", "decisive_observation", "alternative", "decisive_question")}


def aggregate(base, cases, mode):
    valid = [case for case in cases if case.get("_format_valid")
             and case.get("_execution_status") == "SUCCEEDED"]
    votes = Counter(case["relation"] for case in valid if case.get("relation") in {"SUPPORT", "CONFLICT"})
    winner = next((label for label, count in votes.items() if count > len(cases) / 2), None)
    decision = deepcopy(base)
    if winner:
        decision = dict(label={"SUPPORT": "ENTAILS", "CONFLICT": "CONTRADICTS"}[winner],
                        confidence=None, decision_method=mode, _model_cited_evidence_ids=[],
                        _final_decision_valid=False,
                        explanation=f"The {mode} control selected {winner} by strict majority of candidate votes. These votes are not verified evidence.")
    return decision, {"mode": mode, "rounds": 2 if mode == "conventional_debate" else 1,
                      "votes": dict(votes), "resolved": winner is not None,
                      "baseline_fallback": winner is None, "compute_match_established": False,
                      "fallback_reason": "no_strict_majority_including_unavailable_agents" if not winner else None,
                      "decision_grade": False, "cases": deepcopy(cases)}


def run_control_batch(samples, results, mode, hardware_profile, global_seed):
    initial = {sample["index"]: [
        deepcopy(results[sample["index"]]["visual_output"].get("_independent_candidate") or {}),
        deepcopy(results[sample["index"]]["language_output"].get("_independent_candidate") or {}),
    ] for sample in samples}
    final = deepcopy(initial)
    loads = {"vision_model_load_seconds": 0.0, "language_model_load_seconds": 0.0}
    if mode == "conventional_debate":
        from models.vision_model import Qwen3VLVisionModel
        from models.language_model import MistralModel
        runtime = None
        try:
            started = time.perf_counter()
            runtime = Qwen3VLVisionModel(hardware_profile=hardware_profile)
            loads["vision_model_load_seconds"] += time.perf_counter() - started
            for sample in samples:
                index = sample["index"]
                seed_stage(global_seed, sample["raw"]["id"], "control_peer_visual")
                final[index][0] = candidate_case(runtime, sample["image"], sample["caption"],
                    model_family="Qwen3-VL", peer_case=public_case(initial[index][1]))
        finally:
            if runtime is not None:
                del runtime
            GPUManager.clear()
        runtime = adapter = None
        try:
            started = time.perf_counter()
            runtime = MistralModel()
            loads["language_model_load_seconds"] += time.perf_counter() - started
            adapter = TextCandidateRuntime(runtime.model, runtime.tokenizer)
            for sample in samples:
                index = sample["index"]
                seed_stage(global_seed, sample["raw"]["id"], "control_peer_text")
                observations = [item.get("text", "") for item in results[index]["evidence_ledger"]
                                if item.get("source") == "agent1" and item.get("grounded")]
                # Both agents receive only the previous round, never a sequentially
                # updated answer from the same round.
                final[index][1] = candidate_case(adapter, None, sample["caption"], observations,
                    model_family="Mistral", peer_case=public_case(initial[index][0]))
        finally:
            if adapter is not None:
                del adapter
            if runtime is not None:
                del runtime
            GPUManager.clear()
    for sample in samples:
        index = sample["index"]
        result = results[index]
        decision, control = aggregate(result["decision"], final[index], mode)
        control["initial_cases"] = initial[index]
        result["control"] = control
        result["decision"] = decision
        result["round2_confidence"] = decision.get("confidence")
        if mode == "conventional_debate":
            result["timing"]["candidate_seconds"] += sum(case.get("_generation_seconds", 0.0) for case in final[index])
    return loads
