"""Bounded live first-review + production-gate replay on frozen saved hearings.

No upstream inference, later witness hearing, or evaluation labels in model inputs.
Selected development cases are regression checks, never held-out accuracy estimates.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def frozen_input(checkpoint):
    result = checkpoint["payload"]
    review = result["judge"]["tribunal_reviews"][0]
    packet = review["_judge_packet"]
    ids = {e["id"] for key in ("evidence_ledger", "remaining_evidence_index") for e in packet.get(key, [])}
    ledger = [deepcopy(e) for e in result["evidence_ledger"] if e["id"] in ids]
    if {e["id"] for e in ledger} != ids:
        raise ValueError("Cannot reconstruct complete pre-review evidence catalogue")
    # Round-one checkpoint preserves its hearing. Exclude all newly promoted proof
    # and all judge text. Never replay a final record containing a later hearing.
    value = {k: deepcopy(result.get(k, {})) for k in
             ("visual_output", "language_output", "comparison", "debate_details", "pre_hearing")}
    value["evidence_ledger"] = ledger
    value["decision"] = deepcopy(result["initial_decision"])
    if value["decision"]["label"] != result["judge"]["tribunal_resolution"]["previous_label"]:
        raise ValueError("This reconstruction requires an unchanged pre-tribunal decision")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-ids", nargs="+", required=True)
    parser.add_argument("--code-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--precedents")
    parser.add_argument("--diagnostic-guidance-file", help="Oracle-informed interpretation for a single diagnostic case; never accuracy evidence")
    args = parser.parse_args()
    guidance = Path(args.diagnostic_guidance_file).read_text(encoding="utf-8") if args.diagnostic_guidance_file else None
    guidance_by_id = json.loads(guidance) if guidance and guidance.lstrip().startswith('{') else None
    if guidance_by_id is not None:
        if not isinstance(guidance_by_id, dict) or set(guidance_by_id) != set(args.sample_ids) or any(not isinstance(v, str) or len(v)>4000 for v in guidance_by_id.values()):
            raise ValueError('Guidance must map each selected diagnostic ID to bounded text')
    elif guidance and (len(args.sample_ids) != 1 or len(guidance) > 4000):
        raise ValueError("Diagnostic guidance requires one case and at most 4000 characters")
    if len(args.sample_ids) > 6 or len(set(args.sample_ids)) != len(args.sample_ids):
        raise ValueError("Use one to six unique cases for a targeted check")
    sys.path.insert(0, str(Path(args.code_root).resolve()))
    from agents.multimodal_judge import TribunalMediatorAgent
    from models.judge_model import QwenJudgeModel
    from dataset.loaders import load_split, decode_image
    from engine.reproducibility import seed_stage
    from engine.runtime_accounting import begin_accounting, sample_accounting
    from engine.tribunal import apply_tribunal_resolution
    from run_figdebate import pipeline_source_checksum
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=False)
    source = Path(args.source_run)
    config = json.loads((source / "run_config.json").read_text())
    saved = {}
    identities = {}
    for f in (source / "stage_checkpoints").glob("tribunal_round_1_*.json"):
        c = json.loads(f.read_text(encoding="utf-8"))
        if c["sample_id"] in args.sample_ids:
            saved[c["sample_id"]] = frozen_input(c)
            identities[c['sample_id']] = c['input_identity']
    if set(saved) != set(args.sample_ids):
        raise ValueError("Missing saved first-review inputs")
    data = {r["id"]: r for r in load_split(config["dataset"]) if r["id"] in saved}
    recorded = {r["id"]: r for r in map(json.loads, (source / "records.jsonl").read_text(encoding="utf-8").splitlines())}
    for key, raw in data.items():
        from engine.stage_checkpoint import StageCheckpointStore
        if identities[key] != StageCheckpointStore.input_identity({'raw':raw,'caption':raw['caption']}):
            raise ValueError("Saved hearing and current image/caption identity differ")
    library = None
    if args.precedents:
        from engine.tribunal_feedback import TribunalPrecedents
        library = TribunalPrecedents(args.precedents)
        library.assert_disjoint([dict(r, caption_sha256=hashlib.sha256(r["caption"].encode()).hexdigest()) for r in data.values()])
    (target / "manifest.json").write_text(json.dumps({
        "purpose": "targeted_first_review_and_production_gate_not_full_run_or_accuracy_estimate",
        "sample_ids": args.sample_ids, "source_run": str(source.resolve()),
        "pipeline_source_sha256": pipeline_source_checksum(),
        "frozen_input_sha256": {k: digest(v) for k, v in saved.items()},
        "precedent_sha256": library.sha256 if library else None,
        "diagnostic_guidance": guidance,
        "oracle_informed_intervention": bool(guidance),
        "guidance_is_visual_evidence": False,
        "seed": config.get("seed", 42), "max_rounds": 1,
        "excluded_stages": ["initial_arbiter", "new_witness_hearing", "second_tribunal_round"]}, indent=2), encoding="utf-8")
    begin_accounting()
    started = time.perf_counter()
    frozen_hashes={k:digest(v) for k,v in saved.items()}
    runtime = QwenJudgeModel(hardware_profile="paper-8gb")
    agent = TribunalMediatorAgent(runtime)
    for sample_id in args.sample_ids:
        raw, state = data[sample_id], deepcopy(saved[sample_id])
        if guidance:
            state["debate_details"]["tribunal_semantic_questions"] = {
                "questions": [guidance_by_id[sample_id] if guidance_by_id else guidance], "role": "tribunal_only",
                "is_new_visual_evidence": False,
                "origin": "human_supplied_oracle_informed_diagnostic_interpretation"}
        seed_stage(config.get("seed", 42), sample_id, "tribunal_review", 1)
        kwargs = {k: state[k] for k in ("visual_output", "language_output", "comparison", "evidence_ledger", "debate_details", "pre_hearing")}
        if library:
            kwargs["precedents"] = library.retrieve(state["language_output"])
        review = agent.review(decode_image(raw["image_bytes"]), raw["caption"],
                              current_decision=state["decision"], round_number=1, **kwargs)
        debate = state["debate_details"]
        import inspect
        gate_kwargs = {"source_caption": raw["caption"]} if "source_caption" in inspect.signature(apply_tribunal_resolution).parameters else {}
        decision, _, resolution = apply_tribunal_resolution(state["decision"], review, state["evidence_ledger"],
            state["language_output"].get("claim_contract", {}),
            agent2_requirements_valid=debate.get("agent2_requirements_valid", True),
            agent1_critique=debate.get("agent1_critique", {}), agent2_critique=debate.get("agent2_critique", {}),
            semantic_bridge_mode="corroborated", language_output=state["language_output"], **gate_kwargs)
        record = {"id": sample_id, "frozen_input_sha256": frozen_hashes[sample_id],
                  "initial": state["decision"]["label"], "final": decision["label"],
                  "review": review, "resolution": resolution,
                  "accounting": sample_accounting(sample_id), "wall_seconds": time.perf_counter() - started}
        with (target / "records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        print(json.dumps({"id": sample_id, "initial": record["initial"], "final": record["final"],
                          "proposal": review.get("provisional_verdict"), "reason": resolution["reason"],
                          "seconds": review.get("_generation_seconds")}), flush=True)
    (target / "complete.json").write_text(json.dumps({"completed": len(saved), "wall_seconds": time.perf_counter() - started}), encoding="utf-8")


if __name__ == "__main__":
    main()
