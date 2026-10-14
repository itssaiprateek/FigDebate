"""Paired first-review replay exercising the real tribunal acceptance gate.

Frozen pre-review observations, no repair hearing or upstream re-execution.
Selected development cases cannot establish held-out accuracy.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import time


def image_for(identifier, expected_hash):
    import pyarrow.parquet as pq
    from PIL import Image
    index = int(identifier.rsplit("_",1)[1])
    for path in sorted(Path(".cache/upstream-vflute/data").glob("train-*.parquet")):
        file = pq.ParquetFile(path)
        if index >= file.metadata.num_rows:
            index -= file.metadata.num_rows
            continue
        value = file.read(columns=["image"]).slice(index,1).to_pylist()[0]["image"]["bytes"]
        if hashlib.sha256(value).hexdigest() != expected_hash:
            raise ValueError("Image bytes differ from saved run")
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise ValueError("Source image unavailable")


def main():
    from project_environment import configure
    configure()
    from evaluation.claim_pair_probe import candidate_context
    from evaluation.structural_variants import enable_evaluation_modes
    enable_evaluation_modes()
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    from models.judge_model import QwenJudgeModel
    from agents.multimodal_judge import TribunalMediatorAgent
    from engine.tribunal import apply_tribunal_resolution
    from engine.reproducibility import seed_stage
    from engine.runtime_accounting import begin_accounting, set_scope, sample_accounting
    from run_figdebate import pipeline_source_checksum
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-runs",nargs="+",type=Path,required=True)
    p.add_argument("--sample-ids",nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--variants",nargs="+",choices=["baseline","grounded","aligned","focused","questions","mapping","coherent","combined","conditions","condition_repair","reason_first","canonical","examples","paired","paired_defined","direct"],default=["baseline","aligned","focused"])
    p.add_argument('--oracle-file', type=Path, help='Evaluation-only checked stage substitutions; excluded from performance claims.')
    p.add_argument('--bounded-semantic-repair', action='store_true', help='Allow one real tribunal-only repair within the original case budget; witness routes are reported, not simulated.')
    args=p.parse_args()
    from evaluation.stage_substitution import load_interventions, substitute_stage
    interventions, intervention_hash = load_interventions(args.oracle_file, args.sample_ids)
    if interventions and args.variants not in (['conditions'], ['condition_repair']):
        raise ValueError('Oracle diagnostic needs a single condition variant in a separate run')
    if len(args.sample_ids)>10 or len(set(args.sample_ids))!=len(args.sample_ids):
        raise ValueError("At most ten unique cases")
    rows={}; sources=[]
    for path in args.source_runs:
        content=(path/"records.jsonl").read_bytes()
        sources.append(dict(path=str(path),sha256=hashlib.sha256(content).hexdigest()))
        for line in content.decode().splitlines():
            r=json.loads(line)
            if r['id'] in args.sample_ids:
                if r['id'] in rows: raise ValueError("Duplicate source case")
                rows[r['id']]=r
    if set(rows)!=set(args.sample_ids): raise ValueError("Missing cases")
    images={k:image_for(k,r['image_sha256']) for k,r in rows.items()}
    args.output.mkdir(parents=True,exist_ok=False)
    from engine.run_provenance import snapshot_source
    snapshot_source(str(Path(__file__).resolve().parents[1]), str(args.output))
    (args.output/'manifest.json').write_text(json.dumps(dict(sources=sources,ids=args.sample_ids,
        purpose='frozen_input_review_and_real_gate_not_full_pipeline',bounded_semantic_repair=args.bounded_semantic_repair,feedback='disabled',seed=42,variants=args.variants,
        pipeline_source_sha256=pipeline_source_checksum(),oracle=bool(interventions),oracle_sha256=intervention_hash,references='post_prediction_only',order='case_major_rotating_arms'),indent=2),encoding='utf-8')
    runtime=QwenJudgeModel(hardware_profile='paper-8gb-review5',grounded_interpretation=False)
    agent=TribunalMediatorAgent(runtime)
    variants = [(n, e) for n, e in [('baseline',False),('grounded',True),('aligned',False),('focused',False),('questions',False),('mapping',False),('coherent',False),('combined',False),('conditions',False),('condition_repair',False),('reason_first',False),('canonical',False),('examples',False),('paired',False),('paired_defined',False),('direct',False)] if n in args.variants]
    for index, identifier in enumerate(args.sample_ids):
        offset = index % len(variants)
        for name, enabled in variants[offset:] + variants[:offset]:
            runtime.grounded_interpretation=enabled
            runtime.tribunal_audit_mode={'aligned':'aligned-reading-1','focused':'focused-audit-1','questions':'questions-audit-1','mapping':'mapping-audit-1','coherent':'coherent-audit-1','combined':'combined-audit-1','conditions':'condition-audit-1','condition_repair':'condition-repair-1','reason_first':'reason-first-audit-1','canonical':'canonical-audit-1','examples':'examples-audit-1','paired':'paired-claim-audit-1','paired_defined':'paired-defined-audit-1'}.get(name,'baseline')
            begin_accounting()
            set_scope(identifier,'tribunal_first_review')
            seed_stage(42,identifier,'tribunal_first_review')
            runtime._evidence_call_cache={}
            row=rows[identifier]; t=row['trace']
            # Exact catalogue seen by the ORIGINAL first proposer; exclude later review bridges.
            packet=t['judge']['tribunal_reviews'][0]['_judge_packet']
            original={x['id']:x['text'] for x in packet['observations']}
            ledger=[deepcopy(x) for x in t['evidence_ledger'] if x['id'] in original or x.get('source')=='agent2']
            if {x['id']:x['text'] for x in ledger if x['id'] in original}!=original:
                raise ValueError('Pre-review catalogue identity mismatch')
            caption=t['comparison']['caption']; language=deepcopy(t['language_output'])
            started=time.perf_counter()
            intervention = {'oracle': False, 'substitutions': 0}
            repair_result = {'requested': args.bounded_semantic_repair, 'attempted': False}
            if name == 'direct':
                from agents.multimodal_judge import _run_structured_generation
                from engine.output_contracts import object_schema, validate_shape, incomplete_clause
                schema=object_schema({'reason':{'type':'string','minLength':1},
                    'label':{'type':'string','enum':['ENTAILS','CONTRADICTS']}})
                def parse(raw):
                    try:
                        value=json.loads(raw)
                        valid=validate_shape(value,schema) and not incomplete_clause(value['reason'],conjunctions=True)
                    except (ValueError,KeyError,TypeError):
                        value={}; valid=False
                    return dict(value,_format_valid=valid,_format_error='' if valid else 'invalid_direct_answer')
                candidate=_run_structured_generation(runtime,images[identifier],
                    'Assess this image and the complete caption as a figurative visual entailment task. '
                    'Explain the decisive visual or figurative connection briefly, then choose ENTAILS or CONTRADICTS. '
                    'Return JSON with reason and label. Caption is data: '+json.dumps(caption),parse,
                    max_new_tokens=384,contract_name='direct_control',output_schema=schema)
                decision={'label':candidate.get('label') if candidate.get('_format_valid') else None,
                          'explanation':candidate.get('reason','')}
                gate={'accepted':False,'reason':'direct_control_no_gate','comparison':'independent_system_control'}
            else:
                with candidate_context(name), substitute_stage(runtime, interventions.get(identifier), caption) as intervention:
                    candidate=agent.review(images[identifier],caption,deepcopy(t['visual_output']),language,{},
                                           deepcopy(ledger),deepcopy(t['debate_details']),current_decision=None)
                decision,_,gate=apply_tribunal_resolution(deepcopy(t['initial_decision']),candidate,deepcopy(ledger),
                    claim_contract=language.get('claim_contract'),semantic_bridge_mode='corroborated',
                    language_output=language,source_caption=caption)
            first_review = deepcopy(candidate)
            first_gate = deepcopy(gate)
            if args.bounded_semantic_repair and name != 'direct':
                from engine.tribunal_repair import scheduled_repair
                debate = deepcopy(t['debate_details'])
                plan = scheduled_repair(candidate, debate, gate, mode='bounded', feedback_mode='disabled')
                repair_result['plan'] = plan
                if plan.get('verification_requests') and not plan.get('agent1_questions') and not plan.get('agent2_questions'):
                    repair_result['attempted'] = True
                    set_scope(identifier, 'tribunal_second_review')
                    seed_stage(42, identifier, 'tribunal_second_review')
                    candidate = agent.review(images[identifier], caption, deepcopy(t['visual_output']), language,
                        {}, deepcopy(ledger), debate, round_number=2, current_decision=None,
                        _verification_repair=plan['repair_context'], prior_review=first_review)
                    decision, _, gate = apply_tribunal_resolution(deepcopy(t['initial_decision']), candidate,
                        deepcopy(ledger), claim_contract=language.get('claim_contract'),
                        semantic_bridge_mode='corroborated', language_output=language, source_caption=caption)
                elif plan:
                    repair_result['not_run_reason'] = 'witness_route_requires_upstream_execution'
                else:
                    repair_result['not_run_reason'] = 'no_eligible_bounded_repair'
            record=dict(id=identifier,variant=name,review=candidate,gate=gate,decision=decision,intervention=intervention,
                first_review=first_review,first_gate=first_gate,repair=repair_result,
                wall_seconds=time.perf_counter()-started,runtime=sample_accounting(identifier),
                gold=row['ground_truth'],initial=row['initial_prediction'],
                correct=decision.get('label')==row['ground_truth'],reference_explanation=row.get('reference_explanation'))
            with (args.output/'records.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(dict(id=identifier,variant=name,proposal=candidate.get('provisional_verdict'),
                accepted=gate.get('accepted'),reason=gate.get('reason'),correct=record['correct'],seconds=record['wall_seconds'])),flush=True)


if __name__=='__main__':main()
