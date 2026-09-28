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
    p.add_argument("--variants",nargs="+",choices=["baseline","grounded","aligned","focused"],default=["baseline","aligned","focused"])
    args=p.parse_args()
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
        purpose='paired_first_review_and_real_gate_not_full_pipeline',feedback='disabled',seed=42,variants=args.variants,
        pipeline_source_sha256=pipeline_source_checksum(),references='post_prediction_only',order='case_major_rotating_arms'),indent=2),encoding='utf-8')
    runtime=QwenJudgeModel(hardware_profile='paper-8gb-review5',grounded_interpretation=False)
    agent=TribunalMediatorAgent(runtime)
    variants = [(n, e) for n, e in [('baseline',False),('grounded',True),('aligned',False),('focused',False)] if n in args.variants]
    for index, identifier in enumerate(args.sample_ids):
        offset = index % len(variants)
        for name, enabled in variants[offset:] + variants[:offset]:
            runtime.grounded_interpretation=enabled
            runtime.tribunal_audit_mode={'aligned':'aligned-reading-1','focused':'focused-audit-1'}.get(name,'baseline')
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
            candidate=agent.review(images[identifier],caption,deepcopy(t['visual_output']),language,{},
                                   deepcopy(ledger),deepcopy(t['debate_details']),current_decision=None)
            decision,_,gate=apply_tribunal_resolution(deepcopy(t['initial_decision']),candidate,deepcopy(ledger),
                claim_contract=language.get('claim_contract'),semantic_bridge_mode='corroborated',
                language_output=language,source_caption=caption)
            record=dict(id=identifier,variant=name,review=candidate,gate=gate,decision=decision,
                wall_seconds=time.perf_counter()-started,runtime=sample_accounting(identifier),
                gold=row['ground_truth'],initial=row['initial_prediction'],
                correct=decision.get('label')==row['ground_truth'],reference_explanation=row.get('reference_explanation'))
            with (args.output/'records.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(dict(id=identifier,variant=name,proposal=candidate.get('provisional_verdict'),
                accepted=gate.get('accepted'),reason=gate.get('reason'),correct=record['correct'],seconds=record['wall_seconds'])),flush=True)


if __name__=='__main__':main()
