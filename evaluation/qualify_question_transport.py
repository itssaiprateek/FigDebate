"""Bounded actual visual-witness dispatch checks; no label-accuracy claim."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time


def main():
    from project_environment import configure
    configure()
    from agents.visual_grounding import VisualGroundingAgent
    from models.vision_model import Qwen3VLVisionModel
    from engine.debate import DebateEngine
    from engine.question_router import build_question_plan
    from evaluation.qualify_grounded_gate import image_for
    from engine.run_provenance import snapshot_source
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--sample-ids',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if len(args.sample_ids)>4 or len(set(args.sample_ids))!=len(args.sample_ids):
        raise ValueError('At most four unique development cases')
    rows={r['id']:r for r in map(json.loads,(args.source_run/'records.jsonl').read_text(encoding='utf-8').splitlines()) if r['id'] in args.sample_ids}
    if set(rows)!=set(args.sample_ids): raise ValueError('Missing case')
    args.output.mkdir(parents=True,exist_ok=False)
    snapshot_source(str(Path(__file__).resolve().parents[1]),str(args.output))
    (args.output/'manifest.json').write_text(json.dumps({'purpose':'actual_dispatch_and_source_binding_not_accuracy','ids':args.sample_ids,
        'seeds':[42,43,44],'order':'forward_reverse_forward','feedback':'disabled','reference_use':'none'},indent=2),encoding='utf-8')
    agent=VisualGroundingAgent(Qwen3VLVisionModel(hardware_profile='paper-8gb-review5'))
    debate=DebateEngine.__new__(DebateEngine)
    for seed in [42,43,44]:
        debate.global_seed=seed
        ids=args.sample_ids if seed!=43 else list(reversed(args.sample_ids))
        for identifier in ids:
            r=rows[identifier];t=r['trace'];plan=build_question_plan(t['comparison'])
            case=dict(key=identifier,sample_id=identifier,image=image_for(identifier,r['image_sha256']),
                caption=t['comparison']['caption'],visual_output=t['visual_output'],decision=t['initial_decision'],
                comparison=t['comparison'],tribunal_hearing=True,
                mediation_plan={'agent1_questions':[plan.agent1_question]} if plan.agent1_question else {})
            started=time.perf_counter();result=debate._run_visual_questions(agent,case)
            record={'id':identifier,'seed':seed,'issue':plan.issue_type,'result':result,'seconds':time.perf_counter()-started}
            with (args.output/'records.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps({'id':identifier,'seed':seed,'status':result.get('response_status'),'seconds':record['seconds']}),flush=True)


if __name__=='__main__':main()
