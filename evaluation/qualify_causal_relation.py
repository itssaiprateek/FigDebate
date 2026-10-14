"""Bounded 2x2 relation diagnostic. Oracle evidence is not pipeline performance."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from engine.output_contracts import object_schema, validate_shape


def response_schema(kind, known):
    from engine import semantic_protocol as sem
    from engine.evidence_review_v5 import DECISION, citation_schema
    if kind.startswith('existing'):
        schema = citation_schema(sem.reading_schema(DECISION), known)
        if kind == 'existing_reason_first':
            schema = object_schema({'reason': schema['properties']['reason'],
                **{k:v for k,v in schema['properties'].items() if k!='reason'}})
    else:
        fields = {'reason': {'type':'string','minLength':1},
            'evidence_ids': {'type':'array','minItems':1,'maxItems':4,
                             'items':{'type':'string','enum':sorted(known)}},
            'relation': {'type':'string','enum':['SUPPORT','CONFLICT','UNRESOLVED']}}
        if kind == 'minimal_label_first':
            fields = {'relation':fields['relation'],**{k:v for k,v in fields.items() if k!='relation'}}
        schema = object_schema(fields)
    return schema


def common_instructions():
    from engine.semantic_protocol import RULES
    return RULES + (
        'Independently assess the relation between the full source caption and the image. '
        'Use the observations as fallible evidence and check them against the image. '
        'Keep the original assertion fixed. Compare the actual state with the asserted state, '
        'including both sides of comparisons and actual outcomes versus wishes. '
        'Explain the decisive image-to-caption connection in one complete sentence and cite the '
        'deciding observation IDs. A positive incompatibility is CONFLICT; the licensed full '
        'claim being supported is SUPPORT; genuinely missing necessary evidence is UNRESOLVED. '
        'Do not replace an asserted positive claim by its implied sarcastic criticism.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--cases',type=Path,default=Path('evaluation/causal_cases_20261002.json'))
    p.add_argument('--sample-ids',nargs='+')
    p.add_argument('--schemas',nargs='+',choices=['existing','minimal','existing_reason_first','minimal_label_first'],default=['existing','minimal'])
    p.add_argument('--evidence',nargs='+',choices=['original','checked'],default=['original','checked'])
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    cases=json.loads(args.cases.read_text()); ids=args.sample_ids or list(cases)
    if len(ids)>6 or len(set(ids))!=len(ids) or not set(ids)<=set(cases) or args.repeat not in (1,2):
        raise ValueError('Bounded diagnostic needs 1-6 unique registered cases and at most two repetitions')
    from project_environment import configure
    configure()
    from evaluation.qualify_grounded_gate import image_for
    from engine.run_provenance import snapshot_source
    from engine.evidence_verification import obligation_runner, payload_valid, source_span_record_valid
    from engine.evidence_review_v5 import decision_valid
    from engine.reproducibility import seed_stage
    from engine.runtime_accounting import begin_accounting, set_scope
    from models.judge_model import QwenJudgeModel
    content=(args.source_run/'records.jsonl').read_bytes()
    rows={r['id']:r for line in content.decode().splitlines() if (r:=json.loads(line))['id'] in ids}
    if set(rows)!=set(ids):raise ValueError('Missing registered source cases')
    for key,r in rows.items():
        if (cases[key]['source_sha256']!=hashlib.sha256(r['trace']['comparison']['caption'].encode()).hexdigest()
            or cases[key]['image_sha256']!=r['image_sha256']):raise ValueError('Case identity mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    snapshot_source(Path(__file__).resolve().parents[1],args.output)
    manifest=dict(purpose='isolated_relation_diagnostic_not_pipeline_accuracy',ids=ids,schemas=args.schemas,
        evidence=args.evidence,repeat=args.repeat,seed=42,feedback='disabled',max_new_tokens=384,
        cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        source_run_sha256=hashlib.sha256(content).hexdigest(),common_instructions=common_instructions(),
        factors_note='checked evidence also changes length/organization; schema factor includes order and field burden')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    runtime=QwenJudgeModel(hardware_profile='paper-8gb-review5')
    arms=[(e,s)for e in args.evidence for s in args.schemas]
    for index,key in enumerate(ids):
        r=rows[key]; source=r['trace']['comparison']['caption']; image=image_for(key,r['image_sha256'])
        original=r['trace']['judge']['tribunal_reviews'][0]['_judge_packet']['observations']
        for repetition in range(args.repeat):
            offset=index%len(arms)
            for evidence,kind in arms[offset:]+arms[:offset]:
                observations=deepcopy(original if evidence=='original' else cases[key]['checked_observations'])
                records=[dict(x,source='agent1',type='visual_fact',grounded=True)for x in observations]
                known={x['id']for x in observations}; schema=response_schema(kind,known)
                begin_accounting();set_scope(key,'causal_relation');seed_stage(42,key,'causal_relation')
                runtime._evidence_call_cache={}
                start=time.perf_counter()
                call=obligation_runner(runtime,image,source,records,known,'5.0')('relation',common_instructions(),
                    {'source_caption':source,'observations':observations},schema,384,
                    validator=lambda v:None if v.get('evidence_ids') and len(v['evidence_ids'])==len(set(v['evidence_ids'])) else 'Cite distinct deciding observations.')
                if kind.startswith('existing'):
                    from engine.semantic_protocol import ALIGNMENT_VERSION
                    call['_semantic_alignment_protocol']=ALIGNMENT_VERSION
                    checks=call.get('condition_checks')or[]
                    if checks:call.update(decisive_caption_quote=checks[0]['caption_quote'],decisive_observation=checks[0]['image_state'])
                    consistent=decision_valid(call,known,source)
                else:
                    consistent=bool(payload_valid(call,schema)and source_span_record_valid(call,schema,source))
                row=dict(id=key,evidence=evidence,schema=kind,repetition=repetition,call=call,
                    response_contract_valid=consistent,wall_seconds=time.perf_counter()-start,
                    observation_sha256=hashlib.sha256(json.dumps(observations,sort_keys=True).encode()).hexdigest(),
                    observations=observations,gold=r['ground_truth'],reference=r.get('reference_explanation'),
                    primary_semantic_case=cases[key]['primary_semantic_case'],
                    expected_diagnostic=cases[key]['expected_for_diagnostic'])
                with(args.output/'records.jsonl').open('a',encoding='utf-8')as f:f.write(json.dumps(row)+'\n')
                print(json.dumps({k:row[k]for k in('id','evidence','schema','repetition','response_contract_valid','wall_seconds')}
                    |{'relation':call.get('relation'),'reason':call.get('reason')}),flush=True)

if __name__=='__main__':main()
