"""Live one-field recovery from a recorded truncated mapping; no full rerun.

The first response is a saved failure injected for fault reproduction. Only its
replacement reason is generated live. This is a reliability diagnostic, not an
end-to-end semantic or accuracy comparison.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import hashlib
import time


def main():
    from project_environment import configure
    configure()
    from engine.run_provenance import snapshot_source
    from engine.evidence_verification import obligation_runner
    from engine.evidence_review_v5 import catalog, mapping_schema, mapping_error, mapping_response_valid
    from engine.mapping_completion import mapping_reason_prefix
    from engine.source_spans import SourceSpans
    from engine.reproducibility import seed_stage
    from engine.runtime_accounting import begin_accounting, set_scope
    from models.judge_model import QwenJudgeModel
    from evaluation.qualify_grounded_gate import image_for
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--sample-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    data=(args.source_run/'records.jsonl').read_bytes()
    matches=[r for l in data.decode().splitlines() if (r:=json.loads(l))['id']==args.sample_id]
    if len(matches)!=1: raise ValueError('Exactly one source case is required')
    row=matches[0]; t=row['trace']; review=t['judge']['tribunal_reviews'][0]
    proof=review['_independent_verification']; mapping=proof['obligations']['mapping']
    raw=mapping['_generation_diagnostics'][0]['raw_response']
    caption=t['comparison']['caption']; known=set(proof['selected_evidence_ids'])
    records=catalog(t['evidence_ledger']); schema=mapping_schema(known)
    prefix=mapping_reason_prefix(raw,SourceSpans(caption).schema(schema))
    if prefix is None: raise ValueError('Saved failure is not eligible for final-reason-only recovery')
    image=image_for(args.sample_id,row['image_sha256'])
    args.output.mkdir(parents=True,exist_ok=False)
    snapshot_source(Path(__file__).resolve().parents[1],args.output)
    (args.output/'manifest.json').write_text(json.dumps(dict(id=row['id'],
        purpose='fault_injected_prefix_live_field_completion_only',feedback='disabled',seed=42,
        source_sha256=hashlib.sha256(data).hexdigest(),image_sha256=row['image_sha256']),indent=2))
    runtime=QwenJudgeModel(hardware_profile='paper-8gb-review5')
    original=runtime.generate; injected=False
    def generate(image,prompt,max_new_tokens=None,json_schema=None):
        nonlocal injected
        if not injected:
            injected=True
            runtime._last_generation_diagnostics={'output_status':'TRUNCATED','saved_failure_injected':True}
            return raw,0.0
        return original(image,prompt,max_new_tokens=max_new_tokens,json_schema=json_schema)
    runtime.generate=generate
    begin_accounting();set_scope(row['id'],'mapping_field_recovery');seed_stage(42,row['id'],'mapping_field_recovery')
    started=time.perf_counter()
    call=obligation_runner(runtime,image,caption,records,known,'5.0')('mapping',
        'Match caption roles to selected observations. Preserve subject, comparison sides and scope. '
        'Matching participants does not establish a true caption. Give one concise complete mapping reason.',
        {'source_caption':caption,'observations':proof['obligations']['visual']['observations']},
        schema,320,validator=lambda v:mapping_error(v,caption,known),picture=None)
    rebuilt=json.loads(call['_raw_output']) if call.get('_format_valid') else {}
    unchanged=all(rebuilt.get(k)==prefix[k] for k in ('bindings','unmatched_roles'))
    valid=mapping_response_valid(call,known,caption)
    result=dict(id=row['id'],prefix_preserved=unchanged,valid_mapping_response=valid,
                output=call,wall_seconds=time.perf_counter()-started,semantic_gain_claimed=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='output'}),flush=True)
    if not valid or not unchanged: raise SystemExit(1)

if __name__=='__main__':main()
