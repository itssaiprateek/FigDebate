"""Read saved gold-free reviews through the gate; never regenerate or edit source runs."""
import json
from copy import deepcopy
from pathlib import Path
from engine.tribunal import apply_tribunal_resolution
from engine.runtime_accounting import begin_accounting

def replay(source_runs):
    begin_accounting(); rows={}
    for name in source_runs:
        for line in (Path(name)/'records.jsonl').read_text(encoding='utf-8').splitlines():
            r=json.loads(line);rows.setdefault(r['id'],r)
    results=[]
    for sid,r in rows.items():
        t=r['trace']; review=t['judge']['tribunal_reviews'][-1]
        known={x['id'] for x in review['_judge_packet']['observations']}
        ledger=[deepcopy(x) for x in t['evidence_ledger'] if x['id'] in known or x.get('source')=='agent2']
        lang=t['language_output'];d=t['debate_details']
        out,_,gate=apply_tribunal_resolution(deepcopy(t['initial_decision']),deepcopy(review),ledger,lang.get('claim_contract'),
            agent2_requirements_valid=d.get('agent2_requirements_valid',True),agent1_critique=d.get('agent1_critique',{}),
            agent2_critique=d.get('agent2_critique',{}),semantic_bridge_mode='corroborated',language_output=lang,
            source_caption=t['comparison']['caption'])
        results.append(dict(id=sid,label_unchanged=out['label']==r['prediction'],
            confirmation_unchanged=bool(gate.get('confirmation_valid'))==bool(r.get('judge_confirmation_valid')),
            explanation_changed=out.get('explanation')!=t['final_decision'].get('explanation'),
            old_explanation=t['final_decision'].get('explanation'),new_explanation=out.get('explanation'),
            original_label=r['prediction'],reference_label=r['ground_truth'],reference=r['reference_explanation'],
            caption=t['comparison']['caption'],gate=gate.get('reason')))
    return results

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-runs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=replay(a.source_runs)
    result=dict(purpose='deterministic_saved_review_replay_no_new_inference',samples=len(rows),
        label_changes=sum(not r['label_unchanged'] for r in rows),confirmation_changes=sum(not r['confirmation_unchanged'] for r in rows),
        explanation_changes=sum(r['explanation_changed']for r in rows),rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items()if k!='rows'}))
