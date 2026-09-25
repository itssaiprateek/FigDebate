"""Read-only validation of archived experimental proofs; no live generator uses this schema."""
from copy import deepcopy
from engine.output_contracts import object_schema


def schema_with_checks(schema):
    schema=deepcopy(schema)
    short={'type':'string','maxLength':240}
    fields=schema['properties']
    if 'observations' in fields:
        fields['coverage']=object_schema({'required_scope':short,'observed_scope':short,
            'missing_decisive_scope':short,'reason':short})
    elif 'condition_checks' in fields:
        fields['role_check']=object_schema({'claim_bearer':short,'image_bearer':short,
            'claim_property':short,'observed_property':short,'actual_outcome':short,'intended_outcome':short,
            'same_subject_and_scope':{'type':'boolean'},'missing_evidence_only':{'type':'boolean'}})
    elif 'decision_errors' in fields:
        fields['objection_checks']={'type':'array','maxItems':4,'items':object_schema({
            'objection':short,'evidence_ids':{'type':'array','minItems':1,'maxItems':4,'items':{'type':'string'}},
            'supported':{'type':'boolean'},'reason':short})}
    schema['required']=list(fields)
    return schema


def semantic_valid(call, known):
    if not call.get('_semantic_check_version'): return True
    if 'observations' in call:
        c=call.get('coverage',{})
        return bool(c.get('required_scope','').strip() and c.get('observed_scope','').strip()
                    and not c.get('missing_decisive_scope','').strip())
    if 'condition_checks' in call:
        c=call.get('role_check',{})
        return bool(c.get('same_subject_and_scope') and c.get('claim_bearer','').strip()
                    and c.get('image_bearer','').strip()
                    and not (call.get('relation')=='CONFLICT' and c.get('missing_evidence_only')))
    if 'decision_errors' in call:
        checks=call.get('objection_checks',[])
        if any(not set(c.get('evidence_ids',[]))<=known for c in checks): return False
        # The critic's unsupported criticism is not silently converted to a clean proof.
        # It must be explicitly repaired before admission.
        return not any(c.get('supported') for c in checks) and not call.get('decision_errors') and not call.get('role_scope_errors')
    return True
