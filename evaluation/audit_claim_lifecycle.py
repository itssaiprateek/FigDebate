"""Audit saved claim lifecycles without inference or modifying original runs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def compact_call(value):
    keys = ('reading', 'relation', 'reason', 'condition_checks', 'unestablished_conditions',
            'conditions', 'alignment', 'objections', '_format_valid', '_format_error',
            '_semantic_contract_valid', '_semantic_error', 'observations', 'bindings')
    return {k: value[k] for k in keys if k in value}


def audit(paths):
    rows = {}
    sources = []
    executions = 0
    for path in paths:
        raw = (path / 'records.jsonl').read_bytes()
        sources.append({'run': str(path), 'records_sha256': hashlib.sha256(raw).hexdigest()})
        for line in raw.decode('utf-8').splitlines():
            row = json.loads(line)
            executions += 1
            rows.setdefault(row['id'], row)
    counts = Counter()
    items = []
    for row in rows.values():
        trace = row['trace']
        initial, final = trace['initial_decision'], trace['final_decision']
        source = trace['comparison']['caption']
        reviews = trace['judge']['tribunal_reviews']
        first = reviews[0]
        calls = first.get('_independent_verification', {}).get('calls', [])
        relations = [v for v in calls if 'condition_checks' in v]
        counts['unique_samples'] += 1
        counts['initial_correct'] += row['initial_prediction'] == row['ground_truth']
        counts['final_correct'] += row['prediction'] == row['ground_truth']
        counts['full_caption_in_first_proposer'] += first['_judge_packet']['source_caption'] == source
        counts['initial_binary_unverified'] += initial.get('_binary_label_is_unverified') is True
        counts['initial_relation_unresolved'] += initial.get('_semantic_relation_status') == 'UNRESOLVED'
        counts['final_explicit_unresolved'] += 'Relation Status: UNRESOLVED' in final.get('explanation', '')
        counts['correct_label_with_explicit_unresolved'] += (row['prediction'] == row['ground_truth'] and
                'Relation Status: UNRESOLVED' in final.get('explanation', ''))
        if row.get('judge_confirmation_valid'):
            counts['valid_same_label_confirmations'] += 1
            counts['confirmation_retained_initial_reason'] += initial.get('explanation') == final.get('explanation')
            counts['confirmation_prior_relation_unresolved'] += initial.get('_semantic_relation_status') == 'UNRESOLVED'
        for value in relations:
            counts['first_review_relation_calls'] += 1
            counts['relation_with_exact_full_source_condition'] += any(
                x.get('caption_quote') == source for x in value['condition_checks'])
            counts['relation_semantic_contract_failed'] += value.get('_semantic_contract_valid') is False
            directional = {x.get('caption_quote') for x in value['condition_checks']
                           if x.get('relation') in {'SUPPORT', 'CONFLICT'}}
            counts['same_condition_directional_and_unresolved'] += bool(
                directional & set(value.get('unestablished_conditions', [])))
        contract = trace['language_output'].get('claim_contract') or {}
        language = {k: contract.get(k) for k in ('claim_subject', 'claim_predicate', 'asserted_property',
                    'claim_object', 'comparison_direction', 'time_or_panel_scope')}
        language['source_answers'] = [{k: x.get(k) for k in ('question', 'answer', 'unknown', 'answer_status')}
             for x in contract.get('caption_semantic_audit', {}).get('independent_source_answers', [])]
        review_items = []
        for value in reviews:
            proof = value.get('_independent_verification', {})
            review_items.append({'proposal': value.get('provisional_verdict'), 'reason': value.get('reason'),
                'reading': value.get('_compact_value', {}).get('reading'),
                'interpreted_assertion': value.get('_compact_value', {}).get('interpreted_assertion'),
                'stopped_after': proof.get('stopped_after'),
                'obligations': {k: compact_call(v) for k,v in proof.get('obligations', {}).items()},
                'relation_calls': [compact_call(v) for v in proof.get('calls', [])]})
        items.append({'id': row['id'], 'phenomenon': row['phenomenon'], 'caption': source,
            'reference': row['reference_explanation'], 'gold': row['ground_truth'],
            'initial_label': row['initial_prediction'], 'final_label': row['prediction'],
            'language': language, 'initial_explanation': initial.get('explanation'),
            'initial_relation': initial.get('_semantic_relation_status'), 'reviews': review_items,
            'gate': row.get('judge_revision_reason'), 'confirmation_valid': row.get('judge_confirmation_valid'),
            'final_explanation': final.get('explanation')})
    counts['relation_with_only_source_fragments'] = (counts['first_review_relation_calls'] -
            counts['relation_with_exact_full_source_condition'])
    return {'method': 'Saved trace audit; first occurrence retained for duplicate IDs. Exact source and field consistency checks do not establish semantic truth.',
            'executions': executions, 'sources': sources, 'counts': dict(counts), 'items': items}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-runs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.source_runs)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['counts'], indent=2))
