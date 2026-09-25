"""Offline explanation/origin and replacement diagnostics. Never an acceptance gate.

Only explicit reviewer annotations establish semantic correctness. Completion
heuristics flag inspection needs; they cannot certify grammatical completeness.
"""
import argparse
import hashlib
import json
from pathlib import Path

VERSION = 'qualification-report-1'


def explanation_status(text, diagnostics=()):
    from engine.output_contracts import incomplete_clause
    text = str(text or '').strip()
    diagnostics = list(diagnostics or ())
    capped = any(d.get('hit_token_limit') is True for d in diagnostics if isinstance(d, dict))
    dangling = bool(text and incomplete_clause(text, conjunctions=True))
    return {'missing': not bool(text), 'dangling_clause_signal': dangling,
            'token_limit_signal': capped, 'completion_needs_review': not text or dangling or capped,
            'semantic_correctness': 'NOT_ASSESSED'}


def stage_trace(row):
    """Support exported records and frozen-hearing replay states without mutation."""
    trace = row.get('trace') or row
    stages = []
    for name, key in (('visual_agent', 'visual_output'), ('language_agent', 'language_output'),
                      ('initial_arbiter', 'initial_decision')):
        item = trace.get(key)
        if isinstance(item, dict):
            reason = item.get('explanation') or item.get('reason') or ''
            content_keys = ('visual_description', 'visual_facts', 'visible_text') if key == 'visual_output' else (
                ('caption_proposition', 'surface_meaning', 'intended_meaning', 'negation', 'claim_modifiers')
                if key == 'language_output' else ('label', '_arbiter_assessment'))
            stages.append({'stage': name, 'text': reason,
                'recorded_content': {k: item[k] for k in content_keys if k in item},
                'execution_status': item.get('_execution_status', 'NOT_RECORDED'),
                'format_valid': item.get('_format_valid', item.get('schema_format_valid')),
                'explanation_applicable': key == 'initial_decision' or bool(reason),
                **explanation_status(reason, item.get('_generation_diagnostics', []))})
    for index, review in enumerate((trace.get('judge') or {}).get('tribunal_reviews') or []):
        proof = review.get('_independent_verification') or {}
        stages.append({'stage': f'proposal_{index+1}', 'text': review.get('reason', ''),
            'execution_status': review.get('_execution_status', 'NOT_RECORDED'),
            'format_valid': review.get('_format_valid'), 'explanation_applicable': True,
            'verification_stopped_after': proof.get('stopped_after'),
            'followup_status': review.get('_follow_up_budget_status') or review.get('_follow_up_question_status'),
            **explanation_status(review.get('reason'), review.get('_generation_diagnostics', []))})
    final = row.get('final_reason', (trace.get('decision') or {}).get('explanation', ''))
    copied = [s['stage'] for s in stages if str(final or '').strip() and s['text'] == final]
    stages.append({'stage': 'final', 'text': final, 'explanation_applicable': True,
                   'exact_copy_of': copied, **explanation_status(final)})
    return stages


def origin_report(row, semantic_reviews=()):
    stages = stage_trace(row)
    first = next((s['stage'] for s in stages if s.get('format_valid') is False or
                  (s['explanation_applicable'] and s['completion_needs_review'])), None)
    by_stage = {s['stage']: i for i, s in enumerate(stages)}
    judged = [r for r in semantic_reviews if r.get('stage') in by_stage and r.get('semantic_sound') is False]
    judged.sort(key=lambda r: by_stage[r['stage']])
    return {'id': row.get('id'), 'stages': stages,
        'first_recorded_delivery_signal': first,
        'first_reviewed_semantic_error': judged[0] if judged else None,
        'causal_origin_established': False,
        'limitation': 'First recorded error is not proof of causal origin; unreviewed stages remain unknown.'}


def correction_metrics(rows, references, reviews=None):
    """Gold/reference access is evaluator-only. Keep missing outcomes in denominators."""
    reviews = reviews or {}
    ids = [r.get('id') for r in rows]
    if any(not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Unique nonempty case IDs required')
    counts = dict(cases=len(rows), final_correct=0, missing_final=0, missing_reference=0,
        missing_initial=0,
        helpful_changes=0, harmful_changes=0, unchanged=0, incorrect_to_other_incorrect=0,
        proposals=0, reused_proposal_reviews=0, correct_proposals=0, wrong_proposals=0, unresolved_proposals=0,
        accepted_replacements=0, sound_accepted=0, unsound_accepted=0,
        sound_rejected=0, unsound_rejected=0, semantic_unreviewed=0,
        sound_confirmations=0, unsound_confirmations=0, reviewed_unresolved=0,
        correct_label_wrong_reason=0, correct_label_unsound_payload=0, final_completion_flags=0)
    for row in rows:
        trace = row.get('trace') or row
        ref = references.get(row['id']) or {}
        gold = ref.get('label') if isinstance(ref, dict) else ref
        if gold and gold not in {'ENTAILS', 'CONTRADICTS'}:
            raise ValueError('Reference labels must use ENTAILS or CONTRADICTS')
        initial = (trace.get('initial_decision') or {}).get('label')
        final = row.get('final_label', (trace.get('decision') or {}).get('label'))
        if not gold:
            counts['missing_reference'] += 1
        if final not in {'ENTAILS', 'CONTRADICTS'}:
            counts['missing_final'] += 1
        elif gold:
            counts['final_correct'] += final == gold
        if initial not in {'ENTAILS', 'CONTRADICTS'}:
            counts['missing_initial'] += 1
        elif initial == final:
            counts['unchanged'] += 1
        elif gold:
            counts['helpful_changes'] += initial != gold and final == gold
            counts['harmful_changes'] += initial == gold and final != gold
            counts['incorrect_to_other_incorrect'] += initial != gold and final != gold
        judge = trace.get('judge') or {}
        resolution = judge.get('tribunal_resolution') or {}
        counts['accepted_replacements'] += bool(resolution.get('accepted'))
        proposals = judge.get('tribunal_reviews') or []
        for index, proposal in enumerate(proposals):
            label = proposal.get('provisional_verdict')
            counts['proposals'] += 1
            counts['reused_proposal_reviews'] += bool(proposal.get('_proposal_reused_for_audit'))
            binary = label in {'ENTAILS', 'CONTRADICTS'}
            counts['unresolved_proposals'] += not binary
            if binary and gold:
                counts['correct_proposals' if label == gold else 'wrong_proposals'] += 1
            assessment = reviews.get(row['id'], {}).get(f'proposal_{index+1}')
            accepted = bool(index == len(proposals)-1 and resolution.get('accepted'))
            if assessment is None:
                counts['semantic_unreviewed'] += 1
            else:
                sound = assessment.get('semantic_sound')
                if not isinstance(sound, bool):
                    counts['semantic_unreviewed'] += 1
                    continue
                if not binary:
                    counts['reviewed_unresolved'] += 1
                elif initial not in {'ENTAILS', 'CONTRADICTS'}:
                    pass  # A missing baseline cannot establish a correction.
                elif label == initial:
                    counts['sound_confirmations' if sound else 'unsound_confirmations'] += 1
                else:
                    counts[('sound_' if sound else 'unsound_') + ('accepted' if accepted else 'rejected')] += 1
                counts['correct_label_unsound_payload'] += bool(binary and gold and label == gold and not sound)
                counts['correct_label_wrong_reason'] += bool(binary and gold and label == gold
                    and assessment.get('decisive_reason_sound') is False)
        counts['final_completion_flags'] += stage_trace(row)[-1]['completion_needs_review']
    scored = len(rows) - counts['missing_reference']
    counts['official_accuracy_referenced_cases'] = counts['final_correct'] / scored if scored else None
    counts['correct_fraction_all_cases_lower_bound'] = counts['final_correct'] / len(rows) if rows else None
    counts['reference_coverage'] = (len(rows)-counts['missing_reference']) / len(rows) if rows else None
    return counts


def calibration_eligibility(manifest):
    """A prerequisite report only; never changes production thresholds."""
    reasons = []
    if manifest.get('semantic_qualification_passed') is not True:
        reasons.append('verifier_not_semantically_qualified')
    groups = [set(manifest.get(k) or []) for k in ('development_image_groups', 'calibration_image_groups', 'test_image_groups')]
    if not all(groups):
        reasons.append('missing_split_image_groups')
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        reasons.append('image_group_leakage')
    if not manifest.get('frozen_criterion'):
        reasons.append('missing_preregistered_criterion')
    return {'eligible_to_evaluate': not reasons, 'reasons': reasons, 'threshold_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', type=Path)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--semantic-reviews', type=Path, help='Offline ID -> stage -> review map; never inferred from label agreement.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.records.read_text(encoding='utf-8')
    rows = json.loads(raw) if raw.lstrip().startswith('[') else [json.loads(line) for line in raw.splitlines() if line.strip()]
    refs = json.loads(args.references.read_text(encoding='utf-8')) if args.references else {}
    reviews = json.loads(args.semantic_reviews.read_text(encoding='utf-8')) if args.semantic_reviews else {}
    report = {'version': VERSION, 'input_sha256': hashlib.sha256(raw.encode()).hexdigest(),
        'semantic_reviews_sha256': hashlib.sha256(args.semantic_reviews.read_bytes()).hexdigest() if args.semantic_reviews else None,
        'metrics': correction_metrics(rows, refs, reviews),
        'origins': [origin_report(r, [dict(value, stage=stage) for stage, value in reviews.get(r['id'], {}).items()]) for r in rows]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
