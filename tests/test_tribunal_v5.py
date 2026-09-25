"""Mechanism regressions. Scripted model answers are not accuracy evidence."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engine.evidence_review_v5 import verify, audit, catalog, binding
from engine.tribunal_repair import plan_repair
from engine.runtime_accounting import begin_accounting
from tests.test_evidence_review_v4 import fixture, Runtime


def v5_fixture():
    image, proposal, ledger, answers = fixture()
    answers[0].update(coverage_complete=True, missing_observation='')
    answers[0].pop('reason')
    for observation in answers[0]['observations']:
        observation.pop('observed')
        observation['image_location'] = observation.pop('attachment')
    answers[3].update(deciding_evidence_ids=['VF1'], reason='The alarmed expressions exclude a calm meeting.')
    return image, proposal, ledger, answers


class TribunalV5Tests(unittest.TestCase):
    def setUp(self):
        begin_accounting()

    def run_proof(self, mutate=None):
        image, p, ledger, answers = v5_fixture()
        if mutate:
            mutate(p, ledger, answers)
        runtime = Runtime(answers)
        p['independent_verification'] = verify(runtime, image, p, ledger)
        return runtime, p, ledger

    def test_complete_proof_both_directions_without_label_input(self):
        for relation in ('CONFLICT', 'SUPPORT'):
            def mutate(p, ledger, a):
                p['proposed_relation'] = relation
                a[2]['relation'] = relation
                a[2]['condition_checks'][0]['relation'] = relation
            rt, p, ledger = self.run_proof(mutate)
            self.assertTrue(audit(p, ledger)['valid'])
            self.assertEqual(len(rt.prompts), 4)
            for prompt in rt.prompts[:3]:
                payload = json.loads(prompt.rsplit('\n', 1)[-1])
                self.assertNotIn('proposed_relation', payload)
                self.assertNotIn('draft_argument', payload)

    def test_independent_selection_can_use_omitted_record(self):
        def mutate(p, ledger, answers):
            ledger.append(dict(ledger[0], id='VF2', text='The other panel shows concerned participants.'))
            answers[0]['observations'][0]['evidence_id'] = 'VF2'
            answers[1]['bindings'][0]['evidence_ids'] = ['VF2']
            answers[2]['evidence_ids'] = ['VF2']
            answers[3]['deciding_evidence_ids'] = ['VF2']
        rt, p, ledger = self.run_proof(mutate)
        self.assertTrue(audit(p, ledger)['valid'])
        self.assertEqual(p['independent_verification']['selected_evidence_ids'], ['VF2'])
        self.assertIn('VF2', rt.prompts[0])
        changed = deepcopy(ledger); changed[0]['text'] = 'A different panel.'
        self.assertFalse(audit(p, changed)['valid'])
        self.assertEqual(binding(p, ledger), binding(p, list(reversed(ledger))))

    def test_duplicate_observations_do_not_increase_evidence(self):
        _, _, ledger, _ = v5_fixture()
        ledger.append(dict(ledger[0], id='VF2'))
        self.assertEqual(len(catalog(ledger)), 1)
        ledger[1]['panel'] = 'second'
        self.assertEqual(len(catalog(ledger)), 2)

    def test_incomplete_coverage_cannot_certify_binary_answer(self):
        def mutate(p, ledger, a):
            a[0].update(coverage_complete=False, missing_observation='Read the text in the other panel.')
        rt, p, ledger = self.run_proof(mutate)
        self.assertEqual(len(rt.prompts), 1)
        self.assertFalse(audit(p, ledger)['valid'])
        plan = plan_repair(dict(_format_valid=True, _independent_verification=p['independent_verification']))
        self.assertEqual(plan['agent1_questions'], ['Read the text in the other panel.'])
        self.assertFalse(plan['verification_requests'])

    def test_none_alternative_requires_deciding_evidence(self):
        _, p, ledger = self.run_proof(lambda p, l, a: a[3].update(deciding_evidence_ids=[]))
        self.assertFalse(audit(p, ledger)['valid'])

    def test_unsupported_critique_does_not_automatically_change_label(self):
        _, p, ledger = self.run_proof(lambda p, l, a: a[3].update(decision_errors=['The comparison reverses the subjects.']))
        self.assertFalse(audit(p, ledger)['valid'])
        review = dict(_format_valid=True, relation='CONFLICT', _independent_verification=p['independent_verification'])
        plan = plan_repair(review)
        self.assertEqual(plan['agent1_questions'], [])
        self.assertEqual(plan['agent2_questions'], [])
        self.assertTrue(plan['verification_requests'])
        self.assertEqual(review['relation'], 'CONFLICT')
        self.assertEqual(plan_repair(review, enabled=False), {})

    def test_unknown_context_and_repeated_question_stop(self):
        review = dict(_format_valid=True, relation='UNRESOLVED', requested_follow_up='COUNTER_INTERPRETATION',
                      targeted_question='Check historical records for the original design.')
        self.assertEqual(plan_repair(review), {})
        review['targeted_question'] = 'Read the exact visible words above the person.'
        plan = plan_repair(review)
        self.assertTrue(plan['_usable'])
        debate = {'agent1_critique': {'question_answers': [{'question': review['targeted_question']}]}}
        self.assertEqual(plan_repair(review, debate), {})

    def test_caption_question_goes_only_to_language(self):
        review = dict(_format_valid=True, relation='UNRESOLVED', requested_follow_up='CAPTION_PREMISE',
                      targeted_question='What does this caption phrase mean?')
        plan = plan_repair(review)
        self.assertTrue(plan['agent2_questions'])
        self.assertFalse(plan['agent1_questions'] or plan['verification_requests'])

    def test_v5_budget_is_order_independent(self):
        from engine.case_budget import budget_estimates, observe_cost
        runtime = SimpleNamespace(hardware_profile=SimpleNamespace(tribunal_protocol='evidence-review-5.0'))
        original = budget_estimates(runtime)
        observe_cost(runtime, 'proof', 5)
        observe_cost(runtime, 'proposal', 200)
        self.assertEqual(budget_estimates(runtime), original)

    def test_failed_verification_is_not_successful_harm_rejection(self):
        from engine.review_outcome import classify_review
        review = dict(_format_valid=True, _execution_status='SUCCEEDED', provisional_verdict='CONTRADICTS',
            _independent_verification={'schema_version': '5.0', 'obligations': {'visual': {
                '_execution_status': 'FAILED', '_format_valid': False, '_execution_error_type': 'TIMEOUT'}},
                'calls': [], 'stopped_after': 'visual_grounding'})
        outcome = classify_review(review)
        self.assertEqual(outcome['terminal_outcome'], 'EXECUTION_INTERRUPTED_OR_FAILED')
        self.assertEqual(plan_repair(review), {})

    def test_changed_image_or_caption_rejects_cached_proof(self):
        _, p, ledger = self.run_proof()
        for key, value in [('source_caption', 'The meeting is not calm.'), ('image_sha256', 'wrong')]:
            changed = deepcopy(p); changed[key] = value
            self.assertFalse(audit(changed, ledger)['valid'])

    def test_suspend_detection_is_not_a_gpu_slowdown(self):
        from engine.execution_clock import ExecutionClock
        with patch('engine.execution_clock.time.perf_counter', side_effect=[100, 1000]), patch(
                'engine.execution_clock.awake_seconds', side_effect=[10, 20]):
            status = ExecutionClock().sample()
        self.assertTrue(status['host_interrupted'])
        self.assertEqual(status['awake_seconds'], 10)
        self.assertEqual(status['suspended_seconds'], 890)

    def test_bound_compact_contract_uses_new_protocol(self):
        from agents.multimodal_judge import bound_review_contract
        from engine.claim_graph import source_identity_graph
        graph = source_identity_graph('The meeting is calm.')
        packet = {'protocol': 'evidence-review-5.0', 'source_caption': 'The meeting is calm.',
                  'claim_agent': {'claim_graph': graph}, 'evidence_ledger': [{'id': 'VF1', 'type': 'visual_fact'}]}
        _, schema = bound_review_contract(graph, packet, {'VF1'})
        self.assertIn('follow_up', schema['properties'])
        self.assertNotIn('confidence', schema['properties'])

    def test_relation_check_is_routed_to_tribunal_not_silently_dropped(self):
        from engine.review_routing import route_question
        question = 'Does the observed action contradict the stated condition?'
        self.assertEqual(route_question(question, 'COUNTER_INTERPRETATION')[0], 'tribunal')
        self.assertEqual(route_question('What is the ground truth label?')[0], 'blocked')
        self.assertEqual(route_question('Do historical records contradict this claim?')[0], 'blocked')
        review = dict(_format_valid=True, relation='UNRESOLVED', targeted_question=question,
                      requested_follow_up='COUNTER_INTERPRETATION', _case_budget={'remaining_seconds': 200})
        plan = plan_repair(review)
        self.assertEqual(plan['verification_requests'], [question])
        self.assertEqual(plan['agent1_questions'], [])

    def test_compact_proposal_copies_source_and_observation_without_new_claims(self):
        from engine.claim_graph import source_identity_graph
        from engine.compact_proposal import contract
        graph = source_identity_graph('The meeting is calm.')
        packet = {'protocol': 'evidence-review-5.0', 'source_caption': graph['source_caption'],
                  'claim_agent': {'claim_graph': graph}}
        parser, schema = contract(graph, packet, {'VF1'}, set())
        value = {'node_relations': [{'claim_node_id': 'C1', 'condition_checks': [
            {'caption_span': {'start': 0, 'end': 4}, 'image_state': 'The participants appear alarmed.', 'relation': 'CONFLICT'}],
            'relation': 'CONFLICT', 'evidence_ids': ['VF1'], 'unestablished_condition': ''}],
            'alternative': '', 'decisive_reason': 'The alarmed participants conflict with calmness.',
            'follow_up': {'target': 'NONE', 'question': ''}, 'context_requests': []}
        result = parser(json.dumps(value))
        self.assertTrue(result['_format_valid'], result.get('_format_error'))
        self.assertEqual(result['_protocol'], 'evidence-review-5.0')
        self.assertEqual(result['claim_checks'][0]['condition_checks'][0]['caption_quote'], graph['source_caption'])
        self.assertEqual(result['claim_checks'][0]['observation'], 'The participants appear alarmed.')
        self.assertNotIn('observation', schema['properties']['node_relations']['items']['properties'])
        value['node_relations'][0]['condition_checks'][0]['caption_span']['end'] = 99
        invalid = parser(json.dumps(value))
        self.assertFalse(invalid['_format_valid'])
        self.assertIn('_repair_schema', invalid)
        fixed = parser(json.dumps({'replacement': {'start': 0, 'end': 4}}))
        self.assertTrue(fixed['_format_valid'])
        self.assertEqual(fixed['provisional_verdict'], result['provisional_verdict'])
        self.assertEqual(fixed['claim_checks'], result['claim_checks'])
        self.assertFalse(parser(json.dumps({'replacement': {'start': 4, 'end': 4}}))['_format_valid'])
        from agents.multimodal_judge import _run_structured_generation
        parser, schema = contract(graph, packet, {'VF1'}, set())
        runtime = Runtime([value, {'replacement': {'start': 0, 'end': 4}}])
        repaired = _run_structured_generation(runtime, None, 'Read the source.', parser,
            max_new_tokens=256, contract_name='compact_proposal', output_schema=schema)
        self.assertTrue(repaired['_format_valid'])
        self.assertTrue(repaired['_field_repair_used'])
        self.assertEqual(len(runtime.prompts), 2)

    def test_unqualified_model_behavior_is_opt_in_and_hardware_matches(self):
        from engine.runtime_profile import resolve_runtime_profile
        from engine.batch_runner import StagewiseRunner
        old = resolve_runtime_profile('paper-8gb').as_dict()
        candidate = resolve_runtime_profile('paper-8gb-review5').as_dict()
        self.assertEqual(old.pop('tribunal_protocol'), 'evidence-review-4.0')
        self.assertEqual(candidate.pop('tribunal_protocol'), 'evidence-review-5.0')
        old.pop('name'); candidate.pop('name')
        self.assertEqual(old, candidate)
        with self.assertRaisesRegex(ValueError, 'explicit'):
            StagewiseRunner(feedback_mode='integrated', judge_mode='tribunal')

    def test_actual_gate_requires_matching_v5_proof(self):
        from engine.tribunal import apply_tribunal_resolution
        from tests.test_semantic_bridge import ledger as base_ledger, contract
        from tests.test_factored_tribunal import case
        review = case()
        review['_protocol'] = 'evidence-review-5.0'
        _, _, gate = apply_tribunal_resolution({'label': 'ENTAILS', 'confidence': .8}, review,
            base_ledger(), contract(), semantic_bridge_mode='corroborated', source_caption=contract()['source_caption'])
        self.assertFalse(gate['accepted'])
        self.assertEqual(gate['reason'], 'incompatible_or_missing_verification_version')

    def test_actual_gate_accepts_verified_corrections_in_both_directions(self):
        from engine.semantic_bridge import build_semantic_bridge
        from engine.tribunal import apply_tribunal_resolution
        from engine.independent_review import image_subject_hash
        from tests.test_semantic_bridge import ledger as base_ledger, contract
        from tests.test_factored_tribunal import case
        for relation, label, initial in [('CONFLICT', 'CONTRADICTS', 'ENTAILS'), ('SUPPORT', 'ENTAILS', 'CONTRADICTS')]:
            image = fixture()[0]
            entries = base_ledger()
            observation = 'The plotted line falls.' if relation == 'CONFLICT' else 'The plotted line rises.'
            entries[0]['text'] = observation
            review = case(); review.pop('_independent_verification')
            reason = 'The falling line contradicts the claimed rise.' if relation == 'CONFLICT' else 'The rising line matches the claimed rise.'
            review.update(_protocol='evidence-review-5.0', relation=relation, provisional_verdict=label,
                          best_semantic_judgment=label, visual_premise=observation, visual_observations=[observation],
                          semantic_bridge=reason, reason=reason, counter_interpretation='', _case_image_sha256=image_subject_hash(image))
            p = build_semantic_bridge(review, entries, contract())
            answers = [
                {'observations': [{'evidence_id': 'VF001', 'supported': True, 'image_location': 'The plotted line.'}],
                 'coverage_complete': True, 'missing_observation': ''},
                {'bindings': [{'caption_quote': 'line', 'observed_entity': 'The plotted line', 'role_scope': 'Same graph', 'evidence_ids': ['VF001']}],
                 'unmatched_roles': [], 'reason': 'The same line is compared.'},
                {'relation': relation, 'evidence_ids': ['VF001'], 'condition_checks': [{'caption_quote': 'rises', 'image_state': observation, 'relation': relation}],
                 'unestablished_conditions': [], 'reason': reason},
                {'alternative': '', 'alternative_relation': 'UNRESOLVED', 'alternative_status': 'NONE', 'deciding_evidence_ids': ['VF001'],
                 'decision_errors': [], 'role_scope_errors': [], 'reason': 'The visible slope determines the direction.'}]
            review['_independent_verification'] = verify(Runtime(answers), image, p, entries)
            result, _, gate = apply_tribunal_resolution({'label': initial, 'confidence': .35}, review, entries,
                contract(), semantic_bridge_mode='corroborated', source_caption=contract()['source_caption'])
            self.assertTrue(gate['accepted'], gate)
            self.assertEqual(result['label'], label)
            self.assertEqual(gate['terminal_outcome'], 'VERIFIED_CONCLUSION')

    def test_second_hearing_does_not_receive_new_case_budget(self):
        from engine.case_budget import restore_spent_budget, case_budget, remaining_seconds
        runtime = SimpleNamespace()
        prior = {'_case_budget': {'total_seconds': 240, 'remaining_seconds': 75}}
        restore_spent_budget(runtime, 'case-resume', prior)
        with patch('engine.case_budget.time.perf_counter', return_value=100):
            with case_budget(runtime, 'case-resume', 240):
                self.assertEqual(remaining_seconds(runtime), 75)

    def test_specific_repair_reuses_only_unaffected_steps(self):
        from engine.evidence_verification import obligation_runner
        from engine.tribunal_repair import repair_reserve
        rt, p, ledger = self.run_proof()
        reserve = repair_reserve(p['independent_verification'], 'disputed_inference')
        self.assertEqual(reserve, 60)
        _, _, _, answers = v5_fixture()
        rt.answers = answers[2:]
        p['verification_task'] = {'failed_requirement': 'disputed_inference',
                                  'question': 'Does the observed state resolve the disputed comparison?'}
        p['independent_verification'] = verify(rt, fixture()[0], p, ledger)
        self.assertTrue(audit(p, ledger)['valid'])
        self.assertEqual(len(rt.prompts), 6)
        self.assertTrue(p['independent_verification']['obligations']['visual']['_cache_hit'])
        self.assertTrue(p['independent_verification']['obligations']['mapping']['_cache_hit'])

    def test_nested_field_repair_does_not_regenerate_other_fields(self):
        from engine.evidence_verification import obligation_runner
        from engine.evidence_review_v5 import SELECTION_WIRE
        image, _, ledger, answers = v5_fixture()
        original = answers[0]
        original['observations'][0]['image_location'] = 'Participants at the,'
        rt = Runtime([original, {'replacement': 'Participants at the table.'}])
        rt.hardware_profile = SimpleNamespace(judge_source_spans=True)
        result = obligation_runner(rt, image, 'The meeting is calm.', ledger, {'VF1'}, '5.0')(
            'visual', 'Check the observations.', {}, SELECTION_WIRE)
        self.assertTrue(result['_format_valid'])
        self.assertTrue(result['_field_repair_used'])
        self.assertEqual(result['observations'][0]['image_location'], 'Participants at the table.')
        self.assertEqual(result['observations'][0]['evidence_id'], original['observations'][0]['evidence_id'])
        self.assertEqual(rt.token_budgets[-1], 160)

    def test_source_archive_contains_dirty_new_modules_and_checks_resume(self):
        import tempfile
        from pathlib import Path
        from engine.run_provenance import snapshot_source
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'code'; run = Path(directory) / 'run'
            (root / 'engine').mkdir(parents=True); run.mkdir()
            module = root / 'engine' / 'new.py'; module.write_text('value = 1\n')
            first = snapshot_source(root, run)
            self.assertEqual(snapshot_source(root, run), first)
            module.write_text('value = 2\n')
            with self.assertRaisesRegex(ValueError, 'differs'):
                snapshot_source(root, run)


if __name__ == '__main__':
    unittest.main()
