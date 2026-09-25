"""Exercise production boundaries with scripted responses, not accuracy claims."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

from agents.multimodal_judge import TribunalMediatorAgent
from engine.simple_judge import adapt, prompt, schema
from engine.evidence_review_v5 import (verify, audit, catalog, CHALLENGE,
                                     CHALLENGE_REPAIR_FIELDS)
from engine.semantic_bridge import build_semantic_bridge
from engine.tribunal import apply_tribunal_resolution
from engine.review_outcome import stage_outcomes
from engine.runtime_accounting import begin_accounting
from tests.test_evidence_review_v4 import Runtime
from tests.test_tribunal_v5 import v5_fixture


class SimpleJudgeTests(unittest.TestCase):
    def setUp(self):
        begin_accounting()

    def fixture(self):
        image, proposal, ledger, answers = v5_fixture()
        value = {'interpreted_assertion': proposal['source_caption'],
                 'decisive_reason': proposal['bridge_statement'],
                 'evidence_ids': ['VF1'], 'relation': 'CONFLICT'}
        return image, proposal, ledger, answers, value

    def test_production_entry_point_and_real_gate_both_directions(self):
        for relation, initial, label in [('CONFLICT', 'ENTAILS', 'CONTRADICTS'),
                                         ('SUPPORT', 'CONTRADICTS', 'ENTAILS')]:
            image, p, ledger, answers, value = self.fixture()
            value['relation'] = answers[2]['relation'] = relation
            answers[2]['condition_checks'][0]['relation'] = relation
            runtime = Runtime([value] + answers)
            runtime.hardware_profile = SimpleNamespace(tribunal_protocol='evidence-review-5.0')
            from tests.test_semantic_bridge import contract
            language = {'claim_contract': dict(contract(), source_caption=p['source_caption'],
                                               caption_proposition=p['source_caption'])}
            ledger.append({'id': 'LC1', 'source': 'agent2', 'type': 'caption_proposition',
                           'text': p['source_caption'], 'grounded': False})
            review = TribunalMediatorAgent(runtime).review(image, p['source_caption'], {}, language,
                {}, ledger, {}, current_decision={'label': initial, 'secret': 'HIDDEN_VERDICT'})
            final, _, gate = apply_tribunal_resolution({'label': initial, 'confidence': .3}, review,
                ledger, language['claim_contract'], semantic_bridge_mode='corroborated',
                source_caption=p['source_caption'])
            self.assertEqual(review['_proposer_version'], 'compact-judge-1')
            self.assertEqual(len(runtime.prompts), 5)
            self.assertNotIn('HIDDEN_VERDICT', runtime.prompts[0])
            self.assertTrue(gate['accepted'], (gate['reason'], gate.get('verification_failures')))
            self.assertEqual(final['label'], label)

    def test_no_unknown_duplicate_or_incomplete_proposals(self):
        image, p, ledger, _, value = self.fixture()
        for mutation in ({'evidence_ids': ['MISSING']}, {'evidence_ids': ['VF1', 'VF1']},
                         {'decisive_reason': 'It differs because'},
                         {'interpreted_assertion': 'x' * 480}):
            review = adapt(json.dumps(dict(value, **mutation)), p['source_caption'], catalog(ledger), image)
            self.assertFalse(review['_compact_eligible'])
        for raw in ('{', 'null', '[]'):
            self.assertFalse(adapt(raw, p['source_caption'], catalog(ledger), image)['_format_valid'])

    def test_whole_context_is_preserved_without_hidden_labels(self):
        source = 'Qualifier and scope. ' * 100
        observation = 'Complete observation. ' * 100
        context = {'source_caption': source, 'observations': [{'id': 'VF1', 'text': observation}]}
        result = json.loads(prompt(context).rsplit('\n', 1)[-1])
        self.assertEqual(result, context)

    def test_context_overflow_fails_explicitly_before_generation(self):
        image, p, ledger, _, _ = self.fixture()
        rt = Runtime([])
        rt.hardware_profile = SimpleNamespace(tribunal_protocol='evidence-review-5.0')
        rt.count_text_tokens = lambda _: 200
        rt.review_text_budget = lambda *args: 100
        result = TribunalMediatorAgent(rt).review(image, p['source_caption'], {}, {}, {}, ledger, {})
        self.assertEqual(result['_execution_error_type'], 'CONTEXT_BUDGET')
        self.assertFalse(rt.prompts)

    def test_restored_normalized_cache_preserves_locations_and_validates_raw(self):
        from engine.tribunal_repair import restore_proof_cache
        image, p, ledger, answers, value = self.fixture()
        review = adapt(json.dumps(value), p['source_caption'], catalog(ledger), image)
        review['_execution_status'] = 'SUCCEEDED'
        contract = {'source_caption': p['source_caption']}
        proposal = build_semantic_bridge(review, ledger, contract)
        review['_independent_verification'] = verify(Runtime(answers), image, proposal, ledger)
        rt = Runtime([])
        restore_proof_cache(rt, review, ledger, {'claim_contract': contract}, p['source_caption'],
                            {'failed_requirement': 'contradictory_audit'})
        proof = verify(rt, image, proposal, ledger)
        self.assertFalse(rt.prompts)
        self.assertTrue(proof['obligations']['visual']['_cache_hit'])
        self.assertEqual(proof['obligations']['visual']['observations'][0]['attachment'],
                         answers[0]['observations'][0]['image_location'])
        proposal['independent_verification'] = proof
        self.assertTrue(audit(proposal, ledger)['valid'])
        proof['obligations']['visual']['observations'][0]['attachment'] = 'Forged location'
        self.assertFalse(audit(proposal, ledger)['valid'])

    def challenge_run(self, first, replacement=None):
        image, p, ledger, answers, _ = self.fixture()
        answers[3].update(first)
        if replacement is not None:
            answers.append(replacement)
        runtime = Runtime(answers)
        proof = verify(runtime, image, p, ledger)
        p['independent_verification'] = proof
        return runtime, p, ledger, proof['obligations']['arguments']

    def clarification(self, **updates):
        value = v5_fixture()[-1][-1]
        return {k: updates.get(k, value[k]) for k in CHALLENGE_REPAIR_FIELDS}

    def test_placeholder_clarified_once_before_gate(self):
        first = dict(alternative='UNRESOLVED', alternative_status='UNRESOLVED')
        rt, p, ledger, call = self.challenge_run(first, self.clarification())
        self.assertEqual(len(rt.prompts), 5)
        self.assertTrue(call['_field_repair_used'])
        self.assertEqual(call['_field_repair_audit']['original']['alternative'], 'UNRESOLVED')
        self.assertTrue(audit(p, ledger)['valid'])

    def test_placeholder_never_silently_becomes_none(self):
        first = dict(alternative='UNRESOLVED', alternative_status='UNRESOLVED')
        rt, p, ledger, call = self.challenge_run(first, self.clarification(**first))
        self.assertEqual(len(rt.prompts), 5)
        self.assertFalse(audit(p, ledger)['valid'])
        self.assertEqual(call['_contract_error_kind'], 'AUDIT_INCOMPLETE')
        row = stage_outcomes({'_independent_verification': p['independent_verification']})['stage_outcomes'][-1]
        self.assertEqual(row['status'], 'INCOMPLETE_AUDIT')
        self.assertEqual(row['validation'], 'VALID')  # JSON is complete; the audit is not coherent.
        self.assertEqual(row['semantic'], 'NOT_ASSESSED')
        from engine.review_outcome import classify_review
        outcome = classify_review(dict(_format_valid=True, _execution_status='SUCCEEDED',
            relation='CONFLICT', provisional_verdict='CONTRADICTS',
            _independent_verification=p['independent_verification']))
        self.assertEqual(outcome['verification_status'], 'INCOMPLETE_VERIFICATION_AUDIT')
        self.assertEqual(outcome['terminal_output_policy'], 'PRESERVE_INITIAL_WITH_INCOMPLETE_AUDIT')

    def test_clarification_cannot_erase_substantive_objections(self):
        first = dict(alternative='UNRESOLVED', alternative_status='UNRESOLVED',
                     decision_errors=['The observation belongs to a different participant.'])
        _, p, ledger, call = self.challenge_run(first, self.clarification())
        self.assertEqual(call['decision_errors'], first['decision_errors'])
        self.assertFalse(audit(p, ledger)['valid'])

    def test_attempt_to_rewrite_other_fields_is_rejected(self):
        first = dict(alternative='UNRESOLVED', alternative_status='UNRESOLVED',
                     decision_errors=['The observed subject is different.'])
        _, p, ledger, call = self.challenge_run(first, dict(self.clarification(), decision_errors=[]))
        self.assertFalse(call['_format_valid'])
        self.assertFalse(audit(p, ledger)['valid'])

    def test_material_uncertainty_gets_no_contract_retry_or_acceptance(self):
        rt, p, ledger, call = self.challenge_run(dict(alternative='The participants may be reacting to a staged event.',
            alternative_relation='SUPPORT', alternative_status='UNRESOLVED'))
        self.assertEqual(len(rt.prompts), 4)
        self.assertTrue(call['_format_valid'])
        self.assertFalse(audit(p, ledger)['valid'])

    def test_long_complete_error_survives_and_blocks_gate(self):
        from engine.output_contracts import validate_shape
        error = ('The comparison mixes participants and does not establish the claimed property. ' * 4).strip()
        self.assertGreater(len(error), 240)
        _, p, ledger, call = self.challenge_run(dict(decision_errors=[error]))
        self.assertEqual(call['decision_errors'], [error])
        self.assertTrue(call['_format_valid'])
        self.assertFalse(audit(p, ledger)['valid'])
        value = self.fixture()[3][-1]
        value['reason'] = 'Evidence establishes the same subject and scope. ' * 12
        self.assertGreater(len(value['reason']), 480)
        self.assertTrue(validate_shape(value, CHALLENGE))

    def test_runtime_failure_is_not_an_abstention(self):
        image, p, ledger, _, _ = self.fixture()
        rt = Runtime([])
        rt.hardware_profile = SimpleNamespace(tribunal_protocol='evidence-review-5.0')
        def fail(*args, **kwargs):
            raise TimeoutError('timed out')
        rt.generate = fail
        result = TribunalMediatorAgent(rt).review(image, p['source_caption'], {}, {}, {}, ledger, {})
        self.assertEqual(result['_execution_status'], 'FAILED')
        self.assertFalse(result['_format_valid'])
        self.assertNotIn('_independent_verification', result)

    def test_optional_process_audit_keeps_case_binding(self):
        from engine import tribunal_process as process
        image, p, ledger, answers, value = self.fixture()
        answers[3].update(assessed_interpretation='The meeting participants are calm.',
            process_checks={key: {'status': 'PASS', 'evidence_ids': ['VF1'],
                'reason': 'The same meeting participants show alarm.'} for key in process.CHECKS})
        rt = Runtime([value] + answers)
        rt.hardware_profile = SimpleNamespace(tribunal_protocol='evidence-review-5.0')
        rt.tribunal_audit_mode = process.VERSION
        result = TribunalMediatorAgent(rt).review(image, p['source_caption'], {}, {}, {}, ledger, {})
        self.assertEqual(result['_process_audit_version'], process.VERSION)
        proposal = build_semantic_bridge(result, ledger, {'source_caption': p['source_caption']})
        self.assertTrue(audit(proposal, ledger)['valid'])


if __name__ == '__main__':
    unittest.main()
