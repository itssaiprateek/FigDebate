"""General contract regressions; scripted responses are not accuracy evidence."""
from copy import deepcopy
import json
import unittest

from engine.evidence_review_v5 import verify, audit, binding, condition_assessments_consistent
from engine.semantic_bridge import build_semantic_bridge
from engine.simple_judge import adapt, prompt
from engine.tribunal_repair import plan_repair, scheduled_repair
from engine.runtime_accounting import begin_accounting
from tests.test_evidence_review_v4 import Runtime
from tests.test_tribunal_v5 import v5_fixture


class RemainingTribunalTests(unittest.TestCase):
    def setUp(self):
        begin_accounting()

    def test_interpretation_reaches_challenge_but_not_blind_relation(self):
        image, p, ledger, answers = v5_fixture()
        value = dict(interpreted_assertion='The participants show no concern.',
            decisive_reason=p['bridge_statement'], evidence_ids=['VF1'], relation='CONFLICT')
        review = adapt(json.dumps(value), p['source_caption'], ledger, image)
        proposal = build_semantic_bridge(review, ledger, {'source_caption': p['source_caption']})
        rt = Runtime(answers)
        proof = verify(rt, image, proposal, ledger)
        payloads = [json.loads(x.rsplit('\n', 1)[-1]) for x in rt.prompts]
        self.assertEqual(payloads[3]['interpreted_assertion'], value['interpreted_assertion'])
        self.assertEqual(payloads[3]['source_caption'], p['source_caption'])
        self.assertNotIn('interpreted_assertion', payloads[2])
        proposal['independent_verification'] = proof
        self.assertTrue(audit(proposal, ledger)['valid'])
        changed = dict(proposal, interpreted_assertion='A different claim.')
        self.assertNotEqual(binding(changed, ledger), binding(proposal, ledger))
        self.assertFalse(audit(changed, ledger)['bound_to_current_case'])

    def test_complete_diagnostic_reaches_targeted_audit(self):
        image, p, ledger, answers = v5_fixture()
        diagnostic = 'Check the same participant and panel. ' * 12 + 'DECISIVE FINAL QUALIFIER.'
        review = {'_format_valid': True, '_execution_status': 'SUCCEEDED', 'relation': 'CONFLICT',
            '_independent_verification': {'obligations': {'arguments': {
                'decision_errors': [diagnostic], 'role_scope_errors': [],
                'alternative_status': 'NONE', 'alternative': '', 'alternative_relation': 'UNRESOLVED'}}}}
        plan = plan_repair(review)
        self.assertEqual(plan['repair_context']['disputed_detail'], diagnostic)
        self.assertLessEqual(len(plan['repair_context']['question']), 320)
        self.assertEqual(plan['repair_context']['route'], 'tribunal')
        p['verification_task'] = plan['repair_context']
        rt = Runtime(answers)
        verify(rt, image, p, ledger)
        payload = json.loads(rt.prompts[-1].rsplit('\n', 1)[-1])
        self.assertEqual(payload['targeted_check']['disputed_detail'], diagnostic)

    def inconsistent_review(self):
        _, _, _, answers = v5_fixture()
        challenge = deepcopy(answers[-1])
        challenge.update(alternative='The same meeting has alarmed participants.',
            alternative_relation='CONFLICT', alternative_status='DEFEATED')
        challenge.update(_format_valid=False, _execution_status='SUCCEEDED',
            _contract_error_kind='AUDIT_INCOMPLETE', _raw_output=json.dumps(challenge))
        return {'_format_valid': True, '_execution_status': 'SUCCEEDED',
            '_independent_verification': {'obligations': {'arguments': challenge},
                'calls': [{'relation': 'CONFLICT', '_format_valid': True}]}}

    def test_complete_inconsistent_audit_can_get_one_bounded_hearing(self):
        review = self.inconsistent_review()
        plan = scheduled_repair(review, {}, {}, mode='bounded', feedback_mode='disabled')
        self.assertEqual(plan['repair_context']['failed_requirement'], 'contradictory_audit')
        original = json.loads(plan['repair_context']['disputed_detail'])['original_audit']
        self.assertEqual(original['alternative_status'], 'DEFEATED')
        self.assertFalse(scheduled_repair(review, {}, {}, mode='bounded', round_number=2))
        self.assertFalse(scheduled_repair(review, {}, {'accepted': True}, mode='bounded'))

    def test_complete_but_inconsistent_field_patch_can_be_reassessed(self):
        from engine.evidence_review_v5 import CHALLENGE_REPAIR_FIELDS, complete_audit_response
        review = self.inconsistent_review()
        call = review['_independent_verification']['obligations']['arguments']
        original = json.loads(call['_raw_output'])
        patch = {k: original[k] for k in CHALLENGE_REPAIR_FIELDS}
        call['_raw_output'] = json.dumps(patch)
        call['_field_repair_audit'] = {'original': original,
            'paths': {k: [k] for k in CHALLENGE_REPAIR_FIELDS}, 'status': 'REQUESTED_NOT_APPLIED'}
        self.assertTrue(complete_audit_response(call))
        self.assertTrue(plan_repair(review))
        call['decision_errors'] = ['A changed objection not present in the original.']
        self.assertFalse(complete_audit_response(call))
        self.assertFalse(plan_repair(review))

    def test_runtime_invalid_json_and_budget_never_get_extra_hearing(self):
        for mutation in ({'_execution_status': 'FAILED'}, {'_raw_output': '{'},
                         {'_contract_error_kind': 'STRUCTURAL'}):
            review = self.inconsistent_review()
            review['_independent_verification']['obligations']['arguments'].update(mutation)
            self.assertFalse(plan_repair(review))
        review = self.inconsistent_review()
        review['_case_budget'] = {'remaining_seconds': 1}
        self.assertFalse(plan_repair(review))

    def test_audit_repair_reuses_candidate_and_successful_prerequisites(self):
        from hashlib import sha256
        from types import SimpleNamespace
        from agents.multimodal_judge import TribunalMediatorAgent
        from engine.simple_judge import review as compact_review
        from engine.tribunal_repair import restore_proof_cache
        image, p, ledger, answers = v5_fixture()
        value = dict(interpreted_assertion='The meeting is calm.',
            decisive_reason=p['bridge_statement'], evidence_ids=['VF1'], relation='CONFLICT')
        old = adapt(json.dumps(value), p['source_caption'], ledger, image)
        old.update(_execution_status='SUCCEEDED', _judge_packet={
            'source_sha256': sha256(p['source_caption'].encode()).hexdigest()})
        contract = {'source_caption': p['source_caption']}
        answers[-1]['decision_errors'] = ['The caption describes calmness, which conflicts with the image.']
        old['_independent_verification'] = verify(Runtime(answers), image,
            build_semantic_bridge(old, ledger, contract), ledger)
        plan = plan_repair(old)
        self.assertEqual(plan['repair_context']['failed_requirement'], 'audit_objection')
        replacement = v5_fixture()[-1][-1]
        replacement['reason'] = 'The original objection supports the conflict decision rather than invalidating it.'
        rt = Runtime([replacement])
        restore_proof_cache(rt, old, ledger, {'claim_contract': contract}, p['source_caption'], plan['repair_context'])
        result = compact_review(rt, image, p['source_caption'], {}, ledger, plan['repair_context'])
        self.assertTrue(result['_proposal_reused_for_audit'])
        self.assertEqual(result['_compact_value'], value)
        self.assertEqual(len(rt.prompts), 1)
        self.assertIn('original objection', result['_independent_verification']['obligations']['arguments']['reason'])
        proposal = build_semantic_bridge(result, ledger, contract)
        self.assertTrue(audit(proposal, ledger)['valid'])
        # The new audit can still retain a genuine objection; no automatic erasure.
        result['_independent_verification']['obligations']['arguments']['decision_errors'] = ['Wrong participant.']
        self.assertFalse(audit(proposal, ledger)['valid'])
        invalid = deepcopy(plan['repair_context'])
        invalid['frozen_candidate']['source_sha256'] = 'wrong-source'
        rt = Runtime([])
        result = compact_review(rt, image, p['source_caption'], {}, ledger, invalid)
        self.assertFalse(result['_format_valid'])
        self.assertFalse(rt.prompts)

    def test_incompatible_condition_assessments_block_gate_readback(self):
        image, p, ledger, answers = v5_fixture()
        p['independent_verification'] = verify(Runtime(answers), image, p, ledger)
        decision = p['independent_verification']['calls'][0]
        decision['unestablished_conditions'] = ['calm']
        self.assertFalse(condition_assessments_consistent(decision))
        self.assertFalse(audit(p, ledger)['valid'])
        # A distinct unresolved condition need not erase positive counterevidence.
        decision['unestablished_conditions'] = ['meeting']
        self.assertTrue(condition_assessments_consistent(decision))
        decision['condition_checks'].append(dict(decision['condition_checks'][0], relation='SUPPORT'))
        self.assertFalse(condition_assessments_consistent(decision))

    def test_unresolved_condition_is_not_automatically_contradiction(self):
        image, p, ledger, answers = v5_fixture()
        answers[2].update(relation='UNRESOLVED', unestablished_conditions=['calm'])
        answers[2]['condition_checks'][0]['relation'] = 'UNRESOLVED'
        rt = Runtime(answers)
        p['independent_verification'] = verify(rt, image, p, ledger)
        self.assertEqual(p['independent_verification']['stopped_after'], 'relation_direction')
        self.assertFalse(audit(p, ledger)['valid'])
        self.assertEqual(len(rt.prompts), 3)

if __name__ == '__main__':
    unittest.main()
