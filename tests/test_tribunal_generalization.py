"""Dataset-independent contract and adversarial regression checks, not accuracy claims."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace
import unittest

from engine.evidence_review_v5 import verify, audit, challenge_valid
from engine.semantic_bridge import build_semantic_bridge
from engine.simple_judge import adapt, review as compact_review
from engine.tribunal_repair import plan_repair, restore_proof_cache
from engine.runtime_accounting import begin_accounting
from tests.test_evidence_review_v4 import Runtime
from tests.test_tribunal_v5 import v5_fixture


class GeneralTribunalTests(unittest.TestCase):
    def setUp(self):
        begin_accounting()

    def test_none_spelling_does_not_erase_uncertainty_or_objections(self):
        image, proposal, ledger, answers = v5_fixture()
        for text in ('', 'None', 'NONE', ' none. '):
            with self.subTest(text=text):
                answers[-1]['alternative'] = text
                rt = Runtime(deepcopy(answers))
                proposal['independent_verification'] = verify(rt, image, proposal, ledger)
                self.assertTrue(audit(proposal, ledger)['valid'])
                self.assertEqual(len(rt.prompts), 4)
                call = proposal['independent_verification']['obligations']['arguments']
                self.assertEqual(json.loads(call['_raw_output'])['alternative'], text)
                for mutation in (
                    {'alternative': 'unknown'}, {'alternative': 'unresolved'}, {'alternative': '...'},
                    {'alternative': 'No one is alarmed.'}, {'alternative_relation': 'SUPPORT'},
                    {'alternative_status': 'UNRESOLVED'}, {'deciding_evidence_ids': []},
                    {'decision_errors': ['The participant is different.']},
                    {'role_scope_errors': ['The time is different.']},
                ):
                    self.assertFalse(challenge_valid(dict(call, **mutation), {'VF1'}, 'CONFLICT'))

    def test_complete_long_text_and_long_source_spans_survive_every_stage(self):
        from engine.source_spans import SourceSpans
        image, proposal, ledger, answers = v5_fixture()
        source = 'The meeting with ' + 'many concerned participants and ' * 45 + 'a calm atmosphere.'
        proposal.update(source_caption=source, caption_premise=source)
        answers[1]['bindings'][0]['caption_quote'] = source
        answers[2]['condition_checks'][0]['caption_quote'] = source
        long_reason = 'The participants display concern. ' * 18
        answers[1]['reason'] = answers[2]['reason'] = long_reason
        answers[2]['condition_checks'][0]['image_state'] = long_reason
        span = {'start': 0, 'end': len(SourceSpans(source).tokens)}
        answers[1]['bindings'][0].pop('caption_quote')
        answers[1]['bindings'][0]['caption_span'] = span
        answers[2]['condition_checks'][0].pop('caption_quote')
        answers[2]['condition_checks'][0]['caption_span'] = span
        rt = Runtime(answers)
        rt.hardware_profile = SimpleNamespace(judge_source_spans=True)
        proposal['independent_verification'] = verify(rt, image, proposal, ledger)
        self.assertTrue(audit(proposal, ledger)['valid'])
        self.assertEqual(len(rt.prompts), 4)
        self.assertEqual(proposal['independent_verification']['calls'][0]['reason'], long_reason)
        bridge = build_semantic_bridge({'caption_premise': source, 'semantic_bridge': long_reason * 3},
                                      ledger, {'source_caption': source})
        self.assertEqual(bridge['caption_premise'], source)
        self.assertEqual(bridge['bridge_statement'], (long_reason * 3).strip())

    def test_unfinished_or_truncated_reason_still_cannot_certify(self):
        image, proposal, ledger, answers = v5_fixture()
        for text in ('The observed state differs because', 'The participants look calm,'):
            changed = deepcopy(answers)
            changed[2]['reason'] = text
            rt = Runtime(changed[:3] + [deepcopy(changed[2])])
            proposal['independent_verification'] = verify(rt, image, proposal, ledger)
            self.assertFalse(audit(proposal, ledger)['valid'])

    def test_disagreement_repair_cannot_copy_the_verifiers_new_verdict(self):
        image, proposal, ledger, answers = v5_fixture()
        value = dict(interpreted_assertion='The meeting is calm.', evidence_ids=['VF1'],
                     decisive_reason=proposal['bridge_statement'], relation='CONFLICT')
        old = adapt(json.dumps(value), proposal['source_caption'], ledger, image)
        old.update(_execution_status='SUCCEEDED', _judge_packet={
            'source_sha256': sha256(proposal['source_caption'].encode()).hexdigest()})
        contract = {'source_caption': proposal['source_caption']}
        answers[2]['relation'] = answers[2]['condition_checks'][0]['relation'] = 'SUPPORT'
        old['_independent_verification'] = verify(Runtime(deepcopy(answers)), image,
            build_semantic_bridge(old, ledger, contract), ledger)
        plan = plan_repair(old)
        self.assertEqual(plan['repair_context']['failed_requirement'], 'relation_disagreement')
        self.assertEqual(plan['repair_context']['frozen_candidate']['value'], value)
        rt = Runtime(deepcopy(answers[2:]))
        restore_proof_cache(rt, old, ledger, {'claim_contract': contract}, proposal['source_caption'], plan['repair_context'])
        result = compact_review(rt, image, proposal['source_caption'], {}, ledger, plan['repair_context'])
        self.assertEqual(result['_compact_value'], value)
        self.assertEqual(result['provisional_verdict'], 'CONTRADICTS')
        self.assertEqual(len(rt.prompts), 2)
        self.assertFalse(audit(build_semantic_bridge(result, ledger, contract), ledger)['valid'])

    def test_case_identity_citations_and_order_are_not_shortcuts_to_acceptance(self):
        image, proposal, ledger, answers = v5_fixture()
        # Rename identities and reorder equivalent input records: semantics unchanged.
        for identity in ('E-A', 'record_982', 'observation-Z'):
            renamed = json.loads(json.dumps([proposal, ledger, answers]).replace('VF1', identity))
            p, records, responses = renamed
            p['independent_verification'] = verify(Runtime(responses), image, p, records)
            self.assertTrue(audit(p, list(reversed(records)))['valid'])
            wrong = deepcopy(records)
            wrong[0]['text'] = 'An unrelated observation.'
            self.assertFalse(audit(p, wrong)['valid'])
            changed = deepcopy(p)
            changed['source_caption'] += ' Not.'
            self.assertFalse(audit(changed, records)['valid'])

    def test_audit_and_relation_must_match_their_recorded_raw_output(self):
        image, proposal, ledger, answers = v5_fixture()
        proposal['independent_verification'] = verify(Runtime(answers), image, proposal, ledger)
        for stage, field, replacement in (
            ('arguments', 'reason', 'A replacement explanation never generated by the model.'),
            ('arguments', 'alternative', 'None'),
            ('relation', 'reason', 'Another replacement reason.'),
        ):
            changed = deepcopy(proposal)
            proof = changed['independent_verification']
            target = proof['calls'][0] if stage == 'relation' else proof['obligations'][stage]
            target[field] = replacement
            self.assertFalse(audit(changed, ledger)['valid'], (stage, field))

    def test_auxiliary_family_cannot_override_complete_directional_proof(self):
        from engine.semantic_bridge import BRIDGE_FAMILIES
        from engine.semantic_bridge_verifier import verify_semantic_bridge
        from tests.test_semantic_bridge import contract
        image, original, ledger, answers = v5_fixture()
        source = original['source_caption']
        ledger.append(dict(id='LC1', source='agent2', type='caption_proposition', text=source, grounded=False))
        claim = dict(contract(), source_caption=source, caption_proposition=source)
        value = dict(interpreted_assertion=source, decisive_reason=original['bridge_statement'],
                     evidence_ids=['VF1'], relation='CONFLICT')
        p = build_semantic_bridge(adapt(json.dumps(value), source, ledger, image), ledger, claim)
        p['independent_verification'] = verify(Runtime(answers), image, p, ledger)
        for family in BRIDGE_FAMILIES:
            with self.subTest(family=family):
                changed = dict(p, bridge_type=family)
                self.assertTrue(verify_semantic_bridge(changed, ledger, claim)[1]['corroborated'])
                broken = deepcopy(changed)
                broken['independent_verification']['calls'][0]['relation'] = 'UNRESOLVED'
                self.assertFalse(verify_semantic_bridge(broken, ledger, claim)[1]['corroborated'])

    def test_completed_semantic_contradiction_is_not_a_format_retry(self):
        from engine.review_outcome import stage_outcomes, classify_review
        image, proposal, ledger, answers = v5_fixture()
        answers[2]['relation'] = answers[2]['condition_checks'][0]['relation'] = 'SUPPORT'
        answers[2]['unestablished_conditions'] = ['calm']
        rt = Runtime(answers[:3])
        proposal['independent_verification'] = verify(rt, image, proposal, ledger)
        proof = proposal['independent_verification']
        call = proof['calls'][0]
        self.assertEqual(len(rt.prompts), 3)
        self.assertTrue(call['_format_valid'])
        self.assertFalse(call['_semantic_contract_valid'])
        self.assertFalse(call['_format_retry_used'])
        self.assertNotIn(call['_cache_key'], rt._evidence_call_cache)
        self.assertEqual(call['unestablished_conditions'], ['calm'])
        self.assertFalse(audit(proposal, ledger)['valid'])
        review = dict(_format_valid=True, _execution_status='SUCCEEDED', relation='SUPPORT',
                      provisional_verdict='ENTAILS', _independent_verification=proof)
        row = next(x for x in stage_outcomes(review)['stage_outcomes'] if x['stage'] == 'relation')
        self.assertEqual((row['execution'], row['validation'], row['semantic']),
                         ('COMPLETED', 'VALID', 'CONTRADICTORY'))
        self.assertNotEqual(classify_review(review)['execution_status'], 'FAILED')

    def test_slow_case_reserves_complete_remaining_proof_not_one_retry(self):
        from engine.tribunal_repair import repair_reserve
        proof = {'obligations': {
            'visual': {'verified': True, '_cache_key': 'visual-key', '_execution_status': 'SUCCEEDED', '_generation_seconds': 42},
            'mapping': {'verified': True, '_cache_key': 'mapping-key', '_execution_status': 'SUCCEEDED', '_generation_seconds': 26}},
            'calls': [{'_execution_status': 'SUCCEEDED', '_generation_seconds': 50,
                       'relation': 'SUPPORT', 'reason': 'The reported relation remains inconsistent.'}]}
        self.assertEqual(repair_reserve(proof, 'relation_disagreement'), 100)
        review = {'_format_valid': True, 'relation': 'CONFLICT', '_compact_eligible': True,
                  '_compact_value': {}, '_independent_verification': proof,
                  '_case_budget': {'remaining_seconds': 84}}
        self.assertFalse(plan_repair(review))
        self.assertEqual(review['_follow_up_budget_requirement_seconds'], 100)
        faster = deepcopy(proof)
        for call in list(faster['obligations'].values()) + faster['calls']:
            call['_generation_seconds'] = 10
        self.assertEqual(repair_reserve(faster, 'relation_disagreement'), 60)
        self.assertEqual(repair_reserve(proof, 'relation_disagreement'), 100)


if __name__ == '__main__':
    unittest.main()
