"""Operational regressions; these do not establish model semantic quality."""
import json
import unittest
from agents.multimodal_judge import _run_structured_generation
from engine.independent_review import CHECK, ARGUMENTS, _parser, obligation_token_budget
from engine.review_outcome import classify_review, hearing_accounting, review_timing


class ReviewOperationalTests(unittest.TestCase):
    def run_fixture(self, responses):
        class Runtime:
            _last_generation_diagnostics = {}
            limits = []
            def generate(self, image, prompt, max_new_tokens=None, json_schema=None):
                self.limits.append(max_new_tokens)
                text, hit = responses[len(self.limits)-1]
                self._last_generation_diagnostics = {'hit_token_limit': hit,
                    'input_tokens': 100, 'total_token_budget': 2048}
                return text, .01
        runtime = Runtime()
        result = _run_structured_generation(runtime, None, 'fixture', _parser(CHECK),
            max_new_tokens=256, contract_name='factored_visual', output_schema=CHECK)
        return runtime, result

    def test_incomplete_cap_uses_existing_single_larger_retry(self):
        rt, result = self.run_fixture([('{', True), ('{"verified":true,"reason":"Visible."}', False)])
        self.assertEqual(rt.limits, [256, 512])
        self.assertEqual(result['_output_status'], 'VALID')
        self.assertTrue(result['_format_retry_success'])

    def test_valid_json_at_cap_is_not_truncation(self):
        rt, result = self.run_fixture([('{"verified":true,"reason":"Visible."}', True)])
        self.assertEqual(rt.limits, [256])
        self.assertEqual(result['_output_status'], 'VALID')

    def test_truncation_and_malformed_are_separate(self):
        for hit, expected in [(True, 'TRUNCATED'), (False, 'INVALID')]:
            rt, result = self.run_fixture([('{', hit), ('{', hit)])
            self.assertEqual(len(rt.limits), 2)
            self.assertEqual(classify_review(result)['schema_status'], expected)
            self.assertFalse(classify_review(result)['semantic_abstained'])

    def test_different_schema_budgets_and_reason_bound(self):
        self.assertGreater(obligation_token_budget(ARGUMENTS), obligation_token_budget(CHECK))
        self.assertFalse(_parser(CHECK)(json.dumps({'verified':True,'reason':'x'*721}))['_format_valid'])

    def test_planned_is_not_executed(self):
        counts = hearing_accounting({'verification_followup_plan':{'repair_reasons':['VISUAL_PREMISE']}})
        self.assertTrue(counts['tribunal_repair_planned'])
        self.assertEqual(counts['tribunal_repair_hearing_count'], 0)

    def test_actual_reviews_not_scheduling_intentions(self):
        counts = hearing_accounting({'tribunal_session':{'hearing_transitions':[
            {'before':'a','after':'b','reconsidered':True,'repair_reasons':['MAPPING']} ]},
            'tribunal_reviews':[{}]})
        self.assertEqual(counts['tribunal_content_changed_count'], 1)
        self.assertEqual(counts['tribunal_rereview_count'], 0)
        self.assertTrue(counts['tribunal_repair_followup_attempted'])

    def test_nested_timing_does_not_double_count(self):
        prior = {'_generation_seconds':10, '_independent_verification':{'_generation_seconds':4}}
        current = {'_generation_seconds':30, '_independent_verification':{'_generation_seconds':8},
                   '_verification_repair_history':[prior]}
        result = review_timing(current)
        self.assertEqual(result['verification_seconds'], 12)
        self.assertEqual(result['proposal_and_format_retry_seconds'], 18)

    def test_timeout_is_not_semantic_abstention(self):
        result = classify_review({'_execution_status':'FAILED','_execution_error_type':'TIMEOUT',
                                  '_format_valid':False,'provisional_verdict':'ABSTAIN'})
        self.assertFalse(result['semantic_abstained'])
        self.assertEqual(result['terminal_output_policy'], 'PRESERVE_INITIAL_WITH_EXPLICIT_REVIEW_FAILURE')

    def test_relation_calls_alone_are_not_complete_verification(self):
        call = {'_execution_status':'SUCCEEDED', '_format_valid':True}
        result = classify_review({'_independent_verification':{'calls':[call, call]}})
        self.assertEqual(result['verification_status'], 'INCOMPLETE_VERIFICATION')

    def test_grounding_and_direction_defects_request_targeted_hearing(self):
        from engine.tribunal import repair_followup_plan
        from tests.test_factored_tribunal import case
        review = case()
        args = review['_independent_verification']['obligations']['arguments']
        args.update(bridge_grounded=False, bridge_relation='SUPPORT')
        plan = repair_followup_plan(review)
        self.assertEqual(plan['repair_reasons'], ['BRIDGE_GROUNDING', 'BRIDGE_DIRECTION'])
        self.assertTrue(plan['agent1_questions'])
        self.assertTrue(plan['agent2_questions'])

    def test_root_failure_count_does_not_count_dependent_checks(self):
        from tests.test_factored_tribunal import case
        from tests.test_semantic_bridge import ledger, contract
        from engine.semantic_bridge import build_semantic_bridge
        from engine.independent_review import audit_independent_record
        review = case()
        proof = review['_independent_verification']
        proof['obligations']['mapping']['verified'] = False
        proof['obligations']['arguments']['bridge_relation'] = 'SUPPORT'
        audit = audit_independent_record(build_semantic_bridge(review, ledger(), contract()), ledger())
        self.assertEqual(audit['root_failures'], ['entity_scope_mapping', 'bridge_direction'])
        self.assertEqual(len(audit['failed_obligations']), 3)

    def test_counter_prompt_does_not_require_fabricated_opposition(self):
        from agents.multimodal_judge import build_tribunal_review_prompt
        prompt = build_tribunal_review_prompt({}, 1)
        self.assertIn('do not hide it', prompt)
        self.assertIn('empty counter_interpretation rather than inventing', prompt)

    def test_qualified_interpretation_remains_available(self):
        from arbiter.arbiter import Arbiter
        summary = Arbiter._summarize_language({'intended_meaning':'Qualified interpretation',
                                              'claim_contract':{'semantic_qualified':True}})
        self.assertIn('Qualified interpretation', summary)

    def test_verification_timeout_is_included_without_adding_to_phase_total(self):
        call = {'_generation_seconds':120, '_generation_diagnostics':[
            {'termination_reason':'TIMEOUT', 'elapsed_seconds':120}]}
        review = {'_generation_seconds':140, '_independent_verification':{
            '_generation_seconds':120, 'obligations':{'mapping':call}}}
        timing = review_timing(review)
        self.assertEqual(timing['timeout_elapsed_seconds'], 120)
        self.assertEqual(timing['proposal_and_format_retry_seconds'], 20)


if __name__ == '__main__':
    unittest.main()
