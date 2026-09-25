"""Contract and routing regressions; scripted responses do not prove accuracy."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

from engine.source_spans import SourceSpans, restore_field
from engine.evidence_verification import obligation_runner, DECISION
from engine.evidence_review_v5 import verify, audit
from tests.test_evidence_review_v4 import Runtime
from tests.test_tribunal_v5 import v5_fixture


class OverhaulReliabilityTests(unittest.TestCase):
    def test_capped_semantic_prose_stops_without_rewriting_its_meaning(self):
        from engine.evidence_verification import capped_generated_clause
        from engine.output_contracts import object_schema
        schema = object_schema({'reason': {'type': 'string', 'maxLength': 16},
                                'relation': {'type': 'string', 'enum': ['CONFLICT']}})
        self.assertEqual(capped_generated_clause({'reason': 'The comparison *', 'relation': 'CONFLICT'}, schema), ('reason',))
        self.assertIsNone(capped_generated_clause({'reason': 'A complete idea.', 'relation': 'CONFLICT'}, schema))
        image, proposal, ledger, _ = v5_fixture()
        runtime = Runtime([{'reason': 'The comparison *', 'relation': 'CONFLICT'}])
        runtime.hardware_profile = SimpleNamespace(judge_source_spans=True)
        result = obligation_runner(runtime, image, proposal['source_caption'], ledger, {'VF1'}, '5.0')(
            'challenge', 'Audit this comparison.', {}, schema)
        self.assertFalse(result['_format_valid'])
        self.assertEqual(result['reason'], 'The comparison *')
        self.assertEqual(result['relation'], 'CONFLICT')
        self.assertEqual(len(runtime.prompts), 1)
        self.assertEqual(result['_retry_block_reason'], 'capped_semantic_clause_requires_new_review')

    def test_generation_schema_excludes_every_invalid_interval(self):
        from engine.output_contracts import validate_shape
        for text in ('One', 'A longer source with five words.', 'Nobody  won\n—not even Zoë.'):
            codec = SourceSpans(text)
            n = len(codec.tokens)
            for start in range(-1, n + 2):
                for end in range(-1, n + 2):
                    self.assertEqual(validate_shape({'start': start, 'end': end}, codec.interval_schema()),
                                     0 <= start < end <= n)
            self.assertFalse(validate_shape({'start': True, 'end': 1}, codec.interval_schema()))
        self.assertFalse(validate_shape({'start': 0, 'end': 1}, SourceSpans('').interval_schema()))

    def test_terminal_categories_keep_agreement_failure_and_uncertainty_distinct(self):
        from engine.review_outcome import resolution_category
        resolved = {'_format_valid': True, '_execution_status': 'SUCCEEDED', 'relation': 'SUPPORT',
                    'provisional_verdict': 'ENTAILS'}
        self.assertEqual(resolution_category(resolved, {'confirmation_valid': True}), 'SUPPORTED_AGREEMENT')
        self.assertEqual(resolution_category(resolved, {'accepted': True}), 'SUPPORTED_CORRECTION')
        self.assertEqual(resolution_category(dict(resolved, relation='UNRESOLVED', provisional_verdict='ABSTAIN'), {}),
                         'UNRESOLVED_INTERPRETATION')
        failed = dict(resolved, _independent_verification={'obligations': {
            'visual': {'_execution_status': 'FAILED', '_format_valid': False}}})
        self.assertEqual(resolution_category(failed, {}), 'EXECUTION_OR_CONTRACT_FAILURE')
        missing = dict(resolved, _independent_verification={'obligations': {
            'visual': {'_execution_status': 'SUCCEEDED', '_format_valid': True, 'coverage_complete': False}}})
        self.assertEqual(resolution_category(missing, {}), 'MISSING_EVIDENCE')
        contradictory = dict(resolved, _independent_verification={'schema_version': '5.0', 'calls': [{
            '_execution_status': 'SUCCEEDED', '_format_valid': False,
            '_contract_error_kind': 'SEMANTIC_INCONSISTENCY'}]})
        self.assertEqual(resolution_category(contradictory, {}), 'UNRESOLVED_INTERPRETATION')
        from engine.review_outcome import classify_review
        self.assertEqual(classify_review(contradictory)['verification_status'], 'INCONSISTENT_VERIFICATION_JUDGMENT')
        self.assertEqual(classify_review(contradictory)['terminal_output_policy'], 'PRESERVE_INITIAL_WITH_INCONSISTENT_REVIEW')

    def test_all_invalid_spans_repaired_together_without_changing_decision(self):
        codec = SourceSpans('The whole room is unusually quiet.')
        original = {'relation': 'CONFLICT', 'condition_checks': [
            {'caption_span': {'start': 4, 'end': 2}, 'image_state': 'People shout.'},
            {'caption_span': {'start': 99, 'end': 1}, 'image_state': 'A crowded room.'}],
            'unestablished_conditions': [{'start': 3, 'end': 3}]}
        state = {}
        failure = codec.repair_request(original, state)
        self.assertEqual(len(state['paths']), 3)
        restored = restore_field({'replacement_0': {'start': 3, 'end': 6},
            'replacement_1': {'start': 0, 'end': 3}, 'replacement_2': {'start': 4, 'end': 6}}, state)
        self.assertIsNone(codec.repair_request(restored, state))
        self.assertEqual(restored['relation'], original['relation'])
        self.assertEqual(restored['condition_checks'][0]['image_state'], 'People shout.')
        self.assertEqual(original['condition_checks'][0]['caption_span'], {'start': 4, 'end': 2})
        self.assertLessEqual(failure['_repair_max_tokens'], 384)
        with self.assertRaises(ValueError):
            restore_field({'replacement_0': {'start': 0, 'end': 1}}, state)

    def test_real_generation_repairs_two_spans_in_one_retry(self):
        image, proposal, ledger, _ = v5_fixture()
        raw = dict(relation='CONFLICT', evidence_ids=['VF1'], condition_checks=[
            dict(caption_span={'start': 3, 'end': 1}, image_state='People look alarmed.', relation='CONFLICT'),
            dict(caption_span={'start': 2, 'end': 2}, image_state='Participants sit together.', relation='SUPPORT')],
            unestablished_conditions=[], reason='Alarm conflicts with calmness.')
        runtime = Runtime([raw, {'replacement_0': {'start': 3, 'end': 4}, 'replacement_1': {'start': 0, 'end': 2}}])
        runtime.hardware_profile = SimpleNamespace(judge_source_spans=True)
        result = obligation_runner(runtime, image, proposal['source_caption'], ledger, {'VF1'}, '5.0')(
            'relation', 'Compare the sources.', {}, DECISION)
        self.assertTrue(result['_format_valid'], result)
        self.assertEqual(len(runtime.prompts), 2)
        self.assertEqual(result['relation'], raw['relation'])
        self.assertEqual(len(result['_field_repair_audit']['paths']), 2)
        self.assertEqual(json.loads(result['_raw_output'])['condition_checks'][1]['caption_span'], {'start': 0, 'end': 2})

    def test_duplicate_selection_repair_preserves_coverage_failure(self):
        image, proposal, ledger, answers = v5_fixture()
        visual = deepcopy(answers[0])
        visual['observations'] *= 2
        visual.update(coverage_complete=False, missing_observation='Read the other panel.')
        runtime = Runtime([visual, {'replacement': answers[0]['observations']}])
        proof = verify(runtime, image, proposal, ledger)
        proposal['independent_verification'] = proof
        repaired = proof['obligations']['visual']
        self.assertTrue(repaired['_format_valid'])
        self.assertFalse(repaired['coverage_complete'])
        self.assertEqual(repaired['missing_observation'], 'Read the other panel.')
        self.assertFalse(audit(proposal, ledger)['valid'])
        self.assertEqual(len(runtime.prompts), 2)

    def test_failed_second_span_patch_does_not_open_another_retry(self):
        codec, state = SourceSpans('A short caption.'), {}
        codec.repair_request({'caption_span': {'start': 3, 'end': 1}}, state)
        with self.assertRaises(ValueError):
            restore_field({'replacement': {'start': 2, 'end': 1}}, state)
        failure = codec.repair_request({'caption_span': {'start': 2, 'end': 1}}, state)
        self.assertFalse(failure['_format_valid'])
        self.assertNotIn('_repair_schema', failure)


if __name__ == '__main__':
    unittest.main()
