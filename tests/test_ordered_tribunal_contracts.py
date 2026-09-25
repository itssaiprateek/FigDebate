"""Reliability checks, not model-accuracy measurements."""
import json
import unittest
from copy import deepcopy

from engine.evidence_review_v5 import verify, audit, mapping_response_valid, mapping_valid, mapping_schema
from engine.review_outcome import classify_review, failed_review, stage_outcomes
from engine.runtime_accounting import begin_accounting
from tests.test_tribunal_v5 import v5_fixture
from tests.test_evidence_review_v4 import Runtime


class OrderedContractTests(unittest.TestCase):
    def setUp(self):
        begin_accounting()

    def test_unmatched_is_valid_execution_but_never_an_acceptance(self):
        image, p, ledger, answers = v5_fixture()
        answers[1].update(bindings=[], unmatched_roles=['The caption subject cannot be located.'])
        runtime = Runtime(answers)
        proof = verify(runtime, image, p, ledger)
        p['independent_verification'] = proof
        self.assertEqual(len(runtime.prompts), 2)
        self.assertTrue(proof['obligations']['mapping']['response_valid'])
        self.assertFalse(proof['obligations']['mapping']['verified'])
        self.assertFalse(audit(p, ledger)['valid'])
        self.assertTrue(audit(p, ledger)['mapping_response_valid'])
        review = dict(_format_valid=True, _execution_status='SUCCEEDED', relation='CONFLICT',
                      provisional_verdict='CONTRADICTS', _independent_verification=proof)
        outcome = classify_review(review)
        mapping = outcome['stage_outcomes'][2]
        self.assertEqual((mapping['execution'], mapping['validation'], mapping['semantic']),
                         ('COMPLETED', 'VALID', 'UNRESOLVED'))
        self.assertEqual(outcome['verification_status'], 'SEMANTICALLY_REJECTED')
        self.assertEqual(outcome['stage_outcomes'][3]['execution'], 'NOT_RUN')

    def test_source_token_wire_also_allows_empty_honest_mapping(self):
        from engine.source_spans import SourceSpans
        source = 'A traveler returns home.'
        value = dict(bindings=[], unmatched_roles=['The traveler is not identifiable.'], reason='Identity is unresolved.')
        value.update(_format_valid=True, _execution_status='SUCCEEDED', _raw_output=json.dumps(value),
                     _source_span_protocol='indexed_source_tokens_v1')
        self.assertTrue(mapping_response_valid(value, {'VF1'}, source))
        self.assertFalse(mapping_valid(value, {'VF1'}, source))
        # A stored semantic response cannot contradict its original wire output.
        value['unmatched_roles'] = ['A different role.']
        self.assertFalse(mapping_response_valid(value, {'VF1'}, source))

    def test_empty_no_explanation_and_unknown_evidence_are_invalid(self):
        image, p, ledger, answers = v5_fixture()
        for body in [dict(answers[1], bindings=[], unmatched_roles=[]),
                     dict(answers[1], bindings=[dict(answers[1]['bindings'][0], evidence_ids=['WRONG'])])]:
            call = dict(body, _format_valid=True, _execution_status='SUCCEEDED', _raw_output=json.dumps(body))
            self.assertFalse(mapping_response_valid(call, {'VF1'}, p['source_caption']))

    def test_complete_mapping_remains_usable_and_tampering_fails(self):
        image, p, ledger, answers = v5_fixture()
        proof = verify(Runtime(answers), image, p, ledger)
        p['independent_verification'] = proof
        self.assertTrue(audit(p, ledger)['valid'])
        proof['obligations']['mapping']['bindings'][0]['observed_entity'] = 'Another person'
        self.assertFalse(audit(p, ledger)['valid'])

    def test_runtime_format_and_semantic_contradictions_are_distinct(self):
        cases = [
            (failed_review(TimeoutError('timed out'), 'judge'), ('FAILED', 'NOT_PRODUCED', 'NOT_ASSESSED')),
            (dict(_execution_status='SUCCEEDED', _format_valid=False, _output_status='TRUNCATED'),
             ('COMPLETED', 'TRUNCATED', 'NOT_ASSESSED')),
            (dict(_execution_status='SUCCEEDED', _format_valid=False, _contract_error_kind='SEMANTIC_INCONSISTENCY'),
             ('COMPLETED', 'VALID', 'CONTRADICTORY')),
            (dict(_execution_status='SUCCEEDED', _format_valid=True, relation='UNRESOLVED'),
             ('COMPLETED', 'VALID', 'UNRESOLVED')),
        ]
        for call, expected in cases:
            row = stage_outcomes(call)['stage_outcomes'][0]
            self.assertEqual(tuple(row[k] for k in ('execution', 'validation', 'semantic')), expected)

    def test_semantic_contract_failure_is_not_reported_as_runtime_failure(self):
        result = classify_review(dict(_execution_status='SUCCEEDED', _format_valid=False,
                                      _contract_error_kind='SEMANTIC_INCONSISTENCY'))
        self.assertEqual(result['terminal_outcome'], 'CONTRADICTORY_JUDGMENT')
        self.assertEqual(result['schema_status'], 'VALID')
        self.assertFalse(result['semantic_eligible'])
        self.assertFalse(result['semantic_abstained'])
        self.assertEqual(result['terminal_output_policy'], 'PRESERVE_INITIAL_WITH_INCONSISTENT_REVIEW')

    def test_failed_mapping_has_no_produced_schema(self):
        call = dict(failed_review(TimeoutError('timed out'), 'mapping'), response_valid=False)
        row = stage_outcomes({'_independent_verification': {'obligations': {'mapping': call}}})['stage_outcomes'][2]
        self.assertEqual((row['execution'], row['validation'], row['semantic']),
                         ('FAILED', 'NOT_PRODUCED', 'NOT_ASSESSED'))

    def test_typed_contradiction_cannot_reach_acceptance_despite_other_valid_fields(self):
        from engine.tribunal import apply_tribunal_resolution, record_tribunal_round
        from tests.test_factored_tribunal import case
        from tests.test_semantic_bridge import ledger, contract
        review = case()
        review['_contract_error_kind'] = 'SEMANTIC_INCONSISTENCY'
        for format_valid in (True, False):
            review['_format_valid'] = format_valid
            decision, _, gate = apply_tribunal_resolution({'label':'ENTAILS','confidence':.35},
                review,ledger(),contract(),semantic_bridge_mode='corroborated')
            self.assertFalse(gate['accepted'])
            self.assertEqual(decision['label'],'ENTAILS')
            self.assertEqual(gate['reason'],'contradictory_tribunal_judgment')
            self.assertEqual(record_tribunal_round(None,review)['state'],'PRESERVED')


if __name__ == '__main__':
    unittest.main()
