"""General contract/recovery checks; no model-accuracy assertions."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

from engine.output_contracts import validate_shape, incomplete_clause, text_length_boundaries
from engine.source_spans import SourceSpans
from engine.evidence_verification import obligation_runner, MAPPING, DECISION
from tests.test_evidence_review_v4 import Runtime, fixture


class TransportTests(unittest.TestCase):
    def test_integer_type_and_bounds(self):
        schema = {'type': 'integer', 'minimum': 0, 'maximum': 4}
        for value in ('1', True, False, -1, 5, 1.5, 1.0, None, 10**400):
            with self.subTest(value=value):
                self.assertFalse(validate_shape(value, schema))
        for value in range(5):
            self.assertTrue(validate_shape(value, schema))

    def ask(self, name, wire, replacement, validator=None):
        image, proposal, ledger, _ = fixture()
        runtime = Runtime([wire, replacement])
        runtime.hardware_profile = SimpleNamespace(judge_source_spans=True)
        run = obligation_runner(runtime, image, proposal['source_caption'], ledger, {'VF1'}, version='5.0')
        output = run(name, 'Check the evidence.', {}, MAPPING if name == 'mapping' else DECISION,
                     validator=validator)
        return output, runtime

    def mapping(self, span):
        value = deepcopy(fixture()[-1][1])
        value['bindings'][0].pop('caption_quote')
        value['bindings'][0]['caption_span'] = span
        return value

    def test_all_invalid_interval_classes_get_one_local_repair(self):
        for span in ({'start': 1, 'end': 1}, {'start': 3, 'end': 2}, {'start': -1, 'end': 2},
                     {'start': 0, 'end': 5}, {'start': True, 'end': 2}, {'start': 1.5, 'end': 2},
                     {'start': '1', 'end': 2}, {'start': 1}, None):
            with self.subTest(span=span):
                original = self.mapping(span)
                output, runtime = self.ask('mapping', original, {'replacement': {'start': 1, 'end': 2}})
                self.assertTrue(output['_format_valid'])
                self.assertEqual(output['_format_retry_strategy'], 'single_field_repair')
                self.assertEqual(output['bindings'][0]['caption_quote'], 'meeting')
                restored = json.loads(output['_raw_output'])
                expected = deepcopy(original)
                expected['bindings'][0]['caption_span'] = {'start': 1, 'end': 2}
                self.assertEqual(restored, expected)
                self.assertEqual(runtime.token_budgets, [384, 160])

    def test_unestablished_condition_interval_repaired_without_changing_decision(self):
        wire = deepcopy(fixture()[-1][2])
        wire['condition_checks'][0]['caption_span'] = {'start': 3, 'end': 4}
        wire['condition_checks'][0].pop('caption_quote')
        wire['unestablished_conditions'] = [{'start': 4, 'end': 4}]
        output, _ = self.ask('relation', wire, {'replacement': {'start': 0, 'end': 2}})
        self.assertTrue(output['_format_valid'])
        self.assertEqual(output['relation'], 'CONFLICT')
        self.assertEqual(output['unestablished_conditions'], ['The meeting'])
        self.assertEqual(output['condition_checks'][0]['caption_quote'], 'calm.')

    def test_second_invalid_span_cannot_get_third_attempt(self):
        wire = self.mapping({'start': 1, 'end': 1})
        wire['bindings'].append(deepcopy(wire['bindings'][0]))
        output, runtime = self.ask('mapping', wire, {'replacement': {'start': 1, 'end': 2}})
        self.assertFalse(output['_format_valid'])
        self.assertEqual(len(runtime.prompts), 2)

    def test_retry_cannot_change_unrelated_fields_or_skip_validator(self):
        original = self.mapping({'start': 1, 'end': 1})
        for patch in ({'replacement': {'start': 1, 'end': 2}, 'reason': 'Changed'},
                      {'replacement': {'start': 1, 'end': 1}}):
            output, _ = self.ask('mapping', original, patch)
            self.assertFalse(output['_format_valid'])
        output, _ = self.ask('mapping', original, {'replacement': {'start': 1, 'end': 2}},
                             validator=lambda value: 'Deliberate remaining proof defect')
        self.assertFalse(output['_format_valid'])

    def test_valid_input_unchanged_and_no_retry(self):
        original = self.mapping({'start': 1, 'end': 2})
        output, runtime = self.ask('mapping', original, {})
        self.assertTrue(output['_format_valid'])
        self.assertEqual(len(runtime.prompts), 1)
        self.assertEqual(json.loads(output['_raw_output']), original)

    def test_generated_clause_vs_exact_source_fragment(self):
        from engine.tribunal_protocol import generated_clause_error
        for text in ('The panel shows a choice or', 'The people stand and'):
            self.assertFalse(incomplete_clause(text))  # Existing visual-agent behavior is preserved.
            self.assertTrue(incomplete_clause(text, conjunctions=True))
            self.assertTrue(generated_clause_error({'reason': text}))
            self.assertFalse(generated_clause_error({'caption_quote': text}))
        self.assertFalse(incomplete_clause('Left panel'))
        self.assertFalse(incomplete_clause('The first object is red'))

    def test_length_boundary_is_diagnostic_not_rejection(self):
        schema = {'type': 'object', 'properties': {'reason': {'type': 'string', 'maxLength': 9}},
                  'required': ['reason']}
        value = {'reason': 'Red panel'}
        self.assertEqual(text_length_boundaries(json.dumps(value), schema), ['reason'])
        self.assertTrue(validate_shape(value, schema))

    def test_attachment_repair_is_local_and_identifies_field_role(self):
        from engine.evidence_verification import VISUAL
        image, proposal, ledger, answers = fixture()
        wire = answers[0]
        wire['observations'][0]['attachment'] = 'The text says the'
        runtime = Runtime([wire, {'replacement':'Top left text panel.'}])
        runtime.hardware_profile = SimpleNamespace(judge_source_spans=True)
        ask = obligation_runner(runtime, image, proposal['source_caption'], ledger, {'VF1'})
        output = ask('visual', 'Inspect the image.', {}, VISUAL)
        self.assertTrue(output['_format_valid'])
        self.assertEqual(output['observations'][0]['attachment'], 'Top left text panel.')
        self.assertEqual(output['observations'][0]['observed'], wire['observations'][0]['observed'])
        self.assertIn('do not repeat the text printed there', runtime.prompts[1])
        self.assertNotIn('The text says the', runtime.prompts[1])
        self.assertEqual(runtime.token_budgets, [384, 160])


if __name__ == '__main__':
    unittest.main()
