"""Citation failures exercise the actual generation and acceptance path."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from engine import semantic_protocol as protocol
from tests.test_audit_only_protocol import AuditOnlyTests
from tests.test_evidence_review_v4 import Runtime


class AuditCitationTests(unittest.TestCase):
    def test_duplicate_cannot_trigger_a_retry_that_erases_an_objection(self):
        original = Runtime.generate
        attempts = []

        def generate(runtime, image, prompt, max_new_tokens=None, json_schema=None):
            if 'alignment' in json_schema.get('properties', {}):
                attempts.append(prompt)
                if len(attempts) == 1:
                    duplicate = deepcopy(runtime.answers[0])
                    duplicate['evidence_ids'] *= 2
                    return json.dumps(duplicate), .01
                # This tempting "format repair" must never be requested.
                rewritten = deepcopy(runtime.answers[0])
                rewritten.update(objections=[], reason='The conflict is fully grounded.')
                return json.dumps(rewritten), .01
            return original(runtime, image, prompt, max_new_tokens, json_schema)

        with patch.object(Runtime, 'generate', generate):
            _, bridge, _, final, resolution = AuditOnlyTests().run_case(objection=True)
        self.assertEqual(len(attempts), 1)
        self.assertFalse(resolution['accepted'])
        self.assertEqual(final['label'], 'ENTAILS')
        audit = bridge['independent_verification']['obligations']['arguments']
        self.assertEqual(audit['evidence_ids'], ['VF1', 'VF1'])
        self.assertTrue(audit['objections'])
        self.assertEqual(audit['_contract_error_kind'], 'AUDIT_DUPLICATE_CITATION')

    def test_persistent_duplicate_does_not_change_answer(self):
        original = Runtime.generate
        attempts = []

        def generate(runtime, image, prompt, max_new_tokens=None, json_schema=None):
            if 'alignment' in json_schema.get('properties', {}):
                attempts.append(prompt)
                duplicate = deepcopy(runtime.answers[0])
                duplicate['evidence_ids'] *= 2
                return json.dumps(duplicate), .01
            return original(runtime, image, prompt, max_new_tokens, json_schema)

        with patch.object(Runtime, 'generate', generate):
            _, _, _, final, resolution = AuditOnlyTests().run_case()
        self.assertEqual(len(attempts), 1)
        self.assertFalse(resolution['accepted'])
        self.assertEqual(final['label'], 'ENTAILS')

    def test_objection_lists_are_checked_independently(self):
        value = dict(alignment='MATCH', evidence_ids=['A'], reason='A material uncertainty remains.',
                     objections=[dict(type='COUNTEREVIDENCE', disputed_step='Event direction',
                                      basis='The observed roles are reversed.', evidence_ids=['A', 'A'])])
        self.assertEqual(protocol.audit_error(value)['kind'], 'AUDIT_DUPLICATE_CITATION')
        value['objections'][0]['evidence_ids'] = ['A']
        self.assertIsNone(protocol.audit_error(value))

    def test_generator_preserves_both_orders_and_wire_schema(self):
        known = {'A', 'B'}
        wire_before = protocol.audit_schema(known)
        schema = protocol.audit_generation_schema(known)
        choices = schema['properties']['evidence_ids']['enum']
        self.assertIn(['B', 'A'], choices)
        self.assertIn(['A', 'B'], choices)
        self.assertNotIn(['A', 'A'], choices)
        self.assertEqual(wire_before, protocol.audit_schema(known))
        # Larger future registries retain bounded schema construction. Runtime
        # validation and wire uniqueness still apply to their delivered records.
        self.assertEqual(protocol.audit_schema(set('abcdefg')),
                         protocol.audit_generation_schema(set('abcdefg')))


if __name__ == '__main__':
    unittest.main()
