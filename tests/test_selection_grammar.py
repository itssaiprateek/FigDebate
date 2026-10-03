"""Exercise the real decoder and acceptance gate without model weights."""
from copy import deepcopy
import itertools
import json
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from engine import selection_grammar
from engine.evidence_review_v5 import SELECTION_WIRE
from engine.structured_decoder import prefix_constraint
from tests.test_audit_only_protocol import AuditOnlyTests
from tests.test_evidence_review_v4 import Runtime


class ByteTokenizer:
    def __init__(self):
        vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        self.eos_token_id = len(vocab)
        vocab['<eos>'] = self.eos_token_id
        self.backend_tokenizer = Tokenizer(models.BPE(vocab, []))
        self.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.backend_tokenizer.decoder = decoders.ByteLevel()

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size()


def schema(ids):
    result = deepcopy(SELECTION_WIRE)
    result['properties']['observations']['items']['properties']['evidence_id'] = {
        'type': 'string', 'enum': ids}
    return result


def value(ids):
    return dict(observations=[dict(evidence_id=i, supported=True, image_location='left panel')
                              for i in ids], coverage_complete=True, missing_observation='')


class SelectionGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = ByteTokenizer()

    def accepts(self, result, wire):
        allowed = prefix_constraint(self.tokenizer, wire, compact=True)
        prefix = self.tokenizer.backend_tokenizer.encode('Prompt').ids
        raw = result if isinstance(result, str) else json.dumps(result, separators=(',', ':'))
        for token in self.tokenizer.backend_tokenizer.encode(raw).ids:
            if token not in allowed(0, torch.tensor(prefix)):
                return False
            prefix.append(token)
        return self.tokenizer.eos_token_id in allowed(0, torch.tensor(prefix))

    def test_every_subset_available_and_duplicate_or_unknown_rejected(self):
        ids = ['A', 'B', 'C', 'D', 'E']
        wire = schema(ids)
        for size in range(1, 5):
            for subset in itertools.combinations(ids, size):
                self.assertTrue(self.accepts(value(subset), wire), subset)
        for selected in ([], ['A', 'A'], ['B', 'A'], ['F'], ids):
            self.assertFalse(self.accepts(value(selected), wire), selected)

    def test_visibility_and_coverage_are_not_forced(self):
        result = value(['A'])
        result['observations'][0]['supported'] = False
        result.update(coverage_complete=False, missing_observation='Which object?')
        self.assertTrue(self.accepts(result, schema(['A', 'B'])))
        self.assertTrue(self.accepts(json.dumps(result, separators=(',\n', ': ')), schema(['A', 'B'])))

    def test_large_catalogue_is_bounded_and_inputs_are_not_mutated(self):
        wire = schema([f'E{i:02}' for i in range(64)])
        original = deepcopy(wire)
        self.assertTrue(self.accepts(value(['E00', 'E20', 'E40', 'E63']), wire))
        self.assertFalse(self.accepts(value(['E00', 'E20', 'E20', 'E63']), wire))
        self.assertEqual(wire, original)
        self.assertLess(len(selection_grammar.grammar(wire, 1)), 40000)
        self.assertFalse(selection_grammar.eligible(schema([f'E{i}' for i in range(65)])))

    def test_schema_changes_cannot_silently_lose_constraints(self):
        original = schema(['A', 'B'])
        variants = []
        for key, val in [('minItems', 0), ('maxItems', 6), ('prefixItems', [])]:
            changed = deepcopy(original)
            changed['properties']['observations'][key] = val
            variants.append(changed)
        changed = deepcopy(original)
        changed['required'] = ['observations']
        variants.append(changed)
        changed = deepcopy(original)
        changed['allOf'] = []
        variants.append(changed)
        changed = deepcopy(original)
        changed['properties']['observations']['items']['required'] = ['evidence_id']
        variants.append(changed)
        variants.extend([{}, schema([]), schema(['A', 'A']), schema(['A', 2]), schema([''])])
        for changed in variants:
            self.assertFalse(selection_grammar.eligible(changed), changed)
        # The original decoder still enforces a different schema's own bounds.
        other = deepcopy(original)
        other['properties']['observations']['minItems'] = 2
        self.assertFalse(self.accepts(value(['A']), other))
        self.assertTrue(self.accepts(value(['A', 'B']), other))

    def test_existing_gate_rejects_visibility_failure_and_accepts_valid_proof(self):
        with patch('models.judge_model.QwenJudgeModel', side_effect=AssertionError('No weights')):
            _, _, _, final, resolution = AuditOnlyTests().run_case()
            self.assertTrue(resolution['accepted'])
            self.assertEqual(final['label'], 'CONTRADICTS')
            original = Runtime.generate

            def generate(runtime, image, prompt, max_new_tokens=None, json_schema=None):
                raw, elapsed = original(runtime, image, prompt, max_new_tokens, json_schema)
                if 'observations' in json_schema.get('properties', {}):
                    output = json.loads(raw)
                    output['observations'][0]['supported'] = False
                    raw = json.dumps(output)
                return raw, elapsed

            with patch.object(Runtime, 'generate', generate):
                _, _, _, final, resolution = AuditOnlyTests().run_case()
            self.assertFalse(resolution['accepted'])
            self.assertEqual(final['label'], 'ENTAILS')


if __name__ == '__main__':
    unittest.main()
