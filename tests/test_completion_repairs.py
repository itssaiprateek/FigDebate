"""Regression coverage for observed retry and exact-source repair failures."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from agents.visual_adapter import AtomicVisualQuestionController as Controller
from agents.visual_grounding import VisualGroundingAgent
from engine.evidence_verification import verify, audit, quotation, bind_mapping_source_spans, mapping_valid
from engine.tribunal_protocol import generated_clause_error, source_quote_error
from models.vision_model import Qwen3VLVisionModel
from evaluation.tribunal_quality import summarize_tribunal
from tests.test_evidence_review_v4 import fixture, Runtime
from tests.test_qwen3vl_agent1 import FakeQwen3VLRuntime


class CompletionRepairTests(unittest.TestCase):
    def test_optional_visual_failure_is_reported_despite_valid_outer_schema(self):
        row = dict(initial_prediction='ENTAILS', prediction='ENTAILS', ground_truth='ENTAILS',
                   judge_requested=False, agent1_schema_format_valid=True)
        with_diagnostics = dict(row, trace={'visual_output': {'_internal': {'atomic_answers': [
            {'valid': True, 'status': 'OBSERVED', 'error': ''},
            {'valid': False, 'status': 'INVALID_RESPONSE', 'error': 'answer_too_long'}]}}})
        result = summarize_tribunal([with_diagnostics, row])
        self.assertEqual(result['cases_with_initial_visual_diagnostics'], 1)
        self.assertEqual(result['initial_visual_answers'], 2)
        self.assertEqual(result['failed_initial_visual_answers'], 1)
        self.assertEqual(result['cases_with_failed_initial_visual_answers'], 1)
        self.assertEqual(result['initial_visual_answer_errors'], {'answer_too_long': 1})

    def test_failed_retry_reports_its_own_error_and_preserves_primary(self):
        class TruncatedPrimaryRuntime(FakeQwen3VLRuntime):
            def generate(self, *args, **kwargs):
                result = super().generate(*args, **kwargs)
                self._last_generation_diagnostics['hit_token_limit'] = len(self.calls) == 1
                return result

        primary = 'The object is beside'
        retry = 'The red square is beside the blue circle. ' * 16
        runtime = TruncatedPrimaryRuntime([primary, retry])
        result = VisualGroundingAgent(runtime).answer_visual_question(
            object(), 'Describe the visible spatial relationship.', question_type='relation')
        self.assertFalse(result['valid'])
        self.assertFalse(result['retry_success'])
        self.assertEqual(result['answer'], retry.strip())
        self.assertEqual(result['error'], 'answer_too_long')
        self.assertEqual([v['error'] for v in result['validation_attempts']],
                         ['truncated_response', 'answer_too_long'])
        self.assertEqual(result['raw_response'], primary + '\n\nRETRY:\n' + retry)
        self.assertEqual(len(runtime.calls), 2)
        self.assertIn('at most 60 words in total', runtime.calls[1][0])

    def test_successful_retry_keeps_primary_failure_in_audit(self):
        runtime = FakeQwen3VLRuntime(['ENTAILS', 'A red square.'])
        result = VisualGroundingAgent(runtime).answer_visual_question(
            object(), 'What object is visible?', question_type='open')
        self.assertTrue(result['valid'])
        self.assertEqual(result['error'], '')
        self.assertEqual(result['validation_attempts'][0]['error'], 'dataset_decision_leak')
        self.assertTrue(result['validation_attempts'][1]['valid'])

    def test_complete_primary_does_not_trigger_or_change_retry_behavior(self):
        runtime = FakeQwen3VLRuntime(['A red square.'])
        result = VisualGroundingAgent(runtime).answer_visual_question(
            object(), 'What object is visible?', question_type='open')
        self.assertTrue(result['valid'])
        self.assertFalse(result['retry_attempted'])
        self.assertEqual(len(result['validation_attempts']), 1)
        self.assertEqual(len(runtime.calls), 1)
        self.assertIn('Answer briefly using only directly visible evidence.', runtime.calls[0][0])

    def test_ocr_retry_does_not_impose_a_short_prose_budget(self):
        prompt = Controller.retry_prompt('Copy every readable character.', 'ocr')
        self.assertIn('requested visible text', prompt)
        self.assertNotIn('words in total', prompt)
        result = Controller.validate_answer('"SAFE"', 'Read the sign.', 'ocr', {'hit_token_limit': True})
        self.assertFalse(result[2])
        self.assertEqual(result[3], 'truncated_response')

    def test_exact_limit_requires_recorded_eos_for_prose(self):
        class Batch(dict):
            def to(self, _device):
                return self

        for last_token, ended in ((9, True), (8, False)):
            with self.subTest(last_token=last_token):
                runtime = Qwen3VLVisionModel.__new__(Qwen3VLVisionModel)
                runtime.processor = SimpleNamespace(
                    apply_chat_template=lambda *a, **k: Batch(input_ids=torch.tensor([[1, 2]])),
                    batch_decode=lambda *a, **k: ['A red square.'])
                runtime.model = SimpleNamespace(device='cpu',
                    generation_config=SimpleNamespace(eos_token_id=[9, 10]),
                    generate=lambda **k: torch.tensor([[1, 2, 3, last_token]]))
                with patch('torch.cuda.is_available', return_value=False):
                    raw, _, diagnostics = runtime._generate_once(Image.new('RGB', (2, 2)), 'q', 2, use_cache=True)
                self.assertTrue(diagnostics['hit_token_limit'])
                self.assertEqual(diagnostics['ended_by_eos'], ended)
                self.assertEqual(Controller.validate_answer(raw, 'Describe the scene.', 'scene', diagnostics)[2], ended)
        self.assertFalse(Controller.validate_answer('The square is beside the', 'Describe the scene.', 'scene',
            {'hit_token_limit': True, 'ended_by_eos': True})[2])

    def test_repair_identifies_partial_field_without_accepting_it(self):
        error = generated_clause_error({'node_relations': [{'observation': "The label says 'Every ticket includes an album, "}]})
        self.assertIn('node_relations[0].observation', error)
        self.assertIn('Every ticket includes an album, ', error)
        self.assertIn('Do not copy an unfinished quotation', error)
        self.assertEqual(generated_clause_error({'caption_quote': 'with the'}), '')
        self.assertEqual(generated_clause_error({'observation': 'The label describes an album.'}), '')

    def test_capitalized_quote_is_rejected_until_model_returns_exact_source(self):
        image, proposal, ledger, answers = fixture()
        wrong = deepcopy(answers[1])
        wrong['bindings'][0]['caption_quote'] = 'Meeting'
        runtime = Runtime([answers[0], wrong, answers[1], answers[2], answers[3]])
        proposal['independent_verification'] = verify(runtime, image, proposal, ledger)
        self.assertTrue(audit(proposal, ledger)['valid'])
        self.assertTrue(proposal['independent_verification']['obligations']['mapping']['_format_retry_success'])
        self.assertFalse(quotation('Meeting', proposal['source_caption']))
        runtime = Runtime([answers[0], wrong, wrong])
        proposal['independent_verification'] = verify(runtime, image, proposal, ledger)
        self.assertFalse(audit(proposal, ledger)['valid'])

    def test_unique_phrase_capitalization_is_bound_without_changing_roles(self):
        image, proposal, ledger, answers = fixture()
        answers[1]['bindings'][0]['caption_quote'] = 'Meeting is calm'
        original = deepcopy(answers[1])
        runtime = Runtime(answers)
        proposal['independent_verification'] = verify(runtime, image, proposal, ledger)
        call = proposal['independent_verification']['obligations']['mapping']
        self.assertTrue(audit(proposal, ledger)['valid'])
        self.assertEqual(len(runtime.prompts), 4)
        self.assertFalse(call['_format_retry_used'])
        self.assertEqual(call['bindings'][0]['caption_quote'], 'meeting is calm')
        self.assertEqual(json.loads(call['_raw_output']), original)
        self.assertEqual(call['bindings'][0]['observed_entity'], original['bindings'][0]['observed_entity'])
        self.assertEqual(call['bindings'][0]['role_scope'], original['bindings'][0]['role_scope'])
        repair = call['_source_quote_bindings'][0]
        self.assertEqual(proposal['source_caption'][repair['normalized_source_start']:repair['normalized_source_end']],
                         repair['source_quote'])
        call['bindings'][0]['observed_entity'] = 'A different person'
        self.assertFalse(mapping_valid(call, {'VF1'}, proposal['source_caption']))

    def test_source_binding_never_changes_words_names_internal_case_or_ambiguous_spans(self):
        for quote, source in [
            ('Meeting', 'The meeting is calm.'),
            ('US citizens', 'The us citizens text is visible.'),
            ('Meeting is loud', 'The meeting is calm.'),
            ('Meeting is not calm', 'The meeting is calm.'),
            ('The meeting', 'the meeting follows the meeting'),
        ]:
            with self.subTest(quote=quote):
                value = deepcopy(fixture()[-1][1])
                value['bindings'][0]['caption_quote'] = quote
                self.assertEqual(bind_mapping_source_spans(value, source), value)

    def test_source_binding_does_not_certify_unmatched_roles(self):
        image, proposal, ledger, answers = fixture()
        answers[1]['bindings'][0]['caption_quote'] = 'Meeting is calm'
        answers[1]['unmatched_roles'] = ['meeting']
        runtime = Runtime(answers[:2])
        proposal['independent_verification'] = verify(runtime, image, proposal, ledger)
        self.assertFalse(audit(proposal, ledger)['valid'])
        self.assertEqual(proposal['independent_verification']['stopped_after'], 'entity_scope_mapping')
        call = proposal['independent_verification']['obligations']['mapping']
        call['unmatched_roles'] = []
        call['verified'] = True
        self.assertFalse(mapping_valid(call, {'VF1'}, proposal['source_caption']))

    def test_forged_source_binding_audit_cannot_hide_a_malformed_raw_output(self):
        call = dict(fixture()[-1][1], _format_valid=True, _execution_status='SUCCEEDED',
                    _source_quote_bindings=[{'method': 'unique_span_initial_capitalization'}],
                    _raw_output='{"bindings": "invalid"}')
        self.assertFalse(mapping_valid(call, {'VF1'}, 'The meeting is calm.'))

    def test_quote_diagnostic_does_not_invent_a_span_or_choose_an_ambiguous_one(self):
        error = source_quote_error('mystery', 'The meeting is calm.')
        self.assertNotIn('exact source spelling', error)
        error = source_quote_error('Us', 'US and us are written separately.')
        self.assertNotIn('exact source spelling', error)
        self.assertEqual(source_quote_error('meeting', 'The meeting is calm.'), '')


if __name__ == '__main__':
    unittest.main()
