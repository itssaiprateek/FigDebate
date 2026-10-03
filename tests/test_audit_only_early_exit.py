"""Directional disagreement cannot pass; bounded repair still receives its audit."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engine.simple_judge import review
from engine.semantic_bridge import build_semantic_bridge
from engine.evidence_review_v5 import audit
from engine.tribunal import apply_tribunal_resolution
from engine.runtime_accounting import begin_accounting
from tests.test_evidence_review_v4 import Runtime
from tests.test_tribunal_v5 import v5_fixture
from tests.test_semantic_bridge import contract


class AuditEarlyExitTests(unittest.TestCase):
    def run_disagreement(self, repair_mode):
        begin_accounting()
        image, proposal, ledger, responses = v5_fixture()
        source = proposal['source_caption']
        candidate = dict(interpreted_assertion=source,
            decisive_reason='The scene supports the assertion of calmness.',
            evidence_ids=['VF1'], relation='SUPPORT')
        challenge = dict(alignment='MATCH', objections=[dict(type='COUNTEREVIDENCE',
            disputed_step='Claiming a calm scene', basis='The visible expressions show alarm.',
            evidence_ids=['VF1'])], evidence_ids=['VF1'],
            reason='Alarm opposes the claim of calmness.')
        runtime = Runtime([candidate, *deepcopy(responses[:3]), challenge])
        runtime.tribunal_audit_mode = 'audit-only-1'
        runtime.hardware_profile = SimpleNamespace(tribunal_protocol='evidence-review-5.0')
        if repair_mode is not None:
            runtime.tribunal_repair_mode = repair_mode
        ledger.append(dict(id='LC1', source='agent2', type='caption_proposition',
                           text=source, grounded=False))
        language = {'claim_contract': dict(contract(), source_caption=source,
                                           caption_proposition=source)}
        result = review(runtime, image, source, language, ledger)
        bridge = build_semantic_bridge(result, ledger, language['claim_contract'], language)
        decision, _, resolution = apply_tribunal_resolution(
            {'label': 'ENTAILS', 'confidence': .3}, result, ledger,
            language['claim_contract'], semantic_bridge_mode='corroborated', source_caption=source)
        self.assertEqual(decision['label'], 'ENTAILS')
        self.assertFalse(resolution['accepted'])
        self.assertFalse(audit(bridge, ledger)['valid'])
        return runtime, result, resolution

    def test_disabled_repair_skips_unreachable_audit(self):
        runtime, result, resolution = self.run_disagreement('disabled')
        self.assertEqual(len(runtime.prompts), 4)
        self.assertEqual(result['_independent_verification']['audit_skipped_reason'],
                         'direction_disagreement_cannot_pass_existing_gate')
        self.assertEqual(resolution['first_blocking_stage'], 'relation')
        self.assertEqual(resolution['stage_outcomes'][-1]['status'], 'NOT_RUN')

    def test_bounded_or_unknown_repair_keeps_audit(self):
        for mode in ('bounded', None):
            with self.subTest(mode=mode):
                runtime, result, _ = self.run_disagreement(mode)
                self.assertEqual(len(runtime.prompts), 5)
                self.assertNotIn('audit_skipped_reason', result['_independent_verification'])

    def test_batch_propagates_repair_mode(self):
        from engine.batch_runner import StagewiseRunner
        from tests.test_factored_tribunal import case, ledger
        from models.judge_model import QwenJudgeModel
        for mode in ('disabled', 'bounded'):
            with self.subTest(mode=mode):
                saved_review = case()
                saved_review['_generation_seconds'] = .01
                captured = []
                class Mediator:
                    def __init__(self, runtime):
                        captured.append(runtime)
                    def review(self, *args, **kwargs):
                        return deepcopy(saved_review)
                runner = StagewiseRunner.__new__(StagewiseRunner)
                runner.debate_mode = 'enabled'
                runner.semantic_bridge_mode = 'corroborated'
                runner.tribunal_audit_mode = 'audit-only-1'
                runner.tribunal_repair_mode = mode
                sample = {'index': 0, 'image': object(), 'caption': 'The plotted line rises.'}
                results = {0: dict(visual_output={}, language_output={'claim_contract': contract()},
                    comparison={}, evidence_ledger=ledger(), decision={'label': 'ENTAILS', 'confidence': .35},
                    debate_details={'agent2_requirements_valid': True}, timing={}, judge={})}
                with patch.object(QwenJudgeModel, '__init__', side_effect=AssertionError('No live model in test')), \
                     patch('engine.batch_runner.QwenJudgeModel', return_value=SimpleNamespace()), \
                     patch('engine.batch_runner.TribunalMediatorAgent', Mediator), \
                     patch('engine.batch_runner.GPUManager.clear'):
                    runner._run_tribunal_review_round([sample], results, 1)
                self.assertEqual(len(captured), 1)
                self.assertEqual(captured[0].tribunal_repair_mode, mode)


if __name__ == '__main__':
    unittest.main()
