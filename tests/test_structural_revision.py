"""General routing, transport and semantic-projection contracts; no accuracy claims."""
import copy
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from agents.visual_adapter import AtomicVisualQuestionController as V, VisualQuestion
from engine.claim_graph import source_identity_graph
from engine.question_router import build_question_plan, compile_visual_question, ISSUE_TYPES
from engine.review_routing import route_question
from engine.runtime_profile import resolve_reasoning_modes
from engine import condition_review as C


class StructuralRevisionTests(unittest.TestCase):
    def test_source_graph_does_not_hide_invalid_or_external_requirements(self):
        for additions, issue in [({'safe_for_tribunal_reasoning': False}, 'CLAIM_CONTRACT_INVALID'),
                                 ({'requires_background_knowledge': True}, 'BACKGROUND_KNOWLEDGE'),
                                 ({'structural_reasoning_type': 'COMPARATIVE_LAYOUT'}, 'COMPARISON_OR_OUTCOME')]:
            c = {'claim_contract': {'safe_for_tribunal_reasoning': True,
                 'claim_graph': source_identity_graph('A complete source assertion.'), **additions}}
            before = copy.deepcopy(c)
            plan = build_question_plan(c)
            self.assertEqual(plan.issue_type, issue)
            self.assertEqual(c, before)
            if issue != 'COMPARISON_OR_OUTCOME':
                self.assertFalse(plan.visually_resolvable)
                if issue == 'CLAIM_CONTRACT_INVALID':
                    self.assertEqual(plan.agent1_question, '')
                else:
                    self.assertIn('visible', plan.agent1_question)

    def test_all_emitted_visual_plans_share_the_capability_policy(self):
        for issue in ISSUE_TYPES:
            with self.subTest(issue=issue), patch('engine.question_router.classify_issue', return_value=issue):
                plan = build_question_plan({'claim_relation': {'subject': 'an allegedly dishonest winner'}})
                if plan.agent1_question:
                    request = VisualQuestion.witness(plan.agent1_question, 'q')
                    self.assertEqual(request.source, 'image')
                    self.assertNotIn('dishonest', request.text)
                    self.assertEqual(compile_visual_question({}, {'agent1_questions': [request.text]}), request.text)

    def test_invalid_requests_do_not_silently_become_other_questions(self):
        for text in ['', 'What is the final label?', 'Transcribe the sign and describe the person.', 'x' * 321]:
            with self.subTest(text=text):
                self.assertEqual(compile_visual_question({}, {'agent1_questions': [text]}), '')
                with self.assertRaises(ValueError):
                    VisualQuestion.witness(text, 'q')

    def test_caption_source_and_visible_text_are_distinct(self):
        for q in ['What does the caption say?', 'Explain the wording of the caption.', 'What does this idiom mean?']:
            self.assertEqual(route_question(q)[0], 'language')
        for q in ['Transcribe the displayed wording.', 'Which printed word belongs to the left panel?']:
            self.assertEqual(route_question(q, 'VISUAL_PREMISE')[0], 'visual')
            self.assertEqual(V.infer_question_type(q), 'ocr')

    def test_compound_commands_reject_but_relational_scope_remains(self):
        self.assertFalse(V.validate_question('Count the people and explain their motives.')[0])
        self.assertTrue(V.validate_question('Describe the relationship between the two pictured objects.')[0])

    def test_configuration_resolution_preserves_explicit_controls(self):
        settings = dict(judge_mode='tribunal', hardware_profile='paper-8gb-review5')
        self.assertEqual(resolve_reasoning_modes(**settings), ('grounded', 'focused-audit-1'))
        self.assertEqual(resolve_reasoning_modes(**settings, reasoning_mode='baseline', tribunal_audit_mode='baseline'), ('baseline', 'baseline'))
        with self.assertRaises(ValueError):
            resolve_reasoning_modes(tribunal_audit_mode='typo')
        from engine.semantic_protocol import mode
        with self.assertRaises(ValueError):
            mode(SimpleNamespace(tribunal_audit_mode='typo'))

    def test_typed_witness_uses_only_image_and_question(self):
        from agents.visual_grounding import VisualGroundingAgent
        agent = VisualGroundingAgent.__new__(VisualGroundingAgent)
        agent.question_controller = V()
        agent._run_atomic_question = lambda image, request: SimpleNamespace(to_dict=lambda: dict(valid=True, status='OBSERVED', answer='A line connects two circles.', question_id=request.question_id))
        request = VisualQuestion.witness('Describe the visible relation between the circles.', 'q1')
        result = agent.critique(object(), request)
        self.assertEqual(result['question_id'], 'q1')
        self.assertEqual(result['question'], request.text)
        self.assertEqual(result['response_status'], 'VALID_OBSERVATION')
        self.assertEqual(result['claim_relation'], 'UNRESOLVED')
        self.assertFalse(result['witness_contract']['direction_assigned'])
        with self.assertRaises(ValueError):
            VisualGroundingAgent._extract_atomic_question('No request header')

    def test_failed_dispatch_never_calls_a_model(self):
        from engine.debate import DebateEngine
        class Agent:
            def critique(self, *args):
                raise AssertionError('No model call allowed')
        case = dict(key='q', image=object(), caption='source', visual_output={}, decision={}, comparison={},
                    tribunal_hearing=True, mediation_plan={'agent1_questions':['Transcribe text and describe the person.']})
        value = DebateEngine.__new__(DebateEngine)._run_visual_questions(Agent(),case)
        self.assertFalse(value['_format_valid'])
        self.assertEqual(value['communication']['not_dispatched'], 1)

    def test_canonical_projection_preserves_unknown_and_raw_binding(self):
        value = {'reading': {'kind':'LITERAL','referent':'two objects','property':'one is larger'},
            'conditions':[{'claim_id':'C1','evidence_ids':['E1'],'image_state':'The objects have different sizes.',
                           'coverage':'PARTIAL','assessment':'SUPPORT'}], 'reason':'The required comparison side is unclear.'}
        self.assertIsNone(C.error(value))
        result = C.project(value, 'Full source')
        self.assertEqual(result['relation'], 'UNRESOLVED')
        self.assertEqual(result['unestablished_conditions'], ['Full source'])
        value['conditions'][0]['coverage']='COMPLETE'
        value['conditions'][0]['assessment']='CONFLICT'
        self.assertEqual(C.project(value,'Full source')['relation'],'CONFLICT')
        value['reading']['kind']='UNCERTAIN'
        self.assertIsNotNone(C.error(value))


if __name__ == '__main__':
    unittest.main()
