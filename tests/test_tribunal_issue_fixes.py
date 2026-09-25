"""Mechanism tests; synthetic fixtures do not establish semantic accuracy."""
import json
import unittest
from unittest.mock import patch
from engine.review_routing import route_question, observable_followup
from engine.question_router import build_question_plan, compile_visual_question
from engine.tribunal_process import audit_inconsistency
from engine.tribunal_repair import plan_repair
from engine.evidence_review_v5 import audit, verify
from tests.test_tribunal_v5 import v5_fixture
from tests.test_evidence_review_v4 import Runtime


class IssueFixTests(unittest.TestCase):
    def test_visible_expression_is_not_language_but_meaning_still_is(self):
        for q in ['Describe the visible facial expression.', 'Describe visible facial expressions and posture.']:
            self.assertEqual(route_question(q, 'VISUAL_PREMISE')[0], 'visual')
        self.assertEqual(route_question('What does this facial expression mean?', 'VISUAL_PREMISE')[0], 'tribunal')
        self.assertEqual(route_question('Explain the meaning of this expression in the caption.')[0], 'language')

    def test_unqualified_whole_source_specialization_is_not_active(self):
        c={'claim_contract':{'claim_graph':{'representation':'UNDECOMPOSED_SOURCE','errors':[]}},
           'claim_relation':{'subject':'a secretly furious person'}}
        for issue in ['OCR_BINDING','AFFECTIVE_SCENE','METAPHOR_MAPPING','COMPARISON_OR_OUTCOME','TEMPORAL_CAUSAL_SEQUENCE','QUOTED_STATEMENT_AND_REACTION','PRAGMATIC_SCOPE','SYMBOL_ATTACHMENT']:
            with self.subTest(issue=issue), patch('engine.question_router.classify_issue',return_value=issue):
                plan=build_question_plan(c)
                # The live specialization trial lost a useful observation.
                # Preserve the original neutral plan until a better plan qualifies.
                self.assertEqual(plan.issue_type,'SEMANTIC_RELATION')
                self.assertEqual(route_question(plan.agent1_question,'VISUAL_PREMISE')[0],'visual')
                self.assertNotIn('furious',plan.agent1_question)
                self.assertIn('complete expressed caption',plan.agent2_question)

    def test_compiler_does_not_leak_semantic_verification_to_visual(self):
        q='Explain whether the picture supports the caption.'
        self.assertNotEqual(compile_visual_question({}, {'verification_requests':[q]}),q)

    def test_observable_part_of_mixed_question_is_dispatched_without_label(self):
        q='Is there any visible text or explicit emotional expression in the image that supports the caption?'
        r={'_format_valid':True,'relation':'UNRESOLVED','targeted_question':q,
           'requested_follow_up':'COUNTER_INTERPRETATION'}
        p=plan_repair(r)
        self.assertTrue(p['agent1_questions'])
        self.assertFalse(p['verification_requests'])
        self.assertNotIn('supports',p['agent1_questions'][0])
        self.assertEqual(r['_follow_up_routing_audit'][0]['original_question'],q)
        self.assertEqual(p['repair_context']['disputed_detail'],q)

    def test_filter_does_not_suppress_existing_semantic_relevance_checks(self):
        for q in ['Is there any visual or contextual evidence supporting the assertion?',
                  'What historical context confirms this comparison?']:
            r={'_format_valid':True,'relation':'UNRESOLVED','targeted_question':q,
               'requested_follow_up':'COUNTER_INTERPRETATION'}
            self.assertTrue(plan_repair(r)['verification_requests'])

    def test_repeated_compiled_question_does_not_get_another_hearing(self):
        q='Can the location be confirmed from signage in the image?'
        r={'_format_valid':True,'relation':'UNRESOLVED','targeted_question':q,
           'requested_follow_up':'COUNTER_INTERPRETATION'}
        d={'agent1_critique':{'question_answers':[{'question':observable_followup(q)}]}}
        self.assertFalse(plan_repair(r,d))

    def test_placeholder_cannot_certify_or_stand_in_for_competing_reading(self):
        for placeholder in ['UNRESOLVED','unknown','N/A']:
            for status in ['UNRESOLVED','SAME_DIRECTION','DEFEATED']:
                self.assertTrue(audit_inconsistency({'alternative':placeholder,'alternative_status':status,
                    'alternative_relation':'CONFLICT'},'CONFLICT'))
        self.assertFalse(audit_inconsistency({'alternative':'The first person is quoting the second.',
            'alternative_status':'UNRESOLVED','alternative_relation':'UNRESOLVED'},'CONFLICT'))

    def test_actual_verification_accepts_valid_proof_and_rejects_placeholder(self):
        for bad in [False,True]:
            image,p,ledger,answers=v5_fixture()
            if bad:
                answers[3].update(alternative='UNRESOLVED',alternative_status='SAME_DIRECTION',alternative_relation='CONFLICT')
            runtime=Runtime(answers)
            p['independent_verification']=verify(runtime,image,p,ledger)
            self.assertEqual(audit(p,ledger)['valid'],not bad)

    def test_actual_visual_adapter_records_the_request_that_reaches_runtime(self):
        from agents.visual_grounding import VisualGroundingAgent
        from engine.runtime_profile import resolve_runtime_profile
        class Spy:
            processor=None;model=None;backend='qwen3_vl_4b_instruct';supports_atomic_questions=True
            hardware_profile=resolve_runtime_profile('paper-8gb-review5')
            _last_generation_diagnostics={}
            def generate(self,image,prompt,max_new_tokens=96,**kwargs):
                self.seen=(prompt,max_new_tokens)
                return 'The person raises one hand.',.001
        spy=Spy();agent=VisualGroundingAgent(spy)
        answer=agent.answer_visual_question(None,'Describe the visible hands.',question_id='witness_7')
        request=answer['generation_diagnostics']['model_bound_request']
        self.assertEqual((request['prompt'],request['max_new_tokens']),spy.seen)
        self.assertEqual(request['question_id'],'witness_7')
        self.assertNotIn('caption',request['prompt'])

    def test_dispatch_does_not_rename_a_response_from_another_question(self):
        from engine.debate import DebateEngine
        from engine.caption_answers import question_id
        engine=DebateEngine.__new__(DebateEngine);engine.global_seed=42
        engine.build_agent1_challenge_prompt=lambda *args: 'review'
        class WrongAgent:
            def critique(self,*args):return {'question_id':'different_case','question':'Different question',
                                          '_format_valid':True,'response_status':'OBSERVED'}
        case={'key':'synthetic','image':None,'caption':'A group is waiting.','visual_output':{},'decision':{},
              'tribunal_hearing':True,'mediation_plan':{'agent1_questions':['Describe visible posture.']}}
        answer=engine._run_visual_questions(WrongAgent(),case)
        self.assertFalse(answer['_format_valid'])
        self.assertEqual(answer['question_id'],'different_case')
        self.assertEqual(answer['communication']['binding_mismatches'],1)
        self.assertEqual(answer['dispatched_question_id'],question_id(case['caption'],'Describe visible posture.','agent1'))

    def test_mismatched_observation_never_enters_judge_context_or_ledger(self):
        from engine.case_dossier import _reading_view
        from engine.evidence_ledger import add_visual_witness_evidence
        response={'response_status':'QUESTION_BINDING_MISMATCH','_format_valid':False,
                  'observed_state':'A different image has a red sign.','observed_entity':'A sign',
                  'question':'An unrelated question','requested_question':'Read this image.',
                  'witness_contract':{'answer_status':'OBSERVED'}}
        view=_reading_view(response)
        self.assertNotIn('observed_state',view)
        self.assertEqual(view['content_withheld_reason'],'question_binding_mismatch')
        self.assertEqual(add_visual_witness_evidence([],response),[])

    def test_mixed_question_compiler_does_not_drop_scoped_target(self):
        self.assertEqual(observable_followup('Does visible text above the left panel support the claim?'),'')

    def test_repair_crosses_real_dispatch_and_adapter_without_losing_its_question(self):
        from agents.visual_grounding import VisualGroundingAgent
        from engine.debate import DebateEngine
        from engine.runtime_profile import resolve_runtime_profile
        from engine.evidence_ledger import add_visual_witness_evidence
        from engine.case_dossier import _reading_view
        class Spy:
            processor=None;model=None;backend='qwen3_vl_4b_instruct';supports_atomic_questions=True
            hardware_profile=resolve_runtime_profile('paper-8gb-review5')
            _last_generation_diagnostics={}
            def generate(self,image,prompt,max_new_tokens=96,**kwargs):
                self.seen=prompt
                return 'The sign reads OPEN.',.001
        original='Does visible text in the image support the caption?'
        review={'_format_valid':True,'relation':'UNRESOLVED','targeted_question':original,
                'requested_follow_up':'COUNTER_INTERPRETATION'}
        plan=plan_repair(review)
        engine=DebateEngine.__new__(DebateEngine);engine.global_seed=42
        spy=Spy()
        case={'key':'synthetic_transport','image':None,'caption':'The shop is open.',
              'visual_output':{},'decision':{},'tribunal_hearing':True,'mediation_plan':plan}
        reply=engine._run_visual_questions(VisualGroundingAgent(spy),case)
        self.assertTrue(reply['_format_valid'])
        self.assertEqual(reply['communication']['binding_mismatches'],0)
        self.assertEqual(reply['question'],plan['agent1_questions'][0])
        self.assertIn(plan['agent1_questions'][0],spy.seen)
        self.assertNotIn(case['caption'],spy.seen)
        self.assertEqual(plan['repair_context']['disputed_detail'],original)
        self.assertIn('OPEN',_reading_view(reply)['observed_state'])
        ledger=add_visual_witness_evidence([],reply)
        self.assertEqual(len(ledger),1)
        self.assertFalse(ledger[0]['decision_grade'])

if __name__=='__main__':unittest.main()
