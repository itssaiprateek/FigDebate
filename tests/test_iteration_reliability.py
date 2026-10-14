"""Contracts and actual gate tests; not semantic model performance claims."""
import json
import unittest
from copy import deepcopy
from unittest.mock import Mock
from agents.visual_adapter import AtomicVisualQuestionController
from engine.reasoning_contract import completion_status, assessment_payload, explanation_alignment
from engine.evidence_review_v5 import audit
from engine.simple_judge import adapt
from tests import test_grounded_completion as gc
COMPLETE = gc.COMPLETE
from tests import test_semantic_protocol as sp


class ReliabilityTests(unittest.TestCase):
    def test_inline_citations_preserve_identity_and_reject_conflict(self):
        text = COMPLETE.replace('People look alarmed.', '[VF1] People look alarmed.').replace('Evidence IDs: VF1\n', '')
        status = completion_status(text, known={'VF1'})
        self.assertTrue(status['complete'])
        self.assertTrue(status['inline_ids_recovered'])
        self.assertFalse(completion_status(text, known={'VF2'})['complete'])
        self.assertFalse(completion_status(text + '\nEvidence IDs: VF2', known={'VF1','VF2'})['complete'])

    def test_choice_route_and_boolean_route(self):
        self.assertEqual(AtomicVisualQuestionController.infer_question_type('Is the screen fully visible or partially obscured?'), 'open')
        self.assertEqual(AtomicVisualQuestionController.infer_question_type('Is a screen visible?'), 'yes_no')

    def test_budget_projection_preserves_all_ids_and_attachments(self):
        rows=[dict(id='A1',text='NOT 347',panel='left'),dict(id='A2',text='NOT 347',panel='left'),
              dict(id='A3',text='NOT 347',panel='right')]
        p=assessment_payload('A is not larger than B.', rows)
        self.assertEqual(p['source_caption'],'A is not larger than B.')
        self.assertEqual(len(p['observations']),2)
        self.assertEqual({i for x in p['observations'] for i in x['ids']},{'A1','A2','A3'})
        self.assertTrue(all(x['text']=='NOT 347' for x in p['observations']))

    def test_input_overflow_not_retried_unchanged(self):
        fixture=gc.GroundedCompletionTests()
        a=fixture.arbiter([])
        def fail(*args,**kwargs):
            a._last_generation_diagnostics={'finish_reason':'input_budget_exceeded'}
            raise RuntimeError('Complete inputs exceed budget')
        a._generate_response=Mock(side_effect=fail)
        result=fixture.run_initial(a)
        self.assertEqual(a._generate_response.call_count,1)
        self.assertIn('input_budget_exceeded',result['explanation'])
        self.assertTrue(result['_explanation_generation_failed'])

    def test_unresolved_and_mismatched_explanations_never_verified(self):
        for relation,expected in [('UNRESOLVED','UNRESOLVED_JUSTIFICATION'),('CONFLICT','LABEL_EXPLANATION_MISMATCH')]:
            d={'label':'ENTAILS','explanation':'Relation Status: '+relation}
            out=explanation_alignment(d, verified=True)
            self.assertEqual(out['status'],expected)
            self.assertEqual(out['verification_status'],'UNVERIFIED')

    def test_missing_explanation_or_label_is_not_alignment(self):
        self.assertEqual(explanation_alignment({'label':'ENTAILS'}, verified=True)['verification_status'], 'UNVERIFIED')
        self.assertEqual(explanation_alignment({'explanation':'Some text.'})['status'], 'NOT_ASSESSED')

    def test_scoring_overflow_keeps_binary_fallback_unverified(self):
        fixture=gc.GroundedCompletionTests()
        a=fixture.arbiter([(COMPLETE, False)])
        a.grounded_interpretation=False
        a._scoring_budget_exceeded=True
        result=fixture.run_initial(a)
        self.assertTrue(result['_binary_label_is_unverified'])
        self.assertEqual(result['_relation_status'], 'INSUFFICIENT')
        self.assertIn('SCORING_INPUT_BUDGET_EXCEEDED',result['_deficiencies'])

    def test_clipped_proposal_repairs_only_failed_fields_in_existing_retry(self):
        from engine.simple_judge import proposal_parser, schema
        from agents.multimodal_judge import _run_structured_generation
        from tests.test_evidence_review_v4 import Runtime
        original=dict(interpreted_assertion='A calm meeting.', decisive_reason='x'*480,
                      evidence_ids=['VF1'], relation='CONFLICT')
        replacement='Alarmed expressions conflict with the claim of calmness.'
        runtime=Runtime([original, {'decisive_reason':replacement}])
        parser=proposal_parser('A calm meeting.',[{'id':'VF1','text':'Alarmed people.'}],None)
        result=_run_structured_generation(runtime,None,'Check source and image.',parser,
                    max_new_tokens=384,contract_name='tribunal_review',output_schema=schema({'VF1'}))
        self.assertTrue(result['_format_valid'])
        self.assertEqual(len(runtime.prompts),2)
        self.assertEqual(result['_format_retry_strategy'],'single_field_repair')
        self.assertEqual(result['_compact_value'],dict(original,decisive_reason=replacement))
        self.assertIn('ONLY',runtime.prompts[1])

    def test_completion_patch_cannot_change_relation(self):
        from engine.simple_judge import proposal_parser
        parser=proposal_parser('A calm meeting.',[{'id':'VF1','text':'Alarmed people.'}],None)
        original=dict(interpreted_assertion='A calm meeting.', decisive_reason='x'*480,
                      evidence_ids=['VF1'], relation='CONFLICT')
        parser(json.dumps(original))
        self.assertFalse(parser(json.dumps({'decisive_reason':'A completed reason.', 'relation':'SUPPORT'}))['_format_valid'])

    def test_boundary_punctuation_does_not_hide_cutoff(self):
        value=dict(interpreted_assertion='A meeting is calm.',decisive_reason='a'*479+'.',evidence_ids=['VF1'],relation='SUPPORT')
        r=adapt(json.dumps(value),'A meeting is calm.',[dict(id='VF1',text='Calm people.')],None)
        self.assertFalse(r['_format_valid'])

    def test_reinspection_routes_to_pixels_without_preferred_answer(self):
        from engine.tribunal_repair import plan_repair
        fixture=sp.SemanticProtocolTests(); fixture.setUp()
        objection={'type':'COUNTEREVIDENCE','disputed_step':'The meeting is calm.',
                   'basis':'The visible expressions look alarmed.','evidence_ids':['VF1']}
        _,review,_,_,_,_=fixture.run_case(objections=[objection],experiment='reinspect-audit-1')
        plan=plan_repair(review,{})
        self.assertEqual(plan['repair_context']['route'],'visual')
        self.assertTrue(plan['agent1_questions'])
        self.assertNotIn('alarm',plan['agent1_questions'][0].lower())
        self.assertNotIn('frozen_candidate',plan['repair_context'])

    def test_experiments_use_real_gate_and_bind_version(self):
        fixture=sp.SemanticProtocolTests()
        for experiment in ['questions-audit-1','mapping-audit-1','reinspect-audit-1','coherent-audit-1','combined-audit-1']:
            fixture.setUp()
            rt,review,bridge,ledger,_,_=fixture.run_case(experiment=experiment)
            self.assertEqual(len(rt.prompts),5)
            self.assertTrue(audit(bridge,ledger)['valid'], experiment)
            changed=deepcopy(bridge); changed['semantic_experiment']='different'
            self.assertFalse(audit(changed,ledger)['valid'])
            if 'questions' in experiment or experiment=='combined-audit-1':
                self.assertNotIn('SECRET_INITIAL_LABEL',rt.prompts[1])
                self.assertIn('deciding_check',rt.prompts[2] if 'deciding_check' in rt.prompts[2] else rt.prompts[3])

if __name__=='__main__': unittest.main()
