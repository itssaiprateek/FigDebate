"""Regression contracts for observed failures; mocks test wiring, not model accuracy."""
import copy
import json
import unittest
from arbiter.arbiter import Arbiter
from engine.deliberation import deliberation_signature
from engine.independent_review import audit_independent_record, verify_independently, image_subject_hash
from engine.semantic_bridge import build_semantic_bridge
from engine.tribunal import apply_tribunal_resolution
from tests.test_semantic_bridge import ledger, contract
from tests.verification_fixture import with_independent_fixture

def case():
    review = {'status':'RESOLVE','provisional_verdict':'CONTRADICTS','best_semantic_judgment':'CONTRADICTS',
        '_format_valid':True,'_invalid_evidence_ids':[],'_valid_evidence_ids':['VF001'],
        'relation':'CONFLICT','visual_premise':'The plotted line visibly falls from left to right.',
        'caption_premise':'The plotted line rises.','semantic_bridge_type':'COMPARISON_DIRECTION',
        'semantic_bridge':'The falling line contradicts the claimed rise.',
        'counter_interpretation':'The line falls rather than rises.','counter_interpretation_strength':1.0,
        'confidence':0.1,'visual_observations':['The line falls.'],'reason':'The falling line contradicts the claimed rise.'}
    return with_independent_fixture(review,ledger(),contract())

class FactoredTribunalTests(unittest.TestCase):
    def test_source_identity_is_not_semantic_qualification(self):
        from tests.critical_fixture import core
        from engine.claim_graph import source_identity_graph, projected_requirements
        from engine.claim_contract import audit_claim_contract
        fields=core('The line rises.');fields['claim_graph']=source_identity_graph('The line rises.')
        fields['expected_visual_state'],fields['opposite_visual_state']=projected_requirements(fields['claim_graph'])
        fields['_caption_semantic_audit']={'audit_status':'UNKNOWN','_format_valid':True,'semantic_qualification':'pending'}
        c=audit_claim_contract('The line rises.',fields)
        self.assertTrue(c['source_identity_graph_valid']);self.assertFalse(c['semantic_qualified'])
        self.assertEqual(c['semantic_interpretation_status'],'UNKNOWN')
        self.assertFalse(c['fully_valid'] and c['invalid_field_groups'])
        self.assertNotIn('claim_predicate', c['field_groups']['relation']['fields'])
        self.assertNotIn('claim_subject', c['field_groups']['immutable_proposition']['fields'])
        fields['claim_graph']['nodes'][0]['interpreted_proposition']='The line falls.'
        self.assertFalse(audit_claim_contract('The line rises.',fields)['source_identity_graph_valid'])

    def test_verified_correction_reaches_final_despite_uncalibrated_self_scores(self):
        decision,entries,result=apply_tribunal_resolution({'label':'ENTAILS','confidence':0.35},case(),ledger(),contract(),
            semantic_bridge_mode='corroborated')
        self.assertTrue(result['accepted'])
        self.assertEqual(decision['label'],'CONTRADICTS')
        self.assertTrue(decision['_evidence_audit']['valid'])
        self.assertTrue(decision['_review_board']['directionally_grounded'])
        self.assertTrue(decision['_decision_trace'])

    def test_same_label_confirmation_refreshes_evidence_status(self):
        d,_,r=apply_tribunal_resolution({'label':'CONTRADICTS','confidence':0.35},case(),ledger(),contract(),semantic_bridge_mode='corroborated')
        self.assertFalse(r['accepted'])
        self.assertTrue(r['confirmation_valid'])
        self.assertTrue(d['_evidence_audit']['valid'])

    def test_failed_observation_never_promotes_or_changes(self):
        review=case();review['_independent_verification']['obligations']['visual']['verified']=False
        d,_,r=apply_tribunal_resolution({'label':'ENTAILS','confidence':0.35},review,ledger(),contract(),semantic_bridge_mode='corroborated')
        self.assertFalse(r['accepted']);self.assertEqual(d['label'],'ENTAILS')

    def test_opposing_unresolved_counter_blocks(self):
        review=case();a=review['_independent_verification']['obligations']['arguments']
        a.update(counter_relation='SUPPORT',counter_resolved=False)
        p=build_semantic_bridge(review,ledger(),contract())
        self.assertFalse(audit_independent_record(p,ledger())['valid'])

    def test_same_typed_direction_cannot_veto_itself(self):
        review=case();a=review['_independent_verification']['obligations']['arguments']
        a.update(counter_relation='CONFLICT',counter_resolved=False)
        p=build_semantic_bridge(review,ledger(),contract())
        audit=audit_independent_record(p,ledger())
        self.assertTrue(audit['valid'])
        self.assertEqual(audit['counter_resolution_basis'], 'same_typed_direction')

    def test_wrong_prose_direction_blocks_in_both_directions(self):
        for direction in ['SUPPORT','CONFLICT']:
            review=case();p=build_semantic_bridge(review,ledger(),contract());p['proposed_relation']=direction
            for c in p['independent_verification']['calls']:c['relation']=direction
            p['independent_verification']['obligations']['arguments']['bridge_relation']='SUPPORT' if direction=='CONFLICT' else 'CONFLICT'
            self.assertFalse(audit_independent_record(p,ledger())['valid'])

    def test_changed_arguments_invalidate_cached_proof(self):
        p=build_semantic_bridge(case(),ledger(),contract());self.assertTrue(audit_independent_record(p,ledger())['valid'])
        p['counter_interpretation']='A different, unresolved alternative.'
        self.assertFalse(audit_independent_record(p,ledger())['bound_to_current_case'])

    def test_missing_or_old_obligations_fail_closed(self):
        for changes in ({'schema_version':'2.0'},{'obligations':{}}):
            p=build_semantic_bridge(case(),ledger(),contract());p['independent_verification'].update(changes)
            self.assertFalse(audit_independent_record(p,ledger())['valid'])

    @unittest.expectedFailure
    def test_scoring_cannot_condition_on_assessment(self):
        # Open qualification requirement, NOT a repaired behavior: the candidate
        # removing this input regressed 6/10 to 4/10 and was rolled back.
        first=Arbiter._build_relation_choice_prompt('Caption','Visual','Language','Notes','Always SUPPORT')
        second=Arbiter._build_relation_choice_prompt('Caption','Visual','Language','Notes','Always CONFLICT')
        self.assertEqual(first,second)

    def test_unqualified_intended_meaning_is_not_scoring_input(self):
        # The isolated 20-case candidate preserved all baseline labels; the
        # structural contract is now enforced, not a claim of improved accuracy.
        text=Arbiter._summarize_language({'original_caption':'The line rises.','intended_meaning':'SECRET reversed claim'})
        self.assertNotIn('SECRET',text)

    def test_graph_audit_and_reading_changes_trigger_reconsideration(self):
        a={'language_output':{'claim_graph':{'fingerprint':'first'}},'evidence_ledger':[]}
        for b in ({'language_output':{'claim_graph':{'fingerprint':'second'}}},
                  {'language_output':{'claim_graph':{'fingerprint':'first'},'_caption_semantic_audit':{'audit_status':'FAIL'}}},
                  dict(a,debate_details={'agent2_critique':{'reading_clarification':{'answer':'An idiom of disappointment.'}}})):
            self.assertNotEqual(deliberation_signature(a),deliberation_signature(b))

    def test_unavailable_source_answer_does_not_count_as_model_comparison(self):
        from tests.critical_fixture import core
        from engine.claim_semantics import audit_core
        class Agent:
            def _generate_section(self, *args, **kwargs):
                return ({'answer':'Unresolved','source_quotes':[], 'unknown':True}, '', .01, {'schema_valid':True})
        result=audit_core(Agent(), 'The line rises.', core('The line rises.'))
        self.assertEqual(result['source_answer_call_count'], 2)
        self.assertEqual(result['model_comparison_count'], 0)
        self.assertFalse(result['semantic_checks_executed'])
        self.assertEqual(result['audit_status'], 'UNKNOWN')

    def test_failed_factored_premise_routes_to_witness_not_forced_verdict(self):
        from engine.tribunal import repair_followup_plan
        review=case();review['_independent_verification']['obligations']['visual']['verified']=False
        plan=repair_followup_plan(review)
        self.assertEqual(plan['origin'], 'failed_factored_obligations')
        self.assertEqual(plan['repair_reasons'], ['VISUAL_PREMISE'])
        self.assertTrue(plan['agent1_questions']);self.assertFalse(plan['agent2_questions'])
        self.assertEqual(plan['provisional_verdict'], 'ABSTAIN')
        # Strong self-rating does not schedule needless witness work after proof passes.
        self.assertEqual(repair_followup_plan(case()), {})

    def test_witness_cannot_endorse_a_different_source_caption(self):
        from unittest.mock import patch
        from tests.critical_fixture import core
        from engine.claim_graph import source_identity_graph, projected_requirements
        from engine.claim_witness import audit_claim_witness
        fields=core('The line rises.');fields['claim_graph']=source_identity_graph('The line rises.')
        fields['expected_visual_state'],fields['opposite_visual_state']=projected_requirements(fields['claim_graph'])
        with patch('engine.claim_witness.audit_core', return_value={'audit_status':'UNKNOWN'}):
            result=audit_claim_witness(None, 'The line falls.', {'claim_fields':fields})
        self.assertFalse(result['requirements_valid'])
        self.assertIn('WITNESS_SOURCE_CAPTION_MISMATCH', result['requirement_errors'])

    def test_runner_schedules_failed_obligation_hearing_once(self):
        from unittest.mock import patch
        from engine.batch_runner import StagewiseRunner
        review=case();review['_independent_verification']['obligations']['visual']['verified']=False
        review['_generation_seconds']=.01
        class Mediator:
            def __init__(self, *args):pass
            def review(self, *args, **kwargs):return copy.deepcopy(review)
        runner=StagewiseRunner.__new__(StagewiseRunner)
        runner.debate_mode='enabled';runner.semantic_bridge_mode='corroborated'
        sample={'index':0,'image':object(),'caption':'The plotted line rises.'}
        results={0:{'visual_output':{},'language_output':{'claim_contract':contract()},
                    'comparison':{},'evidence_ledger':ledger(), 'decision':{'label':'ENTAILS','confidence':.35},
                    'debate_details':{'agent2_requirements_valid':True}, 'timing':{}, 'judge':{}}}
        with patch('engine.batch_runner.QwenJudgeModel', return_value=object()), \
             patch('engine.batch_runner.TribunalMediatorAgent', Mediator), \
             patch('engine.batch_runner.GPUManager.clear'):
            runner._run_tribunal_review_round([sample], results, 1)
            self.assertEqual(results[0]['judge']['tribunal_session']['state'], 'FOLLOW_UP_REQUIRED')
            self.assertTrue(results[0]['judge']['verification_followup_plan']['agent1_questions'])
            runner._run_tribunal_review_round([sample], results, 2)
            self.assertNotEqual(results[0]['judge']['tribunal_session']['state'], 'FOLLOW_UP_REQUIRED')
        self.assertEqual(results[0]['decision']['label'], 'ENTAILS')

    def test_judge_packet_does_not_reintroduce_inactive_flat_claims(self):
        from engine.case_dossier import render_judge_dossier
        dossier={'source_caption':'The line rises.', 'claim_agent':{
            'caption_proposition':'The line rises.', 'claim_predicate':'UNQUALIFIED REVERSED CLAIM',
            'claim_graph':{'nodes':[]}, 'claim_contract':{'flat_predicate_authoritative':False}},
            'targeted_hearing':{'agent2_critique':{'requirements_valid':True,'_format_valid':True,
                'reading_clarification':{'answer':'UNANCHORED GUESS','source_anchored':False,'unknown':False}}}}
        packet=render_judge_dossier(dossier)
        self.assertNotIn('UNQUALIFIED REVERSED CLAIM', json.dumps(packet))
        self.assertNotIn('UNANCHORED GUESS', json.dumps(packet))
        self.assertIn('claim_graph', packet['claim_agent'])

    def test_paraphrase_preservation_call_is_text_only(self):
        from PIL import Image
        seen=[]
        class Runtime:
            _last_generation_diagnostics={}
            def generate(self,image,prompt,json_schema=None,**kwargs):
                if 'Compare ONLY these texts' in prompt:seen.append(image)
                fields=json_schema['properties']
                value=({'verified':False} if 'verified' in fields else
                    {'bridge_relation':'UNRESOLVED','bridge_grounded':False,'counter_relation':'UNRESOLVED','counter_resolved':False}
                    if 'bridge_relation' in fields else {'relation':'UNRESOLVED','evidence_ids':[]})
                return json.dumps(dict(value,reason='Explicit negative fixture.')),0.01
        image=Image.new('RGB',(2,2));p={'source_caption':'The lamp is lit.','caption_premise':'The lamp is not lit.',
              'image_sha256':image_subject_hash(image),'visual_evidence_ids':[]}
        r=verify_independently(Runtime(),image,p,[])
        self.assertEqual(seen,[None]);self.assertFalse(r['obligations']['caption']['verified'])

if __name__=='__main__':unittest.main()
