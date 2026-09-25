"""General mechanism controls, never production rules or dataset accuracy claims."""
from copy import deepcopy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

from engine import evidence_review_v5 as v5
from engine import tribunal_process as process
from engine.output_contracts import validate_shape
from engine.review_outcome import classify_review
from engine.tribunal_repair import scheduled_repair, validate_options, plan_repair
from tests.test_tribunal_v5 import v5_fixture
from tests.test_evidence_review_v4 import Runtime


class TribunalChangeTests(unittest.TestCase):
    def proof(self, candidate=False, mutate=None, spans=False):
        image, proposal, ledger, answers = v5_fixture()
        if candidate:
            proposal['process_audit_version'] = process.VERSION
            answers[3]['assessed_interpretation'] = 'The stated calm meeting contrasts with alarmed participants.'
            answers[3]['process_checks'] = {key: {'status': 'PASS', 'evidence_ids': ['VF1'],
                'reason': 'Alarmed participants oppose the same meeting being calm.'} for key in process.CHECKS}
        if mutate:
            mutate(proposal, answers)
        runtime = Runtime(answers)
        proposal['independent_verification'] = v5.verify(runtime, image, proposal, ledger)
        return runtime, proposal, ledger

    def test_schema_matches_required_citations_without_changing_shared_schema(self):
        schema = v5.mapping_schema({'VF1'})
        self.assertEqual(schema['properties']['bindings']['items']['properties']['evidence_ids']['minItems'], 1)
        self.assertEqual(v5.core.MAPPING['properties']['bindings']['minItems'], 1)
        value = v5_fixture()[-1][1]
        self.assertTrue(validate_shape(value, schema))
        value['bindings'][0]['evidence_ids'] = []
        self.assertFalse(validate_shape(value, schema))

    def test_entirely_unmatched_mapping_is_honest_semantic_stop(self):
        def mutate(p, a):
            a[1].update(bindings=[], unmatched_roles=['No participant can be identified.'])
        rt, p, ledger = self.proof(mutate=mutate)
        proof = p['independent_verification']
        self.assertTrue(proof['obligations']['mapping']['_format_valid'])
        self.assertEqual(proof['stopped_after'], 'entity_scope_mapping')
        self.assertEqual(len(rt.prompts), 2)
        self.assertFalse(v5.audit(p, ledger)['valid'])

    def test_empty_mapping_without_explanation_cannot_pass(self):
        self.assertIsNotNone(v5.mapping_error({'bindings': [], 'unmatched_roles': []}, 'Any source.', {'VF1'}))

    def test_candidate_retains_blind_relation_and_complete_expression(self):
        rt, p, ledger = self.proof(True)
        self.assertTrue(v5.audit(p, ledger)['valid'])
        for prompt in rt.prompts[:3]:
            payload = json.loads(prompt.rsplit('\n', 1)[-1])
            self.assertNotIn('proposed_relation', payload)
            self.assertNotIn('draft_argument', payload)
            self.assertEqual(payload['task']['proposition'], p['source_caption'])
        relation = json.loads(rt.prompts[2].rsplit('\n', 1)[-1])
        self.assertEqual(relation['complete_source_expressions'][0]['complete_expression'], p['source_caption'])

    def test_baseline_keeps_four_calls_without_process_fields(self):
        rt, p, ledger = self.proof()
        self.assertTrue(v5.audit(p, ledger)['valid'])
        self.assertEqual(len(rt.prompts), 4)
        self.assertNotIn('process_checks', rt.prompts[-1])
        self.assertEqual(rt.token_budgets[-1], 512)

    def test_unsound_same_label_process_audit_cannot_pass(self):
        for status in ('FAIL', 'UNRESOLVED'):
            def mutate(p, a):
                a[3]['process_checks']['relation_inference']['status'] = status
            _, p, ledger = self.proof(True, mutate)
            self.assertFalse(v5.audit(p, ledger)['valid'])

    def test_process_response_mutation_and_version_downgrade_rejected(self):
        _, p, ledger = self.proof(True)
        edited = deepcopy(p)
        edited['independent_verification']['obligations']['arguments']['process_checks']['interpretation']['reason'] = 'Changed.'
        self.assertFalse(v5.audit(edited, ledger)['valid'])
        edited = deepcopy(p); edited.pop('process_audit_version')
        self.assertFalse(v5.audit(edited, ledger)['valid'])

    def test_inner_failure_exposes_stage_and_retains_error_type(self):
        review = dict(_format_valid=True, _execution_status='SUCCEEDED', provisional_verdict='CONTRADICTS',
            _independent_verification={'schema_version':'5.0', 'obligations': {'visual': {
                '_execution_status':'FAILED', '_format_valid':False, '_execution_error_type':'TIMEOUT'}}, 'calls':[]})
        result = classify_review(review)
        self.assertEqual(result['first_blocking_stage'], 'visual')
        self.assertEqual(result['stage_outcomes'][1]['error_type'], 'TIMEOUT')
        self.assertEqual(result['stage_outcomes'][2]['status'], 'NOT_RUN')

    def test_truncation_is_not_abstention(self):
        result = classify_review(dict(_format_valid=False, _execution_status='SUCCEEDED', _output_status='TRUNCATED'))
        self.assertEqual(result['stage_outcomes'][0]['error_type'], 'OUTPUT_TRUNCATED')
        self.assertFalse(result['semantic_abstained'])

    def unresolved(self):
        return dict(_format_valid=True, relation='UNRESOLVED', requested_follow_up='COUNTER_INTERPRETATION',
                    targeted_question='Does the visible state contradict the source condition?')

    def test_repair_independent_of_feedback_and_bounded(self):
        review = self.unresolved()
        self.assertEqual(scheduled_repair(review, {}, {}), {})
        plan = scheduled_repair(review, {}, {}, mode='bounded')
        self.assertEqual(plan['origin'], 'tribunal_only_bounded_repair_v1')
        self.assertTrue(plan['verification_requests'])
        self.assertEqual(scheduled_repair(review, {}, {}, mode='bounded', round_number=2), {})
        self.assertEqual(scheduled_repair(review, {}, {'accepted':True}, mode='bounded'), {})
        review['_case_budget'] = {'remaining_seconds': 1}
        self.assertEqual(scheduled_repair(review, {}, {}, mode='bounded'), {})

    def test_unavailable_information_does_not_force_answer(self):
        review = self.unresolved()
        review.update(_process_audit_version=process.VERSION,
                      targeted_question='Check historical records for the original design.')
        plan = plan_repair(review)
        self.assertTrue(plan['verification_requests'])
        self.assertFalse(plan['agent1_questions'] or plan['agent2_questions'])
        self.assertEqual(review['relation'], 'UNRESOLVED')

    def test_uncertainty_relevance_preserves_honest_unknown(self):
        review = self.unresolved(); review['_process_audit_version'] = process.VERSION
        plan = plan_repair(review)
        self.assertTrue(plan['repair_context']['uncertainty_relevance_check'])
        self.assertIn('If yes or unclear, retain uncertainty', plan['repair_context']['instruction'])
        self.assertFalse(plan['agent1_questions'])

    def test_configuration_rejects_mixed_feedback_and_wrong_protocol(self):
        validate_options('bounded', process.VERSION, 'tribunal', 'enabled', 'evidence-review-5.0', 'disabled')
        for feedback, protocol in [('integrated','evidence-review-5.0'), ('disabled','evidence-review-4.0')]:
            with self.assertRaises(ValueError):
                validate_options('bounded', process.VERSION, 'tribunal', 'enabled', protocol, feedback)

    def test_changes_to_interpretation_invalidate_proof(self):
        _, p, ledger = self.proof(True)
        p['bridge_statement'] = 'A different interpretation.'
        self.assertFalse(v5.audit(p, ledger)['valid'])

    def test_cli_options_and_runner_store_separate_policies(self):
        import run_figdebate
        from engine.batch_runner import StagewiseRunner
        with patch('sys.argv', ['run_figdebate.py', '--tribunal-repair-mode', 'bounded',
                               '--tribunal-audit-mode', process.VERSION, '--feedback-mode', 'disabled']):
            args = run_figdebate.parse_args()
        runner = StagewiseRunner(tribunal_repair_mode=args.tribunal_repair_mode,
            tribunal_audit_mode=args.tribunal_audit_mode, feedback_mode='disabled',
            judge_mode='tribunal', hardware_profile='paper-8gb-review5')
        self.assertEqual(runner.tribunal_repair_mode, 'bounded')
        self.assertEqual(runner.feedback_mode, 'disabled')

    def test_actual_followup_scheduler_does_not_send_semantics_to_witness(self):
        from engine.batch_runner import StagewiseRunner
        runner = StagewiseRunner(tribunal_repair_mode='bounded', feedback_mode='disabled',
                                judge_mode='tribunal', hardware_profile='paper-8gb-review5')
        review = self.unresolved()
        plan = scheduled_repair(review, {}, {}, mode='bounded')
        results = {0: {'judge': {'tribunal_session': {'state':'FOLLOW_UP_REQUIRED'},
            'tribunal_reviews':[review], 'verification_followup_plan':plan},
            'evidence_ledger':[], 'language_output':{}, 'debate_details':{}}}
        runner.debate.run_debate_batch = Mock(side_effect=AssertionError('Semantic question sent to a witness'))
        runner._run_tribunal_review_round = Mock(return_value=0)
        samples=[{'index':0}]
        runner._run_tribunal_followups(samples,results)
        runner.debate.run_debate_batch.assert_not_called()
        runner._run_tribunal_review_round.assert_called_once_with(samples,results,2)
        self.assertTrue(results[0]['debate_details']['tribunal_semantic_questions']['new_resolving_check'])

    def test_first_blocking_stage_includes_disagreement(self):
        _, p, _ = self.proof()
        review = dict(_format_valid=True, _execution_status='SUCCEEDED', relation='SUPPORT',
                      provisional_verdict='ENTAILS', _independent_verification=p['independent_verification'])
        self.assertEqual(classify_review(review)['first_blocking_stage'], 'relation')

    def test_candidate_cache_does_not_reuse_baseline_proof(self):
        image, p, ledger, answers = v5_fixture()
        runtime=Runtime(answers)
        v5.verify(runtime,image,p,ledger)
        p['process_audit_version']=process.VERSION
        candidate=deepcopy(answers)
        candidate[3].update(assessed_interpretation='The calmness assertion conflicts with visible alarm.',
            process_checks={k:dict(status='PASS',evidence_ids=['VF1'],reason='Visible alarm conflicts with calmness.') for k in process.CHECKS})
        runtime.answers=candidate
        proof=v5.verify(runtime,image,p,ledger)
        self.assertFalse(proof['obligations']['visual'].get('_cache_hit',False))
        self.assertEqual(len(runtime.prompts),8)

    def test_visual_followup_uses_witness_path_and_requires_new_information(self):
        from engine.batch_runner import StagewiseRunner
        runner = StagewiseRunner(tribunal_repair_mode='bounded', feedback_mode='disabled',
                                judge_mode='tribunal', hardware_profile='paper-8gb-review5')
        review = dict(_format_valid=True,relation='UNRESOLVED',requested_follow_up='VISUAL_PREMISE',
                      targeted_question='Read the exact visible words above the person.')
        plan=scheduled_repair(review,{}, {},mode='bounded')
        result={'judge':{'tribunal_session':{'state':'FOLLOW_UP_REQUIRED'},
                        'tribunal_reviews':[review],'verification_followup_plan':plan},
                'evidence_ledger':[],'visual_output':{},'language_output':{},'comparison':{},
                'decision':{'label':'ENTAILS'},'debate_details':{}}
        results={0:result};samples=[{'index':0,'image':None,'caption':'A generic caption.'}]
        runner.debate.run_debate_batch=Mock(return_value={})
        runner.debate.last_batch_timing={}
        runner._run_tribunal_review_round=Mock(return_value=0)
        with patch.object(runner,'_merge_debate_results'):
            runner._run_tribunal_followups(samples,results)
        cases=runner.debate.run_debate_batch.call_args.args[0]
        self.assertEqual(cases[0]['mediation_plan']['agent1_questions'],[review['targeted_question']])
        self.assertFalse(cases[0]['mediation_plan']['verification_requests'])
        self.assertEqual(result['judge']['tribunal_session']['state'],'PRESERVED')
        runner._run_tribunal_review_round.assert_called_once_with([],results,2)

    def test_candidate_process_failure_rejected_by_actual_gate(self):
        from engine.tribunal import apply_tribunal_resolution
        from engine.semantic_bridge import build_semantic_bridge
        from engine.independent_review import image_subject_hash
        from tests.test_semantic_bridge import contract, ledger as entries
        from tests.test_factored_tribunal import case
        from PIL import Image
        image=Image.new('RGB',(3,3)); review=case(); data=entries()
        review.update(_protocol='evidence-review-5.0',_process_audit_version=process.VERSION,
            relation='CONFLICT', provisional_verdict='CONTRADICTS', best_semantic_judgment='CONTRADICTS',
            visual_premise=data[0]['text'], visual_observations=[data[0]['text']],
            semantic_bridge='The falling line contradicts the claimed rise.',
            reason='The falling line contradicts the claimed rise.', counter_interpretation='',
            _case_image_sha256=image_subject_hash(image))
        proposal=build_semantic_bridge(review,data,contract())
        _,_,_,a=v5_fixture()
        a[0]['observations'][0]['evidence_id']='VF001'
        a[1]['bindings'][0].update(caption_quote='line',evidence_ids=['VF001'])
        a[2].update(evidence_ids=['VF001'],condition_checks=[{'caption_quote':'rises','image_state':'The line falls.','relation':'CONFLICT'}])
        a[3].update(deciding_evidence_ids=['VF001'],assessed_interpretation='The plotted line rises.',
            process_checks={k:dict(status='PASS',evidence_ids=['VF001'],reason='The same plotted line falls.') for k in process.CHECKS})
        a[3]['process_checks']['relation_inference']['status']='UNRESOLVED'
        review['_independent_verification']=v5.verify(Runtime(a),image,proposal,data)
        decision,_,gate=apply_tribunal_resolution({'label':'ENTAILS','confidence':.35},review,data,
            contract(),semantic_bridge_mode='corroborated',source_caption=contract()['source_caption'])
        self.assertFalse(gate['accepted'])
        self.assertEqual(decision['label'],'ENTAILS')
        a[3]['process_checks']['relation_inference']['status']='PASS'
        review['_independent_verification']=v5.verify(Runtime(a),image,proposal,data)
        decision,_,gate=apply_tribunal_resolution({'label':'ENTAILS','confidence':.35},review,data,
            contract(),semantic_bridge_mode='corroborated',source_caption=contract()['source_caption'])
        self.assertTrue(gate['accepted'],gate['reason'])
        self.assertEqual(decision['label'],'CONTRADICTS')

    def test_source_span_candidate_retains_valid_complete_proof(self):
        image,p,data,a=v5_fixture()
        p['process_audit_version']=process.VERSION
        a[1]['bindings'][0].pop('caption_quote')
        a[1]['bindings'][0]['caption_span']={'start':1,'end':2}
        a[2]['condition_checks'][0].pop('caption_quote')
        a[2]['condition_checks'][0]['caption_span']={'start':3,'end':4}
        a[3].update(assessed_interpretation='The meeting is calm.',process_checks={
            k:dict(status='PASS',evidence_ids=['VF1'],reason='Visible alarm contradicts calmness.') for k in process.CHECKS})
        rt=Runtime(a);rt.hardware_profile=SimpleNamespace(judge_source_spans=True)
        p['independent_verification']=v5.verify(rt,image,p,data)
        self.assertTrue(v5.audit(p,data)['valid'])

    def test_candidate_flags_invalidate_only_downstream_checkpoint_settings(self):
        from pathlib import Path
        from engine.cache_identity import stage_fingerprints
        root=Path(__file__).resolve().parents[1]
        config={'seed':42,'pipeline_source_sha256':'fixed', 'feedback_mode':'disabled',
                'ablation_signature':{'tribunal_repair':'disabled','tribunal_audit':'baseline'}}
        first=stage_fingerprints(root,config)
        changed=deepcopy(config);changed['ablation_signature']['tribunal_audit']=process.VERSION
        second=stage_fingerprints(root,changed)
        self.assertEqual(first['initial_reasoning'],second['initial_reasoning'])
        self.assertNotEqual(first['default'],second['default'])

    def test_contradictory_audit_has_one_specific_recheck_without_normalization(self):
        # Legacy archived contradictions still get an explicit hearing plan.
        # New generation catches these inside its bounded clarification instead.
        rt,p,data=self.proof()
        p['independent_verification']['obligations']['arguments']['alternative']='A competing reading.'
        review=dict(_format_valid=True,relation='CONFLICT',_independent_verification=p['independent_verification'])
        before=deepcopy(review)
        plan=scheduled_repair(review,{}, {},mode='bounded')
        self.assertEqual(plan['repair_context']['failed_requirement'],'contradictory_audit')
        self.assertEqual(review,before)
        self.assertEqual(classify_review(review)['stage_outcomes'][-1]['status'],'CONTRADICTORY_AUDIT')
        self.assertFalse(plan['agent1_questions'] or plan['agent2_questions'])
        self.assertEqual(scheduled_repair(review,{}, {},mode='bounded',round_number=2),{})

    def test_audit_only_recheck_reuses_unchanged_relation_without_forcing_acceptance(self):
        image,p,data,a=v5_fixture()
        original=deepcopy(a)
        rt=Runtime(a); first=v5.verify(rt,image,p,data)
        p['verification_task']={'failed_requirement':'contradictory_audit','question':'Reconcile the alternative fields from evidence.'}
        rt.answers=[original[3]]
        second=v5.verify(rt,image,p,data)
        self.assertEqual(len(rt.prompts),5)
        self.assertTrue(second['calls'][0]['_cache_hit'])
        self.assertTrue(second['obligations']['mapping']['_cache_hit'])
        p['independent_verification']=second
        self.assertTrue(v5.audit(p,data)['valid'])


if __name__ == '__main__':
    unittest.main()
