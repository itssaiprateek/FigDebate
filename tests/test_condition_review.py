"""Real gate and bounded-recovery regressions; scripted semantics are not accuracy evidence."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from engine import condition_review as cr
from engine.evidence_review_v5 import audit, mapping_schema, mapping_response_valid
from engine.evidence_verification import obligation_runner
from engine.mapping_completion import mapping_reason_prefix
from engine.tribunal import apply_tribunal_resolution
from engine.tribunal_repair import plan_repair, scheduled_repair
from engine.source_spans import SourceSpans
from tests import test_semantic_protocol as helpers
from tests.test_evidence_review_v4 import Runtime, fixture


class ConditionTests(unittest.TestCase):
    def run_case(self, relation='CONFLICT', objections=None, mode='condition-audit-1'):
        helper = helpers.SemanticProtocolTests()
        helper.setUp()
        return helper.run_case(relation, objections=objections, experiment=mode)

    def test_real_gate_both_directions_and_blind_stages(self):
        for relation, label in [('SUPPORT','ENTAILS'),('CONFLICT','CONTRADICTS')]:
            rt, review, bridge, ledger, language, source = self.run_case(relation)
            self.assertTrue(audit(bridge, ledger)['valid'])
            final, _, gate = apply_tribunal_resolution({'label':'CONTRADICTS' if label=='ENTAILS' else 'ENTAILS'},
                review, ledger, language['claim_contract'], semantic_bridge_mode='corroborated', source_caption=source)
            self.assertTrue(gate['accepted'])
            self.assertEqual(final['label'],label)
            self.assertEqual(len(rt.prompts),5)
            for prompt in rt.prompts:
                self.assertNotIn('SECRET_INITIAL_LABEL', prompt)
                self.assertIn(source, prompt)
            for prompt in rt.prompts[1:4]:
                self.assertNotIn('"candidate":',prompt)

    def test_mutating_any_authoritative_or_derived_field_invalidates_gate(self):
        _, _, base, ledger, _, _ = self.run_case()
        mutations = [lambda d:d.update(relation='SUPPORT'),
            lambda d:d['condition_checks'][0].update(caption_quote='calm'),
            lambda d:d.update(unestablished_conditions=['calm']),
            lambda d:d['conditions'][0].update(coverage='PARTIAL'),
            lambda d:d['_claim_target'].update(source_caption='A different claim.'),
            lambda d:d.pop('_condition_protocol'),
            lambda d:d.update(reason='A new unaudited argument.')]
        for mutate in mutations:
            bridge=deepcopy(base); mutate(bridge['independent_verification']['calls'][0])
            self.assertFalse(audit(bridge,ledger)['valid'])

    def test_partial_coverage_can_never_be_promoted_to_support(self):
        _, _, bridge, _, _, source = self.run_case('SUPPORT')
        value=bridge['independent_verification']['calls'][0]['_condition_value']
        for coverage in ('PARTIAL','UNCLEAR'):
            value['conditions'][0]['coverage']=coverage
            result=cr.project(value,source)
            self.assertEqual(result['relation'],'UNRESOLVED')
            self.assertEqual(result['unestablished_conditions'],[source])

    def test_whole_source_is_not_replaced_by_a_model_chosen_clause(self):
        _,_,bridge,_,_,_=self.run_case()
        value=bridge['independent_verification']['calls'][0]['_condition_value']
        for source in ('The first person left, but the second did not.',
                       'Either the first object is red or the second object is blue.',
                       'Before the change she disliked it; afterwards she liked it.'):
            projected=cr.project(value,source)
            self.assertEqual(projected['condition_checks'][0]['caption_quote'],source)
            self.assertFalse(cr.task(source)['decomposition_semantically_verified'])

    def test_unknown_reading_cannot_certify_direction(self):
        _, _, bridge, _, _, _ = self.run_case()
        value=bridge['independent_verification']['calls'][0]['_condition_value']
        value['reading']['kind']='UNCERTAIN'
        self.assertEqual(cr.error(value)['kind'],'SEMANTIC_INCONSISTENCY')

    def test_new_candidate_for_candidate_objection_and_bound_claim(self):
        objection=dict(type='UNSUPPORTED_INFERENCE',disputed_step='The meeting is calm.',
                       basis='The people look alarmed.',evidence_ids=['VF1'])
        _,review,bridge,ledger,_,source=self.run_case(objections=[objection],mode='condition-repair-1')
        self.assertFalse(audit(bridge,ledger)['valid'])
        debate={}; plan=plan_repair(review,debate)
        self.assertTrue(plan)
        self.assertNotIn('frozen_candidate',plan['repair_context'])
        self.assertEqual(plan['repair_context']['immutable_claim'],cr.task(source))
        self.assertFalse(plan_repair(review,debate))
        self.assertFalse(scheduled_repair(review,{}, {}, mode='bounded', round_number=2))

    def test_regenerated_candidate_reenters_real_proof_and_cannot_reuse_old_certificate(self):
        from agents.multimodal_judge import TribunalMediatorAgent
        from engine.semantic_bridge import build_semantic_bridge
        from engine.runtime_accounting import begin_accounting
        objection=dict(type='UNSUPPORTED_INFERENCE',disputed_step='The meeting is calm.',
                       basis='The people look alarmed.',evidence_ids=['VF1'])
        _,old,_,ledger,language,source=self.run_case('SUPPORT',objections=[objection],mode='condition-repair-1')
        plan=plan_repair(old,{})
        _,good,_,_,_,_=self.run_case('CONFLICT',mode='condition-repair-1')
        proof=good['_independent_verification']; obs=proof['obligations']
        answers=[good['_compact_value']]+[json.loads(x['_raw_output']) for x in
                    (obs['visual'],obs['mapping'],proof['calls'][0],obs['arguments'])]
        rt=Runtime(answers); rt.hardware_profile=SimpleNamespace(tribunal_protocol='evidence-review-5.0')
        rt.tribunal_audit_mode='condition-repair-1'; begin_accounting()
        image=fixture()[0]
        new=TribunalMediatorAgent(rt).review(image,source,{},language,{},ledger,{},
                    current_decision=None,_verification_repair=plan['repair_context'])
        self.assertEqual(len(rt.prompts),5)
        bridge=build_semantic_bridge(new,ledger,language['claim_contract'],language)
        self.assertTrue(audit(bridge,ledger)['valid'])
        self.assertFalse(new.get('_proposal_reused_for_audit',False))
        bridge['independent_verification']=old['_independent_verification']
        self.assertFalse(audit(bridge,ledger)['bound_to_current_case'])

    def test_candidate_scope_objection_also_allows_one_new_candidate(self):
        objection=dict(type='SCOPE_ERROR',disputed_step='Candidate replaces the caption property.',
                       basis='The assertion concerns calmness.',evidence_ids=['VF1'])
        helper=helpers.SemanticProtocolTests();helper.setUp()
        _,review,*_=helper.run_case(objections=[objection],alignment='MISMATCH',experiment='condition-repair-1')
        plan=plan_repair(review,{})
        self.assertTrue(plan)
        self.assertNotIn('frozen_candidate',plan['repair_context'])

    def test_baseline_preserves_existing_frozen_policy(self):
        objection=dict(type='UNSUPPORTED_INFERENCE',disputed_step='The meeting is calm.',
                       basis='The people look alarmed.',evidence_ids=['VF1'])
        helper=helpers.SemanticProtocolTests(); helper.setUp()
        _,review,*_=helper.run_case(objections=[objection])
        self.assertIn('frozen_candidate',plan_repair(review,{})['repair_context'])

    def test_exact_final_explanation_is_bound_separately_from_label(self):
        from engine.final_artifact import final_artifact
        _, review, _, ledger, language, source = self.run_case()
        final, updated, gate = apply_tribunal_resolution({'label':'ENTAILS','explanation':'Original.'},
            review, ledger, language['claim_contract'], semantic_bridge_mode='corroborated', source_caption=source)
        result=dict(decision=final,evidence_ledger=updated,judge=dict(tribunal_resolution=gate))
        artifact=final_artifact(source,result)
        self.assertTrue(artifact['audited_explanation']['label_verified'])
        self.assertTrue(artifact['audited_explanation']['exact_audited_explanation'])
        final['explanation']='A different complete but unaudited rationale.'
        artifact=final_artifact(source,result)
        self.assertTrue(artifact['audited_explanation']['label_verified'])
        self.assertFalse(artifact['audited_explanation']['exact_audited_explanation'])
        altered=final_artifact(source+' Altered.',result)
        self.assertFalse(altered['audited_explanation']['label_verified'])

    def test_oracle_substitutions_are_source_bound_and_single_stage(self):
        from evaluation.stage_substitution import substitute_stage
        image, p, ledger, _ = fixture(); source=p['source_caption']
        _,_,bridge,_,_,_=self.run_case()
        wire=bridge['independent_verification']['calls'][0]['_condition_value']
        rt=Runtime([{'ordinary':'model response'}])
        entry=dict(source_sha256=cr.task(source)['source_sha256'],stage='conditions',
                   value=wire,reviewed_by='Test fixture author',basis='Synthetic fixture, not model accuracy.')
        original=rt.generate
        with substitute_stage(rt,entry,source) as telemetry:
            raw,_=rt.generate(image,'Data',json_schema=cr.schema({'VF1'}))
            self.assertEqual(json.loads(raw),wire)
            self.assertEqual(telemetry['substitutions'],1)
            self.assertEqual(rt.prompts,[])
            with self.assertRaises(ValueError):
                rt.generate(image,'Data',json_schema=cr.schema({'VF1'}))
        self.assertEqual(rt.generate,original)
        with self.assertRaises(ValueError):
            with substitute_stage(rt,entry,source+' Changed.'):
                pass

    def test_comparator_policy_is_opt_in_and_preserves_source(self):
        from comparators.evidence_comparator import compare
        source='This room is a palace.'
        language={'figurative_type':'metaphor','caption_proposition':source}
        baseline=compare({},language,source)
        candidate=compare({},language,source,preserve_assertion=True)
        self.assertIn("Compare the intended meaning, not only the caption's literal wording.",baseline['review_questions'])
        self.assertTrue(any('full asserted property' in x for x in candidate['review_questions']))
        self.assertEqual(baseline['caption'],candidate['caption'])

    def test_completion_preserves_complete_mapping_prefix_through_real_validator(self):
        image,p,ledger,_=fixture(); source=p['source_caption']; known={'VF1'}
        schema=mapping_schema(known); wire_schema=SourceSpans(source).schema(schema)
        prefix=dict(bindings=[dict(caption_span={'start':1,'end':2},observed_entity='People at the table',
            role_scope='Same meeting',evidence_ids=['VF1'])],unmatched_roles=[])
        raw=json.dumps(prefix)[:-1]+', "reason":"The meeting corresponds to'
        class RawRuntime(Runtime):
            def generate(self,image,prompt,max_new_tokens=None,json_schema=None):
                self.prompts.append(prompt); self.token_budgets.append(max_new_tokens)
                return self.answers.pop(0), .01
        rt=RawRuntime([raw,json.dumps({'replacement':'Both refer to the same meeting.'})])
        rt.hardware_profile=SimpleNamespace(judge_source_spans=True)
        call=obligation_runner(rt,image,source,ledger,known,'5.0')('mapping','Bind roles.',{},schema,320)
        self.assertTrue(mapping_response_valid(call,known,source))
        reconstructed=json.loads(call['_raw_output'])
        self.assertEqual(reconstructed['bindings'],prefix['bindings'])
        self.assertEqual(reconstructed['unmatched_roles'],[])
        self.assertEqual(len(rt.prompts),2)
        self.assertEqual(rt.token_budgets[-1],160)
        self.assertEqual(call['_field_repair_audit']['truncated_raw'],raw)
        for bad in [raw[:raw.index('role_scope')],raw.replace('"unmatched_roles": []','"unexpected": []'),
                    raw.replace('"reason":','"bindings": [], "reason":'),json.dumps(dict(prefix,reason='Complete.'))]:
            self.assertIsNone(mapping_reason_prefix(bad,wire_schema))

if __name__=='__main__': unittest.main()
