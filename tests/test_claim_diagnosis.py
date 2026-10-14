from copy import deepcopy
import unittest
from unittest.mock import patch
from evaluation.structural_variants import enable_evaluation_modes
from evaluation.claim_pair_probe import candidate_context
from engine import condition_review as cr
from engine.evidence_review_v5 import audit
from engine.tribunal import apply_tribunal_resolution
from tests import test_semantic_protocol as helper
from tests.test_evidence_review_v4 import Runtime

class CanonicalRuntime(Runtime):
    def generate(self,image,prompt,max_new_tokens=None,json_schema=None):
        if 'conditions' in (json_schema or {}).get('properties',{}):
            row=self.answers[0]['conditions'][0]
            if 'state_pairs' in row:
                row['image_state']=row.pop('state_pairs')[0]['actual']
                row.pop('connection')
        return super().generate(image,prompt,max_new_tokens,json_schema)

class ClaimDiagnosisTests(unittest.TestCase):
    def test_canonical_contract_reaches_real_gate_both_directions(self):
        from engine import semantic_protocol as sp
        with patch.dict(sp.EXPERIMENTS), patch.object(sp,'MODES',set(sp.MODES)):
            enable_evaluation_modes()
            for relation,expected,initial in [('SUPPORT','ENTAILS','CONTRADICTS'),('CONFLICT','CONTRADICTS','ENTAILS')]:
                h=helper.SemanticProtocolTests();h.setUp()
                with patch.object(helper,'Runtime',CanonicalRuntime):
                    rt,review,bridge,ledger,lang,source=h.run_case(relation,experiment='canonical-audit-1')
                self.assertTrue(audit(bridge,ledger)['valid'])
                final,_,gate=apply_tribunal_resolution({'label':initial},review,ledger,lang['claim_contract'],semantic_bridge_mode='corroborated',source_caption=source)
                self.assertTrue(gate['accepted']);self.assertEqual(final['label'],expected)
                altered=deepcopy(bridge);altered['independent_verification']['calls'][0]['_condition_protocol']=cr.VERSION
                self.assertFalse(audit(altered,ledger)['valid'])
                altered=deepcopy(bridge);altered['independent_verification']['calls'][0]['reason']='Unaudited replacement.'
                self.assertFalse(audit(altered,ledger)['valid'])
                self.assertEqual(len(rt.prompts),5)
    def test_evaluation_context_changes_order_not_fields_and_restores(self):
        old=cr.schema({'VF1'},True)
        from engine import simple_judge
        fn=simple_judge.prompt
        with candidate_context('paired'):
            new=cr.schema({'VF1'},True)
            def normalize(v):
                if isinstance(v,dict):return {k:sorted(x) if k=='required' else normalize(x) for k,x in v.items()}
                if isinstance(v,list):return [normalize(x) for x in v]
                return v
            self.assertEqual(normalize(old),normalize(new))
            self.assertEqual(list(new['properties']),['reading','reason','conditions'])
            self.assertIsNot(simple_judge.prompt,fn)
        self.assertIs(simple_judge.prompt,fn)
        self.assertEqual(list(cr.schema({'VF1'},True)['properties']),list(old['properties']))

    def test_defined_followup_preserves_existing_field_definitions(self):
        from engine import simple_judge, semantic_protocol
        original = simple_judge.prompt
        with candidate_context('paired_defined'):
            prompt = simple_judge.prompt({'source_caption': 'Example assertion.'})
            self.assertIn(semantic_protocol.READING_INSTRUCTIONS, prompt)
            self.assertIn(semantic_protocol.READING_INSTRUCTIONS, cr.CANONICAL_INSTRUCTIONS)
        self.assertIs(simple_judge.prompt, original)

    def test_confirmation_replaces_unresolved_reason_only_after_valid_gate(self):
        h=helper.SemanticProtocolTests();h.setUp()
        _,review,_,ledger,language,source=h.run_case('SUPPORT')
        initial={'label':'ENTAILS','explanation':'The relation remains unresolved.',
                 '_semantic_relation_status':'UNRESOLVED','_explanation_generation_failed':False}
        final,_,gate=apply_tribunal_resolution(deepcopy(initial),review,ledger,language['claim_contract'],semantic_bridge_mode='corroborated',source_caption=source)
        self.assertTrue(gate['confirmation_valid']);self.assertFalse(gate['accepted'])
        self.assertEqual(final['label'],initial['label'])
        self.assertEqual(final['explanation'],review['reason'])
        self.assertEqual(final['_previous_unverified_explanation'],initial['explanation'])
        bad=deepcopy(review);bad['_format_valid']=False
        final,_,gate=apply_tribunal_resolution(deepcopy(initial),bad,ledger,language['claim_contract'],semantic_bridge_mode='corroborated',source_caption=source)
        self.assertEqual(final['explanation'],initial['explanation'])
        initial['_semantic_relation_status']='SUPPORT'
        final,_,gate=apply_tribunal_resolution(deepcopy(initial),review,ledger,language['claim_contract'],semantic_bridge_mode='corroborated',source_caption=source)
        self.assertEqual(final['explanation'],initial['explanation'])
