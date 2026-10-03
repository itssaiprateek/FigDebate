"""Exercise the real review and acceptance path, not model accuracy."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from engine.simple_judge import review
from engine.semantic_bridge import build_semantic_bridge
from engine.evidence_review_v5 import audit
from engine.tribunal import apply_tribunal_resolution
from engine.runtime_accounting import begin_accounting
from tests.test_evidence_review_v4 import Runtime
from tests.test_tribunal_v5 import v5_fixture
from tests.test_semantic_bridge import contract


class AuditOnlyTests(unittest.TestCase):
    def run_case(self, mode='audit-only-1', objection=False):
        begin_accounting()
        image,proposal,ledger,old=v5_fixture();source=proposal['source_caption']
        first=dict(interpreted_assertion=source,decisive_reason='Alarmed expressions conflict with a calm meeting.',
                   evidence_ids=['VF1'],relation='CONFLICT')
        objections=[dict(type='UNSUPPORTED_INFERENCE',disputed_step='Inferring alarm from expressions',
            basis='Expressions are ambiguous.',evidence_ids=['VF1'])] if objection else []
        challenge=dict(alignment='MATCH',objections=objections,evidence_ids=['VF1'],
                       reason='A material uncertainty remains.' if objection else 'Alarmed expressions oppose the asserted calmness.')
        rt=Runtime([first,*old[:3],challenge if mode=='audit-only-1' else old[3]])
        rt.hardware_profile=SimpleNamespace(tribunal_protocol='evidence-review-5.0')
        rt.tribunal_audit_mode=mode
        ledger.append(dict(id='LC1',source='agent2',type='caption_proposition',text=source,grounded=False))
        language={'claim_contract':dict(contract(),source_caption=source,caption_proposition=source)}
        result=review(rt,image,source,language,ledger)
        bridge=build_semantic_bridge(result,ledger,language['claim_contract'],language)
        final,_,resolution=apply_tribunal_resolution({'label':'ENTAILS','confidence':.3},result,ledger,
            language['claim_contract'],semantic_bridge_mode='corroborated',source_caption=source)
        return rt,bridge,ledger,final,resolution

    def test_accepts_correct_conflict_through_real_gate(self):
        rt,bridge,ledger,final,resolution=self.run_case()
        self.assertTrue(audit(bridge,ledger)['valid'])
        self.assertTrue(resolution['accepted'])
        self.assertEqual(final['label'],'CONTRADICTS')
        self.assertEqual(len(rt.prompts),5)

    def test_upstream_prompts_unchanged(self):
        baseline,*_=self.run_case(mode='baseline')
        candidate,*_=self.run_case()
        self.assertEqual(baseline.prompts[:4],candidate.prompts[:4])

    def test_material_objection_still_blocks_change(self):
        _,bridge,ledger,final,resolution=self.run_case(objection=True)
        self.assertFalse(audit(bridge,ledger)['valid'])
        self.assertFalse(resolution['accepted'])
        self.assertEqual(final['label'],'ENTAILS')

    def test_protocol_cannot_be_stripped(self):
        _,bridge,ledger,_,_=self.run_case()
        for erase_both in (False,True):
            edited=deepcopy(bridge);edited.pop('audit_only_protocol')
            if erase_both:edited['independent_verification'].pop('audit_only_protocol')
            self.assertFalse(audit(edited,ledger)['valid'])

    def test_audit_must_match_delivered_wire_record(self):
        _,bridge,ledger,_,_=self.run_case(objection=True)
        edited=bridge['independent_verification']['obligations']['arguments']
        edited['objections']=[];edited['decision_errors']=[]
        self.assertFalse(audit(bridge,ledger)['valid'])


if __name__=='__main__':unittest.main()
