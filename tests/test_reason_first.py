"""Field-order changes must preserve all gate and schema constraints."""
from copy import deepcopy
import unittest
from engine import semantic_protocol as sem
from engine.simple_judge import schema, prompt
from engine.evidence_review_v5 import audit
from tests import test_semantic_protocol as helpers

class ReasonFirstTests(unittest.TestCase):
    def test_only_order_changes_in_proposal_schema(self):
        original=schema({'E1'},True); changed=schema({'E1'},True,True)
        self.assertEqual(original['properties'],changed['properties'])
        self.assertEqual(set(original['required']),set(changed['required']))
        self.assertEqual(next(iter(changed['properties'])),'decisive_reason')
        self.assertEqual(next(iter(original['properties'])),'reading')
        self.assertEqual(original['additionalProperties'],changed['additionalProperties'])

    def test_only_first_instruction_changes(self):
        context={'semantic_alignment_protocol':sem.ALIGNMENT_VERSION,'source_caption':'Original assertion.'}
        self.assertEqual(prompt(context).replace('Generate reading first:','Describe reading:'),prompt(context,True))

    def test_existing_full_gate_still_accepts_valid_both_directions(self):
        for relation in ['SUPPORT','CONFLICT']:
            helper=helpers.SemanticProtocolTests();helper.setUp()
            rt,review,bridge,ledger,*_=helper.run_case(relation,experiment='reason-first-audit-1')
            self.assertEqual(len(rt.prompts),5)
            self.assertTrue(audit(bridge,ledger)['valid'])
            bad=deepcopy(bridge);bad['reading']['kind']='UNCERTAIN'
            self.assertFalse(audit(bad,ledger)['valid'])
            bad=deepcopy(bridge);bad['independent_verification']['calls'][0]['unestablished_conditions']=['calm']
            if relation=='SUPPORT':self.assertFalse(audit(bad,ledger)['valid'])

    def test_real_audit_objection_still_blocks(self):
        helper=helpers.SemanticProtocolTests();helper.setUp()
        objection=dict(type='UNSUPPORTED_INFERENCE',disputed_step='Claimed calmness.',basis='People look alarmed.',evidence_ids=['VF1'])
        _,_,bridge,ledger,*_=helper.run_case(objections=[objection],experiment='reason-first-audit-1')
        self.assertFalse(audit(bridge,ledger)['valid'])

if __name__=='__main__':unittest.main()
