"""First-pass output contracts; simulated responses do not prove semantic truth."""
from copy import deepcopy
import json
import unittest
from engine.evidence_review_v5 import verify,audit,selection_valid,catalog,LOCATION_PROTOCOL
from engine.evidence_verification import capped_generated_clause,unfinished_generated_field
from tests.test_tribunal_v5 import v5_fixture
from tests.test_evidence_review_v4 import Runtime

class PrimaryMechanismTests(unittest.TestCase):
    def make_proof(self):
        image,p,ledger,answers=v5_fixture()
        rt=Runtime(answers);p['independent_verification']=verify(rt,image,p,ledger)
        return rt,p,ledger

    def test_location_field_is_a_lossless_translation_not_extra_reasoning(self):
        rt,p,ledger=self.make_proof()
        self.assertTrue(audit(p,ledger)['valid'])
        self.assertEqual(len(rt.prompts),4)
        self.assertIn('image_location names ONLY WHERE',rt.prompts[0])
        v=p['independent_verification']['obligations']['visual']
        raw=json.loads(v['_raw_output'])
        self.assertEqual(v['_location_wire_protocol'],LOCATION_PROTOCOL)
        self.assertEqual(v['observations'][0]['attachment'],raw['observations'][0]['image_location'])
        self.assertEqual(v['observations'][0]['observed'],ledger[0]['text'])
        self.assertFalse(v['_format_retry_used'])

    def test_long_source_record_does_not_have_to_fit_location_field(self):
        image,p,ledger,answers=v5_fixture()
        ledger[0]['text']='An observed event with its source and panel preserved. '*30
        rt=Runtime(answers);proof=verify(rt,image,p,ledger)
        v=proof['obligations']['visual']
        self.assertTrue(v['verified'])
        self.assertEqual(v['observations'][0]['observed'],ledger[0]['text'])
        self.assertLess(len(v['observations'][0]['attachment']),240)

    def test_tampered_location_and_copied_observation_fail(self):
        _,p,ledger=self.make_proof()
        v=p['independent_verification']['obligations']['visual']
        for field in ['attachment','observed']:
            changed=deepcopy(v);changed['observations'][0][field]='A different region or observation.'
            self.assertFalse(selection_valid(changed,catalog(ledger)))
        changed=deepcopy(v);changed['_location_wire_protocol']='unknown'
        self.assertFalse(selection_valid(changed,catalog(ledger)))

    def test_existing_record_schema_remains_auditable(self):
        _,p,ledger=self.make_proof()
        v=deepcopy(p['independent_verification']['obligations']['visual'])
        raw=json.loads(v['_raw_output'])
        for x in raw['observations']:x['attachment']=x.pop('image_location')
        v['_raw_output']=json.dumps(raw);v.pop('_location_wire_protocol')
        self.assertTrue(selection_valid(v,catalog(ledger)))

    def test_location_completeness_is_checked_before_recovery(self):
        from engine.evidence_review_v5 import SELECTION_WIRE
        value={'observations':[{'evidence_id':'VF1','supported':True,'image_location':'At the,'}],
               'coverage_complete':True,'missing_observation':''}
        self.assertTrue(unfinished_generated_field(value))
        value['observations'][0]['image_location']='x'*240
        self.assertTrue(capped_generated_clause(value,SELECTION_WIRE))

    def test_missing_evidence_or_unsupported_record_still_blocks(self):
        for change in ['coverage','unsupported']:
            image,p,ledger,answers=v5_fixture()
            if change=='coverage':answers[0].update(coverage_complete=False,missing_observation='Read the other panel.')
            else:answers[0]['observations'][0]['supported']=False
            rt=Runtime(answers);p['independent_verification']=verify(rt,image,p,ledger)
            self.assertFalse(audit(p,ledger)['valid'])
            self.assertEqual(len(rt.prompts),1)

if __name__=='__main__':unittest.main()
