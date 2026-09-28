"""Mechanism tests; scripted answers do not establish model accuracy."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock
from arbiter.arbiter import Arbiter
from engine.reasoning_contract import completion_status, VERSION
from engine.evidence_review_v5 import verify, audit
from tests.test_tribunal_v5 import v5_fixture
from tests.test_evidence_review_v4 import Runtime

COMPLETE = """Visual Evidence: People look alarmed.
Evidence IDs: VF1
Expression: a calm meeting
Referent: meeting participants
Property: calmness
Caption Meaning: The meeting is calm.
Relation Status: CONFLICT
Relation Analysis: The participants' alarmed expressions conflict with the claimed calmness."""


class GroundedCompletionTests(unittest.TestCase):
    def arbiter(self, responses):
        a = Arbiter.__new__(Arbiter)
        a.completion_checks = True
        a.grounded_interpretation = True
        a._nli_verifier = None
        def generate(prompt, max_new_tokens, **kwargs):
            text, hit = responses.pop(0)
            a._last_generation_diagnostics = {"hit_token_limit": hit, "generated_tokens": max_new_tokens if hit else 90}
            if "output_schema" in kwargs and text == COMPLETE:
                text = json.dumps(dict(evidence_ids=["VF1"],referent="meeting participants",property="calmness",
                    meaning="The meeting is calm.",relation="CONFLICT",
                    reason="The participants' alarmed expressions conflict with the claimed calmness."))
            return text, .1
        a._generate_response = Mock(side_effect=generate)
        a._score_semantic_relations = Mock(return_value=("CONTRADICTS", .8, {"ENTAILS":.2,"CONTRADICTS":.8}))
        return a

    def run_initial(self, a):
        return a.analyze("a calm meeting", {}, {}, {"grounded_evidence_catalog": [
            {"id":"VF1","text":"People look alarmed.","source":"agent1","grounded":True,"relation":"NEUTRAL"}]})

    def test_truncated_argument_is_replaced_before_scoring(self):
        partial = COMPLETE[:-20]
        a = self.arbiter([(partial,True),(COMPLETE,False)])
        result = self.run_initial(a)
        self.assertEqual(a._generate_response.call_count, 2)
        self.assertIn("conflict with the claimed calmness",a._score_semantic_relations.call_args.args[-1])
        self.assertTrue(result['_assessment_completion']['complete'])
        self.assertEqual(result['_assessment_attempts'][0]['text'],partial)

    def test_two_failures_leave_honest_unresolved_status_and_no_partial_scoring(self):
        a = self.arbiter([(COMPLETE[:-20],True),(COMPLETE[:-10],True)])
        result = self.run_initial(a)
        self.assertEqual(a._generate_response.call_count,2)
        self.assertFalse(result['_assessment_completion']['complete'])
        self.assertEqual(result['_relation_status'],'INSUFFICIENT')
        self.assertTrue(result['_binary_label_is_unverified'])
        self.assertIn('could not be completed',a._score_semantic_relations.call_args.args[-1])
        self.assertLessEqual(result['confidence'],.35)

    def test_complete_first_pass_has_no_retry(self):
        a = self.arbiter([(COMPLETE,False)])
        result = self.run_initial(a)
        self.assertEqual(a._generate_response.call_count,1)
        self.assertEqual(result['_semantic_relation_status'],'CONFLICT')

    def test_oversize_input_fails_before_generation_instead_of_silent_truncation(self):
        from types import SimpleNamespace
        class Inputs(dict):
            def to(self, device):
                return self
        a = Arbiter.__new__(Arbiter)
        a.completion_checks = True
        a.model = Mock(device="cpu")
        a.tokenizer = Mock(return_value=Inputs(input_ids=SimpleNamespace(shape=(1, 2049))))
        with self.assertRaisesRegex(RuntimeError, "context budget"):
            a._generate_response("Long source and evidence", max_new_tokens=256)
        self.assertFalse(a.tokenizer.call_args.kwargs["truncation"])
        a.model.generate.assert_not_called()
        self.assertEqual(a._last_generation_diagnostics["finish_reason"], "input_budget_exceeded")

    def test_failure_marker_reaches_final_artifact(self):
        from engine.final_artifact import final_artifact
        a = self.arbiter([(COMPLETE[:-20], True), (COMPLETE[:-10], True)])
        result = self.run_initial(a)
        artifact = final_artifact("a calm meeting", {"decision": result})
        self.assertFalse(artifact["explanation_completeness"]["complete"])
        self.assertIn("bounded_generation_failed", artifact["explanation_completeness"]["issues"])
        self.assertTrue(result["_binary_label_is_unverified"])

    def test_missing_property_is_not_a_complete_assessment(self):
        self.assertFalse(completion_status(COMPLETE.replace('Property: calmness\n',''), grounded=True)['complete'])

    def test_dictionary_gloss_is_not_a_source_expression(self):
        gloss = COMPLETE.replace("Expression: a calm meeting", "Expression: the absence of agitation")
        self.assertFalse(completion_status(gloss, grounded=True, source="a calm meeting")["complete"])

    def proof(self):
        image, proposal, ledger, answers = v5_fixture()
        proposal['grounded_reading_protocol'] = VERSION
        reading = dict(reading='LITERAL', expression='calm', referent='meeting participants', property='calmness',
                       evidence_ids=['VF1'],reason='The caption attributes calmness to the depicted meeting participants.')
        answers.insert(1,reading)
        runtime = Runtime(answers)
        proposal['independent_verification'] = verify(runtime,image,proposal,ledger)
        return runtime,proposal,ledger

    def test_grounded_reading_is_blind_and_required_by_real_audit(self):
        rt,p,ledger = self.proof()
        self.assertTrue(audit(p,ledger)['valid'])
        payload = json.loads(rt.prompts[1].rsplit('\n',1)[-1])
        self.assertEqual(set(payload),{'source_caption','observations'})
        self.assertIn('reading_hypothesis',json.loads(rt.prompts[3].rsplit('\n',1)[-1]))
        p['independent_verification']['obligations'].pop('grounded_reading')
        self.assertFalse(audit(p,ledger)['valid'])

    def test_reading_tampering_cannot_pass_audit(self):
        _,p,ledger=self.proof()
        p['independent_verification']['obligations']['grounded_reading']['property']='wealth'
        self.assertFalse(audit(p,ledger)['valid'])

    def test_removing_requirement_cannot_bypass_case_binding(self):
        _,p,ledger=self.proof()
        p.pop('grounded_reading_protocol')
        p['independent_verification'].pop('grounded_reading_protocol')
        self.assertFalse(audit(p,ledger)['valid'])

    def test_unresolved_mapping_does_not_reach_relation_or_acceptance(self):
        image,p,ledger,answers=v5_fixture()
        p['grounded_reading_protocol']=VERSION
        answers.insert(1,dict(reading='UNRESOLVED',expression='calm',referent='UNRESOLVED',property='calmness',
                             evidence_ids=['VF1'],reason='The relevant participant cannot be identified.'))
        rt=Runtime(answers)
        p['independent_verification']=verify(rt,image,p,ledger)
        self.assertEqual(p['independent_verification']['stopped_after'],'grounded_reading')
        self.assertEqual(len(rt.prompts),2)
        self.assertFalse(audit(p,ledger)['valid'])


if __name__=='__main__':
    unittest.main()
