import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from engine.case_dossier import (context_catalog, render_judge_dossier,
    disclose_review_context, fit_judge_packet)
from engine.claim_graph import source_identity_graph
from models.judge_model import QwenJudgeModel
from agents.multimodal_judge import TribunalMediatorAgent


class CompleteContextTests(unittest.TestCase):
    def fixture(self):
        return {'source_caption':'The line rises.', 'schema_version':'1.0',
            'claim_agent':{'claim_graph':source_identity_graph('The line rises.')},
            'candidate_cases':[{'claim_reading':'First fallible reading'}, {'claim_reading':'Second fallible reading'}],
            'evidence_catalog':[]}

    def test_all_indexed_context_is_disclosed_without_changing_source(self):
        dossier=self.fixture(); original=copy.deepcopy(dossier)
        packet=render_judge_dossier(dossier)
        packet['remaining_context_index']=[{'context_id':x['context_id']} for x in packet['candidate_cases']]
        packet['candidate_cases']=[]
        fitted, ids=disclose_review_context(packet,dossier)
        self.assertEqual(ids,{x['context_id'] for x in context_catalog(dossier)})
        self.assertEqual(fitted['context_records'],context_catalog(dossier))
        self.assertEqual(fitted['remaining_context_index'],[])
        self.assertEqual(dossier,original)
        self.assertEqual(fitted['source_caption'],dossier['source_caption'])
        self.assertTrue(all(x['decision_grade'] is False for x in fitted['context_records']))
        again, _=disclose_review_context(fitted,dossier)
        self.assertEqual(again,fitted)

    def test_protected_context_cannot_be_silently_reindexed(self):
        dossier=self.fixture()
        packet, ids=disclose_review_context(render_judge_dossier(dossier),dossier)
        with self.assertRaises(ValueError):
            fit_judge_packet(packet,json.dumps,token_counter=len,max_tokens=100,protected_ids=ids)

    def test_stale_context_rejected(self):
        dossier=self.fixture(); packet=render_judge_dossier(dossier)
        packet['remaining_context_index']=[{'context_id':'CTX_missing'}]
        with self.assertRaises(ValueError):disclose_review_context(packet,dossier)

    def test_multimodal_budget_uses_total_cap_and_output_reserve(self):
        rt=QwenJudgeModel.__new__(QwenJudgeModel)
        rt.hardware_profile=SimpleNamespace(judge_max_pixels=1048576,judge_total_tokens=6144)
        calls=[]
        class Processor:
            def apply_chat_template(self,messages,**kwargs):
                calls.append((messages,kwargs))
                return {'input_ids':SimpleNamespace(shape=(1,800))}
        rt.processor=Processor()
        self.assertEqual(rt.review_text_budget(None,1024),4256)
        self.assertEqual(calls[0][0],rt._messages(None,''))
        self.assertFalse(calls[0][1]['enable_thinking'])
        with self.assertRaises(ValueError):rt.review_text_budget(None,6144)

    def test_v5_compact_proposer_receives_source_without_peer_readings(self):
        dossier=self.fixture(); graph=dossier['claim_agent']['claim_graph']
        packet=render_judge_dossier(dossier)
        packet['remaining_context_index']=[{'context_id':x['context_id']} for x in packet['candidate_cases']]
        packet['candidate_cases']=[]
        seen=[]
        class Runtime:
            hardware_profile=SimpleNamespace(tribunal_protocol='evidence-review-5.0',judge_text_tokens=100000,
                judge_output_tokens=512,judge_evidence_first=False)
            count_text_tokens=staticmethod(len)
            _last_generation_diagnostics={}
        def generation(runtime,image,prompt,*args,**kwargs):
            seen.append(prompt)
            return {'_format_valid':True,'_execution_status':'SUCCEEDED','status':'ABSTAIN',
                'relation':'UNRESOLVED','provisional_verdict':'ABSTAIN','evidence_ids':[],
                'context_requests':[],'_generation_seconds':.01}
        with patch('agents.multimodal_judge.build_case_dossier',return_value=dossier), \
             patch('agents.multimodal_judge.render_judge_dossier',return_value=packet), \
             patch('agents.multimodal_judge._run_structured_generation',side_effect=generation):
            result=TribunalMediatorAgent(Runtime()).review(None,dossier['source_caption'],{},
                {'claim_graph':graph},{},[],{})
        self.assertEqual(len(seen),1)
        # The integrated compact proposer independently reads the image and
        # full literal catalogue; peer interpretations are no longer its input.
        self.assertNotIn('First fallible reading',seen[0])
        self.assertNotIn('Second fallible reading',seen[0])
        self.assertEqual(result['_judge_packet']['source_caption'], dossier['source_caption'])
        self.assertEqual(result['_proposer_version'], 'compact-judge-1')


if __name__=='__main__':unittest.main()
