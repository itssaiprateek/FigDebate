"""Transport/provenance tests. Model semantics are qualified separately."""
import copy
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from engine.caption_answers import answer_caption_question, question_id
from engine.claim_graph import attach_graph, bind_readings, source_identity_graph, projected_requirements
from engine.claim_witness import audit_claim_witness
from engine.case_dossier import (build_case_dossier, render_judge_dossier, fit_judge_packet,
    retrieve_dossier_context, context_catalog)
from engine.deliberation import deliberation_signature
from engine.batch_runner import StagewiseRunner
from engine.debate import DebateEngine
from engine.final_artifact import final_artifact
from engine.tribunal import followup_plan, repair_followup_plan
from engine.candidate_cases import candidate_hearing_plan
from agents.multimodal_judge import TribunalMediatorAgent, bound_review_contract, build_tribunal_review_prompt
from agents.visual_grounding import VisualGroundingAgent
from agents.visual_adapter import AtomicVisualQuestionController
from tests.critical_fixture import core


class CaptionAgent:
    def __init__(self, unknown=False):self.calls=[];self.unknown=unknown
    def _generate_section(self, instruction, caption, budget, schema=None):
        self.calls.append((instruction,caption,budget,schema))
        v={"answer":"An explicitly fallible answer.","source_ids":["C1"],"unknown":self.unknown}
        return v,json.dumps(v),0,{"schema_valid":True}


def fields(caption):
    value=core(caption);attach_graph(None,caption,value)
    return value


def proposal(context_requests=()):
    return {"relation":"UNRESOLVED","admissibility":"INSUFFICIENT",
        "node_relations":[{"claim_node_id":"C1","relation":"UNRESOLVED","evidence_ids":[]}],
        "visual_premise":"", "caption_premise":"The line rises.",
        "semantic_bridge_type":"GENERAL_SEMANTIC_RELATION", "semantic_bridge":"",
        "counter_interpretation":"", "evidence_ids":[], "context_requests":list(context_requests),
        "confidence":0.0,"counter_interpretation_strength":0.0,"requested_follow_up":"NONE",
        "reason":"Need the actual source and arguments; fixture makes no semantic claim."}


class InformationExchangeTests(unittest.TestCase):
    def test_software_citations_preserve_exact_source_bytes(self):
        caption="i can t  believe.\nUnicode: café — ?"
        answer,attempt=answer_caption_question(CaptionAgent(),caption,"What is expressed?")
        self.assertEqual(answer["source_quotes"],[caption])
        self.assertEqual(answer["source_spans"][0],{"start":0,"end":len(caption),"text":caption})
        self.assertFalse(answer["semantic_qualified"])
        self.assertNotIn("source_quotes",attempt["schema"]["properties"])

    def test_wrong_source_id_is_not_repaired_into_a_valid_citation(self):
        agent=CaptionAgent()
        agent._generate_section=lambda *a,**k: ({"answer":"guess","source_ids":["C99"],"unknown":False},"",0,{"schema_valid":True})
        answer,_=answer_caption_question(agent,"caption","question")
        self.assertFalse(answer["source_anchored"])
        self.assertEqual(answer["answer_status"],"INVALID_RESPONSE")

    def test_all_requested_questions_reach_the_model_exactly(self):
        agent=CaptionAgent();caption="The line rises."
        questions=["What is asserted?","What is unknown?","Which roles are expressed?"]
        with patch("engine.claim_witness.audit_core",side_effect=AssertionError("Do not rerun generic audit")):
            result=audit_claim_witness(agent,caption,{"claim_fields":fields(caption),"advisory_questions":questions})
        self.assertEqual(len(agent.calls),3)
        for call,answer,question in zip(agent.calls,result["question_answers"],questions):
            self.assertIn(question,call[0]);self.assertEqual(answer["question"],question)
            self.assertEqual(answer["question_id"],question_id(caption,question))
        self.assertEqual(result["communication"]["delivered"],3)
        self.assertTrue(result["requirements_valid"])
        self.assertEqual(result["communication"]["semantic_qualified"],0)

    def test_no_requested_question_is_not_a_missing_response(self):
        agent=CaptionAgent();caption="The line rises."
        result=audit_claim_witness(agent,caption,{"claim_fields":fields(caption),"advisory_questions":[]})
        self.assertEqual(agent.calls,[])
        self.assertEqual(result["communication"]["requested"],0)

    def test_generation_failure_retains_record_and_continues_next_question(self):
        agent=CaptionAgent();generate=agent._generate_section
        def flaky(instruction,*args,**kwargs):
            if "Requested question: first" in instruction:raise RuntimeError("controlled failure")
            return generate(instruction,*args,**kwargs)
        agent._generate_section=flaky
        result=audit_claim_witness(agent,"caption",{"claim_fields":fields("caption"),"advisory_questions":["first","second"]})
        self.assertEqual([r["answer_status"] for r in result["question_answers"]],["FAILED","ANSWERED"])
        self.assertEqual(result["communication"]["failed"],1)

    def test_unknown_answer_is_visible_as_partial_testimony_not_verified_fact(self):
        caption="The line rises.";agent=CaptionAgent(unknown=True)
        answer,_=answer_caption_question(agent,caption,"What cannot be known?")
        dossier={"source_caption":caption,"targeted_hearing":{"agent2_critique":{
            "_format_valid":True,"requirements_valid":True,"question_answers":[answer],"reading_clarification":answer}}}
        packet=render_judge_dossier(dossier)
        shown=packet["targeted_hearing"]["agent2_critique"]["question_answers"][0]
        self.assertEqual(shown["answer"],answer["answer"])
        self.assertTrue(shown["unknown"])
        self.assertIn("fallible",shown["authority"])

    def test_graph_readings_cannot_change_source_truth_conditions(self):
        graph=source_identity_graph("heart sank")
        answer,_=answer_caption_question(CaptionAgent(),"heart sank","Explain the idiom.")
        updated=bind_readings(graph,[answer])
        self.assertEqual(updated["fingerprint"],graph["fingerprint"])
        self.assertEqual(updated["nodes"],graph["nodes"])
        self.assertFalse(updated["semantic_verified"])
        changed=dict(answer,answer="A different fallible interpretation.")
        self.assertNotEqual(updated["reading_fingerprint"],bind_readings(graph,[changed])["reading_fingerprint"])

    def test_reading_for_different_source_is_rejected(self):
        answer,_=answer_caption_question(CaptionAgent(),"source A","question")
        with self.assertRaisesRegex(ValueError,"not bound"):
            bind_readings(source_identity_graph("source B"),[answer])

    def test_graph_default_does_not_generate_discarded_paraphrases(self):
        agent=CaptionAgent();value=fields("heart sank")
        attach_graph(agent,"heart sank",value)
        self.assertFalse(agent.calls)
        self.assertEqual(value["_graph_generation"]["status"],"NOT_RUN_UNQUALIFIED_DECOMPOSER")
        self.assertEqual(value["claim_graph"]["nodes"][0]["interpreted_proposition"],"heart sank")

    def test_changed_answer_triggers_deliberation_even_if_graph_source_unchanged(self):
        answer,_=answer_caption_question(CaptionAgent(),"source","question")
        a={"language_output":{"claim_graph":bind_readings(source_identity_graph("source"),[answer])},"evidence_ledger":[]}
        b=copy.deepcopy(a);b["language_output"]["claim_graph"]=bind_readings(b["language_output"]["claim_graph"],[dict(answer,answer="changed")])
        self.assertNotEqual(deliberation_signature(a),deliberation_signature(b))

    def test_visual_questions_do_not_collapse_into_first_request(self):
        seen=[]
        class Agent:
            def critique(self,image,prompt):
                question=VisualGroundingAgent._extract_atomic_question(prompt);seen.append(question)
                return {"_format_valid":True,"question":question,"response_status":"VALID_OBSERVATION","observed_state":"fixture"}
        debate=DebateEngine.__new__(DebateEngine)
        case={"key":0,"image":object(),"caption":"source","visual_output":{},"decision":{},"comparison":{},
              "tribunal_hearing":True,"mediation_plan":{"_usable":True,"agent1_questions":["Where is the line?","Read the text\nand its location."]}}
        result=debate._run_visual_questions(Agent(),case)
        self.assertEqual(seen,["Where is the line?","Read the text and its location."])
        self.assertEqual(len(result["question_answers"]),2)
        self.assertEqual(len({r["question_id"] for r in result["question_answers"]}),2)

    def test_followup_plan_keeps_all_questions(self):
        result=followup_plan({"status":"FOLLOW_UP","_format_valid":True,
            "agent1_questions":["a","b"],"agent2_questions":["c","d"],"verification_requests":["e","f"]})
        self.assertEqual(result["agent1_questions"],["a","b"])
        self.assertEqual(result["agent2_questions"],["c","d"])
        self.assertEqual(result["verification_requests"],["e","f"])

    def test_dossier_keeps_original_caption_and_hides_labels(self):
        caption="one  two\nthree"
        dossier=build_case_dossier(caption,{}, {}, {},{"label":"SECRET_PRIOR_LABEL"},[])
        packet=render_judge_dossier(dossier)
        self.assertEqual(packet["source_caption"],caption)
        self.assertNotIn("SECRET_PRIOR_LABEL",json.dumps(packet))

    def test_current_questions_are_mandatory_when_budgeting(self):
        answer,_=answer_caption_question(CaptionAgent(),"source","question")
        dossier={"source_caption":"source","targeted_hearing":{"agent2_critique":{
            "_format_valid":True,"requirements_valid":True,"question_answers":[answer]}}}
        packet=render_judge_dossier(dossier)
        packet["visual_agent"]={"optional":"x"*1000}
        fitted,_=fit_judge_packet(packet,json.dumps,token_counter=len,max_tokens=len(json.dumps(packet))-900)
        self.assertEqual(fitted["targeted_hearing"],packet["targeted_hearing"])

    def test_candidate_removed_from_prompt_remains_retrievable(self):
        candidate={"model_family":"fixture","relation":"UNRESOLVED","claim_reading":"argument "*90}
        dossier={"source_caption":"source","candidate_cases":[candidate]}
        packet=render_judge_dossier(dossier);identifier=packet["candidate_cases"][0]["context_id"]
        fitted,budget=fit_judge_packet(packet,json.dumps,token_counter=len,max_tokens=650)
        self.assertEqual(fitted["candidate_cases"],[])
        self.assertIn(identifier,{i["context_id"] for i in fitted["remaining_context_index"]})
        self.assertEqual(retrieve_dossier_context(dossier,[identifier])[0]["content"],candidate)
        self.assertTrue(budget["all_context_discoverable"])

    def test_unknown_context_cannot_be_requested(self):
        graph=source_identity_graph("The line rises.")
        parser,_=bound_review_contract(graph,{"claim_agent":{"claim_graph":graph}},set())
        self.assertFalse(parser(json.dumps(proposal(["CTX_FAKE"])))["_format_valid"])

    def test_context_id_cannot_be_cited_as_visual_evidence(self):
        graph=source_identity_graph("The line rises.")
        parser,_=bound_review_contract(graph,{"claim_agent":{"claim_graph":graph},
            "remaining_context_index":[{"context_id":"CTX_A"}]},{"VF001"})
        value=proposal();value.update(relation="SUPPORT",admissibility="CORROBORATED",evidence_ids=["CTX_A"])
        value["node_relations"]=[{"claim_node_id":"C1","relation":"SUPPORT","evidence_ids":["CTX_A"]}]
        self.assertFalse(parser(json.dumps(value))["_format_valid"])

    def test_new_hearing_retains_previous_answers_in_final_artifact(self):
        old={"agent2_critique":{"question_answers":[{"question_id":"old","answer":"old answer"}]}}
        result={0:{"decision":{},"evidence_ledger":[],"debate_details":old,"timing":{}}}
        decision={"label":"ENTAILS","_debate":{"rounds":1,"agent2_critique":{"question_answers":[{"question_id":"new","answer":"new answer"}]}}}
        StagewiseRunner._merge_debate_results([{"index":0}],result,{0:decision})
        artifact=final_artifact("source",result[0])
        self.assertEqual(artifact["information_exchange"]["hearing_history"][0],old)
        self.assertIn("new answer",json.dumps(artifact["information_exchange"]))

    def test_context_retrieval_is_reinspected_before_final_review(self):
        # Force a real argument into the index, then exercise the mediator's
        # retrieval/rebudget/regeneration path. Injected outputs are not accuracy evidence.
        graph=source_identity_graph("The line rises.")
        candidate={"model_family":"fixture","relation":"UNRESOLVED","claim_reading":"ARGUMENT"}
        dossier={"source_caption":"The line rises.","claim_agent":{"claim_graph":graph},
                 "candidate_cases":[candidate],"evidence_catalog":[],"schema_version":"1.0"}
        packet=render_judge_dossier(dossier);item=packet["candidate_cases"].pop()
        identifier=item["context_id"]
        packet["remaining_context_index"]=[{"context_id":identifier,"kind":"candidate_argument","requires_retrieval":True}]
        seen=[]
        class Runtime:
            hardware_profile=SimpleNamespace(judge_text_tokens=100000,judge_output_tokens=512)
            count_text_tokens=staticmethod(len)
            _last_generation_diagnostics={}
            def generate(self,image,prompt,**kwargs):
                seen.append(prompt)
                return json.dumps(proposal([identifier] if len(seen)==1 else [])),0.01
        with patch("agents.multimodal_judge.build_case_dossier",return_value=dossier),\
             patch("agents.multimodal_judge.render_judge_dossier",return_value=packet):
            result=TribunalMediatorAgent(Runtime()).review(None,"The line rises.",{},
                {"claim_graph":graph},{},[],{})
        self.assertEqual(len(seen),2)
        packets=[json.loads(prompt.split("CASE PACKET:\n",1)[1]) for prompt in seen]
        self.assertNotIn("ARGUMENT",json.dumps(packets[0]))
        self.assertEqual(packets[1]["context_records"][0]["content"]["claim_reading"],"ARGUMENT")
        self.assertEqual(result["_retrieval_audit"]["requested_context_ids"],[identifier])
        self.assertTrue(result["_communication_audit"]["source_caption_unchanged"])

    def test_current_answer_is_not_duplicated_as_alias_and_graph_context(self):
        answer,_=answer_caption_question(CaptionAgent(),"source","question")
        graph=bind_readings(source_identity_graph("source"),[answer])
        dossier={"source_caption":"source","claim_agent":{"claim_graph":graph},
            "targeted_hearing":{"agent2_critique":{"_format_valid":True,"requirements_valid":True,
                "question_answers":[answer],"reading_clarification":answer,
                "support_requirement":"duplicate source condition","conflict_requirement":"duplicate negation"}}}
        packet=render_judge_dossier(dossier)
        witness=packet["targeted_hearing"]["agent2_critique"]
        self.assertNotIn("reading_clarification",witness)
        self.assertNotIn("support_requirement",witness)
        self.assertEqual(witness["question_answers"][0]["answer"],answer["answer"])
        self.assertEqual(packet["context_records"],[])
        self.assertEqual(dossier["targeted_hearing"]["agent2_critique"]["reading_clarification"],answer)

    def test_candidate_positions_cannot_send_image_questions_to_caption_witness(self):
        plan={"agent1_questions":["Report visible states."],
              "agent2_questions":["Explain the expressed caption."],"verification_requests":["Verify mapping."]}
        questions=["What happened before this photograph?","Which image depicts the same object?"]
        routed=candidate_hearing_plan(plan,{"unresolved":True,"decisive_questions":questions})
        self.assertEqual(routed["agent1_questions"],plan["agent1_questions"])
        self.assertEqual(routed["agent2_questions"],plan["agent2_questions"])
        self.assertEqual([r["question"] for r in routed["candidate_question_routing"]],questions)
        self.assertTrue(all(r["destination"]=="judge_candidate_context" for r in routed["candidate_question_routing"]))
        self.assertNotIn("candidate_question_routing",plan)

    def test_successive_context_requests_progress_and_visible_request_stops(self):
        graph=source_identity_graph("The line rises.")
        dossier={"source_caption":"The line rises.","claim_agent":{"claim_graph":graph},
            "candidate_cases":[{"claim_reading":"first argument"},{"claim_reading":"second argument"}],
            "evidence_catalog":[],"schema_version":"1.0"}
        packet=render_judge_dossier(dossier);items=packet["candidate_cases"];packet["candidate_cases"]=[]
        ids=[i["context_id"] for i in items]
        packet["remaining_context_index"]=[{"context_id":key,"kind":"candidate_argument"} for key in ids]
        seen=[]
        class Runtime:
            hardware_profile=SimpleNamespace(judge_text_tokens=100000,judge_output_tokens=512)
            count_text_tokens=staticmethod(len)
            _last_generation_diagnostics={}
            def generate(self,image,prompt,**kwargs):
                seen.append(prompt)
                return json.dumps(proposal([ids[0] if len(seen)==1 else ids[1]])),0.01
        with patch("agents.multimodal_judge.build_case_dossier",return_value=dossier),\
             patch("agents.multimodal_judge.render_judge_dossier",return_value=packet):
            result=TribunalMediatorAgent(Runtime()).review(None,"The line rises.",{},
                {"claim_graph":graph},{},[],{})
        self.assertEqual(len(seen),3)
        self.assertEqual(result["_retrieval_audit"]["cycles"],2)
        self.assertEqual(result["_retrieval_audit"]["requested_context_ids"],ids)
        self.assertEqual({r["context_id"] for r in result["_judge_packet"]["context_records"]},set(ids))
        self.assertIsNot(result.get("_context_valid"),False)

    def test_failed_retrieval_does_not_claim_unsent_packet_was_disclosed(self):
        graph=source_identity_graph("The line rises.")
        dossier={"source_caption":"The line rises.","claim_agent":{"claim_graph":graph},
            "candidate_cases":[{"claim_reading":"UNSENT_MARKER"*4000}],"evidence_catalog":[],"schema_version":"1.0"}
        packet=render_judge_dossier(dossier);item=packet["candidate_cases"].pop();identifier=item["context_id"]
        packet["remaining_context_index"]=[{"context_id":identifier,"kind":"candidate_argument"}]
        class Runtime:
            hardware_profile=SimpleNamespace(judge_text_tokens=10000,judge_output_tokens=512)
            count_text_tokens=staticmethod(len)
            _last_generation_diagnostics={}
            def generate(self,*args,**kwargs): return json.dumps(proposal([identifier])),0.01
        with patch("agents.multimodal_judge.build_case_dossier",return_value=dossier),\
             patch("agents.multimodal_judge.render_judge_dossier",return_value=packet):
            result=TribunalMediatorAgent(Runtime()).review(None,"The line rises.",{},
                {"claim_graph":graph},{},[],{})
        self.assertEqual(result["_execution_status"],"FAILED")
        self.assertNotIn("UNSENT_MARKER",json.dumps(result["_judge_packet"]))
        self.assertEqual(result["_retrieval_audit"]["requested_context_ids"],[identifier])

    def test_long_repair_subject_does_not_overflow_visual_question_contract(self):
        premise="A fallible proposed observation. "*100
        review={"relation":"SUPPORT","visual_premise":premise,"_independent_verification":{
            "schema_version":"3.0","obligations":{"caption":{"verified":True},"visual":{"verified":False},
                "mapping":{"verified":True},"arguments":{"bridge_grounded":True,"bridge_relation":"SUPPORT","counter_resolved":True}},
            "calls":[{"relation":"SUPPORT"},{"relation":"SUPPORT"}]}}
        plan=repair_followup_plan(review)
        question=plan["agent1_questions"][0]
        valid,error=AtomicVisualQuestionController.validate_question(question,
            AtomicVisualQuestionController.infer_question_type(question))
        self.assertTrue(valid,error)
        self.assertNotIn(premise,question)
        self.assertEqual(plan["repair_subject"]["visual_premise"],premise)

    def test_visual_witness_does_not_copy_caption_subject_into_observed_entity(self):
        agent=VisualGroundingAgent.__new__(VisualGroundingAgent)
        agent.question_controller=AtomicVisualQuestionController()
        agent.answer_visual_question=lambda *a,**k: {"status":"OBSERVED","valid":True,
            "answer":"Two sandwiches have different contents.","elapsed_seconds":0,"diagnostics":{}}
        result=agent._atomic_critique(None,"Question ID: example\nReview question: Describe visible contents.\nClaim subject: SECRET_CAPTION_SUBJECT\nTRIBUNAL_VISUAL_WITNESS_ONLY")
        self.assertNotIn("SECRET_CAPTION_SUBJECT",result["observed_entity"])
        self.assertEqual(result["observed_state"],"Two sandwiches have different contents.")


if __name__=="__main__":unittest.main()
