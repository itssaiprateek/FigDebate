"""Acceptance tests for the diagnosed September 8 boundary failures."""
import copy
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from engine.claim_graph import compile_draft, truth_conditions, conditions_are_compiled, CONDITION_ORIGIN, fingerprint, repair_graph
from engine.claim_graph import projected_requirements
from engine.claim_semantics import audit_core, CHECKS
from engine.claim_validation import core_errors
from engine.case_dossier import fit_judge_packet, compact_graph
from agents.multimodal_judge import bound_review_contract, _run_structured_generation
from utils.judge_parser import parse_tribunal_review_response
from tests.test_claim_graph import graph_fixture
from tests.critical_fixture import core
from engine.review_outcome import classify_review
from engine.claim_graph import protect_source_obligation


def draft_fixture():
    g = graph_fixture()
    return {"composition":g["composition"], "nodes":[
        {k:v for k,v in n.items() if k not in {"id","depends_on","support_condition","conflict_condition"}}
        | {"depends_on_indices":[]} for n in g["nodes"]]}


class CriticalPipelineTests(unittest.TestCase):
    def test_compiled_guard_handles_malformed_model_fields(self):
        for graph in (None, [], {"condition_origin": CONDITION_ORIGIN, "nodes": ["bad"]},
                      {"condition_origin": CONDITION_ORIGIN, "nodes": [{"interpreted_proposition": None}]}):
            self.assertFalse(conditions_are_compiled(graph))

    def test_source_identical_prose_cannot_certify_wrong_role_metadata(self):
        caption = "The dog chased the man."
        graph = protect_source_obligation(caption, self.compiled())
        graph["nodes"][0].update(subject="man", predicate="chased dog")
        admitted = protect_source_obligation(caption, graph)
        self.assertEqual(admitted["nodes"][0]["subject"], caption)
        self.assertEqual(admitted["nodes"][0]["predicate"], "asserts")

    def test_blocked_context_is_not_recorded_as_session_abstention(self):
        from engine.tribunal import record_tribunal_round, new_tribunal_session
        review = {"_format_valid": True, "_context_valid": False, "_context_status": "BLOCKED_UPSTREAM", "status": "ABSTAIN"}
        self.assertEqual(record_tribunal_round(new_tribunal_session(), review)["state"], "BLOCKED_UPSTREAM")
        self.assertEqual(classify_review(review)["verification_status"], "BLOCKED_CONTEXT")

    def test_executed_verification_is_not_the_same_as_passed_verification(self):
        # A V3 execution report requires every obligation to have completed;
        # negative semantic judgments must still count as completed execution.
        call = {"_format_valid": True, "_execution_status": "SUCCEEDED", "verified": False}
        review = {"_format_valid": True, "_independent_verification": {"schema_version":"3.0",
            "calls": [dict(call), dict(call)],
            "obligations": {name:dict(call) for name in ("caption", "visual", "mapping", "arguments")}}}
        self.assertEqual(classify_review(review)["verification_status"], "EXECUTED")

    def test_unqualified_paraphrase_cannot_replace_full_caption(self):
        for caption, propositions in (
            ("Products one dislikes disappear quickly, while products one loves last a long time.",
             ["Disliked products last less long than loved ones."] * 2),
            ("His heart within him is fully rotten.",
             ["His physical heart is decayed.", "He is morally corrupt."]),
            ("The faculty meeting was calm and relaxing.", ["The meeting was quiet."]),
        ):
            graph = self.compiled()
            for node, proposition in zip(graph["nodes"], propositions):
                node["interpreted_proposition"] = proposition
            admitted = protect_source_obligation(caption, graph)
            self.assertEqual(admitted["composition"], "SINGLE")
            self.assertEqual(admitted["nodes"][0]["interpreted_proposition"], caption)
            self.assertEqual(admitted["representation"], "UNDECOMPOSED_SOURCE")
            self.assertFalse(admitted["semantic_verified"])

    def test_source_identity_cannot_be_forged_by_rehashing(self):
        g = protect_source_obligation("The boat crossed slowly.", self.compiled())
        g["nodes"][0]["interpreted_proposition"] = "The boat crossed."
        n = g["nodes"][0]
        n["support_condition"], n["conflict_condition"] = truth_conditions(n["interpreted_proposition"])
        g["fingerprint"] = fingerprint(g)
        fields = core(g["source_caption"])
        fields["claim_graph"] = g
        fields["expected_visual_state"], fields["opposite_visual_state"] = projected_requirements(g)
        self.assertIn("SOURCE_IDENTITY_MODIFIED", core_errors(fields))

    def test_audit_metric_excludes_unknown_and_not_run(self):
        import pandas as pd
        from evaluation.evaluate_predictions import evaluated_boolean_rate
        self.assertEqual(evaluated_boolean_rate(pd.Series([True, False, None, "UNKNOWN"])), 0.5)
        self.assertIsNone(evaluated_boolean_rate(pd.Series([None, "NOT_RUN"])))

    def compiled(self):
        g = compile_draft(draft_fixture())
        g.update(source_caption=graph_fixture()["source_caption"], errors=[], condition_origin=CONDITION_ORIGIN)
        g["fingerprint"] = fingerprint(g)
        return g

    def test_ids_and_truth_conditions_are_not_model_generated(self):
        g = self.compiled()
        self.assertEqual([n["id"] for n in g["nodes"]], ["C1","C2"])
        self.assertTrue(conditions_are_compiled(g))
        self.assertEqual(g["nodes"][0]["support_condition"], truth_conditions("The red line rises.")[0])

    def test_fractional_and_future_dependencies_rejected(self):
        for dependency in (1.5, 2, 1):
            d = draft_fixture()
            d["nodes"][0]["depends_on_indices"] = [dependency]
            self.assertEqual(compile_draft(d), {})

    def test_modified_compiled_conditions_rejected(self):
        g = self.compiled()
        g["nodes"][0]["support_condition"] = "An invented image fact."
        fields = core(g["source_caption"])
        fields["claim_graph"] = g
        fields["expected_visual_state"], fields["opposite_visual_state"] = projected_requirements(g)
        self.assertIn("COMPILED_CONDITIONS_MODIFIED", core_errors(fields))

    def test_blocked_audit_does_not_claim_failed_semantics(self):
        fields = core("The line rises.")
        fields["expected_visual_state"] = "None"
        agent = SimpleNamespace(_generate_section=lambda *a,**k:self.fail("Should not call model"))
        result = audit_core(agent, "The line rises.", fields)
        self.assertEqual(result["audit_status"], "BLOCKED_UPSTREAM")
        self.assertFalse(result["semantic_checks_executed"])
        self.assertTrue(all(result[k] is None for k in CHECKS))

    def test_compiled_conditions_require_source_audit_not_blind_approval(self):
        for status in ("PASS", "FAIL", "UNKNOWN"):
            g = self.compiled()
            fields = core(g["source_caption"])
            fields["claim_graph"] = g
            fields["expected_visual_state"], fields["opposite_visual_state"] = projected_requirements(g)
            calls=[]
            def generate(prompt, caption, budget, schema=None):
                calls.append(prompt)
                value=({"answer":caption,"source_ids":["C1"],"unknown":False}
                       if "answer" in schema["properties"] else {"status":status,"reason":"Controlled source audit."})
                return value,json.dumps(value),0,{"schema_valid":True}
            result=audit_core(SimpleNamespace(_generate_section=generate),g["source_caption"],fields)
            self.assertEqual(len(calls),3)
            self.assertEqual(result["expected_state_supports_full_claim"], True if status=="PASS" else None)

    def test_budget_never_drops_graph_or_catalog_ids(self):
        graph = compact_graph(self.compiled())
        packet = {"source_caption":"caption", "claim_agent":{"claim_graph":graph, "extra":"x"*1000},
                  "targeted_hearing":{"observation":"important"},
                  "evidence_ledger":[{"id":"E1","text":"a"*600},{"id":"E2","text":"b"*600}],
                  "remaining_evidence_index":[], "evidence_rendering":{"active_count":2}}
        original=copy.deepcopy(packet)
        fitted, audit=fit_judge_packet(packet,json.dumps,token_counter=len,max_tokens=len(json.dumps(packet))-1200)
        self.assertEqual(fitted["claim_agent"]["claim_graph"],graph)
        self.assertEqual(fitted["targeted_hearing"],packet["targeted_hearing"])
        ids={i["id"] for i in fitted["evidence_ledger"]+fitted["remaining_evidence_index"]}
        self.assertEqual(ids,{"E1","E2"})
        self.assertEqual(packet,original)
        self.assertTrue(audit["all_evidence_discoverable"])

    def test_impossible_budget_fails_explicitly_not_by_losing_graph(self):
        with self.assertRaisesRegex(ValueError,"mandatory"):
            fit_judge_packet({"source_caption":"caption","claim_agent":{"claim_graph":self.compiled()},
                              "evidence_ledger":[{"id":"E1","text":"observation"}]},json.dumps,max_tokens=10)

    def review(self):
        return {"relation":"SUPPORT","admissibility":"VERIFIED","node_relations":[
            {"claim_node_id":n,"relation":"SUPPORT","evidence_ids":["E1"]} for n in ("C1","C2")],
            "visual_premise":"The observations match.","caption_premise":"Caption.",
            "semantic_bridge_type":"GENERAL_SEMANTIC_RELATION","semantic_bridge":"The claim matches.",
            "evidence_ids":["E1"],"counter_interpretation":"","counter_interpretation_strength":0.0,
            "confidence":0.9,"requested_follow_up":"NONE","reason":"Test reason."}

    def test_contextual_errors_receive_bounded_retry(self):
        graph=self.compiled()
        parser,schema=bound_review_contract(graph,{"claim_agent":{"claim_graph":compact_graph(graph)}},{"E1"})
        wrong=self.review()
        wrong["node_relations"][0]["claim_node_id"]="E1"
        outputs=[json.dumps(wrong),json.dumps(self.review())]
        calls=[]
        class Runtime:
            _last_generation_diagnostics={}
            def generate(self,image,prompt,max_new_tokens=None,json_schema=None):
                calls.append(prompt)
                return outputs.pop(0),0
        result=_run_structured_generation(Runtime(),None,"Case",parser,max_new_tokens=100,
                                          contract_name="tribunal_review",output_schema=schema)
        self.assertTrue(result["_format_valid"])
        self.assertTrue(result["_format_retry_success"])
        self.assertIn("UNKNOWN_OR_DUPLICATE_CLAIM_NODE",calls[1])
        self.assertEqual(schema["properties"]["node_relations"]["items"]["properties"]["claim_node_id"]["enum"],["C1","C2"])
        self.assertNotIn("enum",schema["properties"]["caption_premise"])

    def test_upstream_failure_is_not_syntax_failure_or_abstention(self):
        g=self.compiled();g["errors"]=["C1:MISSING_BINDING"]
        parser,_=bound_review_contract(g,{"claim_agent":{"claim_graph":compact_graph(g)}},{"E1"})
        result=parser(json.dumps(self.review()))
        self.assertTrue(result["_format_valid"])
        self.assertEqual(result["_context_status"],"BLOCKED_UPSTREAM")
        self.assertFalse(classify_review(result)["semantic_abstained"])
        self.assertFalse(classify_review(result)["semantic_eligible"])

    def test_missing_graph_cannot_be_accepted(self):
        parser,_=bound_review_contract(self.compiled(),{"claim_agent":{}},{"E1"})
        self.assertEqual(parser(json.dumps(self.review()))["_format_error"],"mandatory_graph_not_visible_or_stale")

    def test_partial_support_does_not_establish_conjunction(self):
        graph=self.compiled()
        parser,_=bound_review_contract(graph,{"claim_agent":{"claim_graph":compact_graph(graph)}},{"E1"})
        proposed=self.review();proposed["node_relations"][1]["relation"]="UNRESOLVED"
        result=parser(json.dumps(proposed))
        self.assertEqual(result["relation"],"UNRESOLVED")
        self.assertEqual(result["provisional_verdict"],"ABSTAIN")
