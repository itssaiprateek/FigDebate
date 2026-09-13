"""Semantic boundary and operational regression tests; no claims of model accuracy."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from engine.claim_graph import source_identity_graph
from engine.evidence_verification import verify, audit, repair_plan, repair_argument
from engine.independent_review import image_subject_hash
from engine.tribunal_protocol import proposal_schema, expand_proposal, unfinished_generated_field
from engine.case_budget import case_budget, require_time, CaseBudgetExceeded
from evaluation.tribunal_quality import summarize_tribunal


def fixture():
    image = Image.new("RGB", (3, 3))
    proposal = {"source_caption": "The meeting is calm.", "caption_premise": "The meeting is calm.",
        "visual_premise": "People look alarmed.", "visual_evidence_ids": ["VF1"],
        "proposed_relation": "CONFLICT", "bridge_statement": "The alarmed expressions conflict with calmness.",
        "image_sha256": image_subject_hash(image)}
    ledger = [{"id": "VF1", "text": "People look alarmed.", "source": "agent1", "type": "visual_fact", "grounded": True}]
    answers = [
        {"observations": [{"evidence_id": "VF1", "supported": True, "observed": "People look alarmed.", "attachment": "Participants around the meeting table."}], "reason": "Visible expressions."},
        {"bindings": [{"caption_quote": "meeting", "observed_entity": "Participants at the table", "role_scope": "Same meeting", "evidence_ids": ["VF1"]}], "unmatched_roles": [], "reason": "Same event, differing state."},
        {"relation": "CONFLICT", "evidence_ids": ["VF1"],
         "condition_checks": [{"caption_quote": "calm", "image_state": "Alarmed expressions", "relation": "CONFLICT"}],
         "unestablished_conditions": [], "reason": "Alarm conflicts with calm."},
        {"decision_errors": [], "role_scope_errors": [], "alternative": "", "alternative_relation": "UNRESOLVED", "alternative_status": "NONE", "deciding_evidence_ids": [], "reason": "No supported alternative."},
    ]
    return image, proposal, ledger, answers


class Runtime:
    _last_generation_diagnostics = {}
    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []
        self.token_budgets = []
    def generate(self, image, prompt, max_new_tokens=None, json_schema=None):
        self.prompts.append(prompt)
        self.token_budgets.append(max_new_tokens)
        return json.dumps(self.answers.pop(0)), .01


class EvidenceReviewTests(unittest.TestCase):
    def test_dangling_generated_clause_cannot_be_accepted_as_evidence(self):
        self.assertEqual(unfinished_generated_field({"condition_checks": [{"image_state": "He appears shocked,"}]}),
                         "condition_checks[0].image_state")
        self.assertEqual(unfinished_generated_field({"caption_quote": "with the", "image_state": "A visible meeting."}), "")
        image, p, ledger, answers = fixture()
        wrong = deepcopy(answers[0]); wrong["observations"][0]["observed"] = "People look alarmed,"
        rt = Runtime([wrong, wrong])
        p["independent_verification"] = verify(rt, image, p, ledger)
        self.assertFalse(audit(p, ledger)["valid"])
        self.assertEqual(len(rt.prompts), 2)
        _, p, ledger = self.run_proof()
        p["independent_verification"]["obligations"]["visual"]["observations"][0]["observed"] = "People look alarmed,"
        self.assertFalse(audit(p, ledger)["valid"])

    def run_proof(self, mutate=None):
        image, p, ledger, answers = fixture()
        if mutate:
            mutate(answers)
        rt = Runtime(answers)
        p["independent_verification"] = verify(rt, image, p, ledger)
        return rt, p, ledger

    def test_same_entities_opposite_states_are_not_a_mapping_failure(self):
        rt, p, ledger = self.run_proof()
        self.assertTrue(audit(p, ledger)["valid"])
        self.assertEqual(len(rt.prompts), 4)
        self.assertFalse(audit(p, ledger)["actual_order_swap"])
        self.assertNotIn(p["bridge_statement"], rt.prompts[2])

    def test_unsupported_visual_premise_short_circuits(self):
        rt, p, ledger = self.run_proof(lambda a: a[0]["observations"][0].update(supported=False))
        self.assertEqual(len(rt.prompts), 1)
        self.assertFalse(audit(p, ledger)["valid"])

    def test_excess_citations_fail_before_spending_inference(self):
        image, p, ledger, answers = fixture()
        p["visual_evidence_ids"] = [f"VF{i}" for i in range(5)]
        rt = Runtime(answers)
        proof = verify(rt, image, p, ledger)
        self.assertEqual(proof["stopped_after"], "evidence_budget")
        self.assertEqual(len(rt.prompts), 0)

    def test_visual_output_budget_scales_with_requested_observations(self):
        image, p, ledger, answers = fixture()
        p["visual_evidence_ids"] = [f"VF{i}" for i in range(4)]
        observation = answers[0]["observations"][0]
        answers[0]["observations"] = [dict(observation, evidence_id=k, supported=False) for k in p["visual_evidence_ids"]]
        rt = Runtime(answers)
        verify(rt, image, p, ledger)
        self.assertEqual(rt.token_budgets, [832])

    def test_argument_repair_time_is_counted_once_as_verification(self):
        from engine.review_outcome import review_timing
        review = {"_generation_seconds": 12, "_independent_verification": {"_generation_seconds": 7},
                  "_argument_repair": {"_generation_seconds": 2,
                      "_generation_diagnostics": [{"elapsed_seconds": 2, "termination_reason": "TIMEOUT"}]}}
        timing = review_timing(review)
        self.assertEqual(timing["verification_seconds"], 9)
        self.assertEqual(timing["proposal_and_format_retry_seconds"], 3)
        self.assertEqual(timing["timeout_elapsed_seconds"], 2)

    def test_disabled_judge_is_not_an_execution_failure(self):
        result = summarize_tribunal([dict(initial_prediction="ENTAILS", prediction="ENTAILS",
            ground_truth="ENTAILS", judge_requested=False, judge_format_valid=False)])
        self.assertEqual(result["failed_reviews"], 0)
        self.assertEqual(result["reviews_not_requested"], 1)

    def test_mapping_cannot_be_certified_by_forged_boolean(self):
        rt, p, ledger = self.run_proof()
        mapping = p["independent_verification"]["obligations"]["mapping"]
        mapping.update(verified=True, unmatched_roles=["speaker"])
        self.assertFalse(audit(p, ledger)["entity_scope_verified"])

    def test_role_swap_blocks_an_otherwise_agreeing_decision(self):
        rt, p, ledger = self.run_proof(lambda a: a[3].update(role_scope_errors=["Speaker is reversed."]))
        self.assertFalse(audit(p, ledger)["valid"])
        plan = repair_plan({"_independent_verification": p["independent_verification"]})
        self.assertIn("Speaker is reversed", plan["agent1_questions"][0])
        self.assertTrue(plan["_usable"])

    def test_late_followup_is_skipped_without_changing_the_review(self):
        from engine.tribunal import followup_plan, repair_followup_plan
        _, p, _ = self.run_proof(lambda a: a[3].update(role_scope_errors=["Speaker is reversed."]))
        review = {"status": "FOLLOW_UP", "_format_valid": True,
                  "targeted_question": "Who is speaking?", "requested_follow_up": "ENTITY_BINDING",
                  "_independent_verification": p["independent_verification"],
                  "_case_budget": {"remaining_seconds": 28, "minimum_followup_seconds": 90}}
        self.assertEqual(followup_plan(review), {})
        self.assertEqual(repair_followup_plan(review), {})
        self.assertEqual(review["_follow_up_budget_status"], "SKIPPED_INSUFFICIENT_REMAINING_BUDGET")
        self.assertTrue(review["_format_valid"])
        review["_case_budget"]["remaining_seconds"] = 120
        self.assertTrue(repair_followup_plan(review)["_usable"])

    def test_empty_proposer_counter_does_not_override_new_alternative(self):
        rt, p, ledger = self.run_proof(lambda a: a[3].update(alternative="The scene is a performance.",
            alternative_relation="SUPPORT", alternative_status="UNRESOLVED", deciding_evidence_ids=["VF1"]))
        self.assertFalse(audit(p, ledger)["counter_resolved"])

    def test_none_status_cannot_hide_a_nonempty_counter(self):
        rt, p, ledger = self.run_proof(lambda a: a[3].update(alternative="A supported alternative."))
        self.assertFalse(audit(p, ledger)["valid"])

    def test_support_with_missing_qualifier_is_not_valid(self):
        def missing(a):
            a[2].update(relation="SUPPORT", unestablished_conditions=["calm"])
            a[2]["condition_checks"][0]["relation"] = "SUPPORT"
        rt, p, ledger = self.run_proof(missing)
        self.assertEqual(len(rt.prompts), 3)
        self.assertFalse(audit(p, ledger)["valid"])

    def test_unknown_citation_and_invented_caption_quote_block(self):
        for mutation in (lambda a: a[2].update(evidence_ids=["MISSING"]),
                         lambda a: a[2]["condition_checks"][0].update(caption_quote="relaxing")):
            rt, p, ledger = self.run_proof(mutation)
            self.assertFalse(audit(p, ledger)["valid"])

    def test_argument_repair_preserves_case_and_rechecks_only_changed_argument(self):
        image, p, ledger, answers = fixture()
        bad_challenge = deepcopy(answers[3])
        bad_challenge["decision_errors"] = ["The explanation should identify the contrast clearly."]
        rt = Runtime(answers[:3] + [bad_challenge, {"argument": "Alarmed participants are incompatible with a calm meeting."}, answers[3]])
        proof = verify(rt, image, p, ledger)
        before = deepcopy(p)
        edited = repair_argument(rt, image, p, proof)
        self.assertEqual(p, before)
        self.assertEqual(set(rt.answers[0]), set(answers[3]))
        self.assertTrue(edited["_format_valid"])
        p["bridge_statement"] = edited["argument"]
        p["independent_verification"] = verify(rt, image, p, ledger)
        self.assertTrue(audit(p, ledger)["valid"])
        self.assertEqual(len(rt.prompts), 6)  # Four checks, explanation edit, fresh challenge.
        self.assertEqual(p["independent_verification"]["generation_call_count"], 1)
        self.assertEqual(p["proposed_relation"], before["proposed_relation"])
        self.assertEqual(p["visual_evidence_ids"], before["visual_evidence_ids"])

    def test_changed_provenance_invalidates_verifier_cache(self):
        image, p, ledger, answers = fixture()
        rt = Runtime(answers + deepcopy(answers))
        verify(rt, image, p, ledger)
        ledger[0]["source"] = "another_extractor"
        verify(rt, image, p, ledger)
        self.assertEqual(len(rt.prompts), 8)

    def test_wrong_source_quote_gets_one_contract_repair_before_rejection(self):
        image, p, ledger, answers = fixture()
        wrong = deepcopy(answers[2]); wrong["condition_checks"][0]["caption_quote"] = "text printed on the image"
        rt = Runtime(answers[:2] + [wrong, answers[2], answers[3]])
        p["independent_verification"] = verify(rt, image, p, ledger)
        self.assertTrue(audit(p, ledger)["valid"])
        self.assertEqual(len(rt.prompts), 5)
        self.assertIn("NOT image text", rt.prompts[3])

    def test_omitted_requested_observation_gets_contract_repair(self):
        image, p, ledger, answers = fixture()
        wrong = deepcopy(answers[0]); wrong["observations"] = []
        rt = Runtime([wrong] + answers)
        p["independent_verification"] = verify(rt, image, p, ledger)
        self.assertTrue(audit(p, ledger)["valid"])
        self.assertEqual(len(rt.prompts), 5)

    def test_image_caption_and_evidence_changes_invalidate_proof(self):
        rt, p, ledger = self.run_proof()
        for key, value in (("source_caption", "Different caption."), ("image_sha256", "different")):
            changed = deepcopy(p); changed[key] = value
            self.assertFalse(audit(changed, ledger)["bound_to_current_case"])
        changed = deepcopy(ledger); changed[0]["grounded"] = False
        self.assertFalse(audit(p, changed)["bound_to_current_case"])

    def test_matching_votes_cannot_override_unsupported_argument(self):
        rt, p, ledger = self.run_proof(lambda a: a[3].update(decision_errors=["Accuracy is not established."]))
        self.assertFalse(audit(p, ledger)["valid"])

    def test_budget_accumulates_between_reviews_of_same_case(self):
        runtime = SimpleNamespace()
        with patch("engine.case_budget.time.perf_counter", side_effect=[10, 12]):
            with case_budget(runtime, "case", 3):
                pass
        with patch("engine.case_budget.time.perf_counter", side_effect=[20, 22, 23]):
            with case_budget(runtime, "case", 3):
                with self.assertRaises(CaseBudgetExceeded):
                    require_time(runtime)
        self.assertIsNone(runtime._review_budget_deadline)

    def test_budget_survives_model_reload_between_hearings(self):
        from engine.runtime_accounting import begin_accounting
        begin_accounting()
        with patch("engine.case_budget.time.perf_counter", side_effect=[10, 12]):
            with case_budget(SimpleNamespace(), "reloaded", 3):
                pass
        with patch("engine.case_budget.time.perf_counter", side_effect=[20, 22, 23]):
            reloaded = SimpleNamespace()
            with case_budget(reloaded, "reloaded", 3):
                with self.assertRaises(CaseBudgetExceeded):
                    require_time(reloaded)
        begin_accounting()

    def test_argument_repair_reuses_only_unchanged_proof_inputs(self):
        rt, p, ledger = self.run_proof()
        p["bridge_statement"] = "A revised explanation of the same visible alarm."
        rt.answers = [fixture()[3][-1]]
        p["independent_verification"] = verify(rt, fixture()[0], p, ledger)
        self.assertTrue(audit(p, ledger)["valid"])
        self.assertEqual(len(rt.prompts), 5)  # Only the changed argument audit reruns.
        self.assertEqual(p["independent_verification"]["generation_call_count"], 1)
        self.assertTrue(p["independent_verification"]["obligations"]["visual"]["_cache_hit"])
        changed = deepcopy(ledger); changed[0]["text"] = "The image has changed evidence."
        negative = deepcopy(fixture()[3][0]); negative["observations"][0]["supported"] = False
        rt.answers = [negative]
        p["independent_verification"] = verify(rt, fixture()[0], p, changed)
        self.assertFalse(audit(p, changed)["valid"])
        self.assertEqual(len(rt.prompts), 6)

    def test_compact_proposal_preserves_source_and_rejects_inconsistent_support(self):
        graph = source_identity_graph("The meeting is calm.")
        packet = {"source_caption": graph["source_caption"], "claim_agent": {"claim_graph": graph}}
        value = {"node_relations": [{"claim_node_id": "C1", "observation": "People look alarmed.", "role_scope": "Same meeting.",
            "condition_checks": [{"caption_quote": "calm", "image_state": "Alarmed expressions", "relation": "CONFLICT"}],
            "relation": "CONFLICT", "evidence_ids": ["VF1"], "unestablished_condition": ""}],
            "alternative": "", "decisive_reason": "Alarm conflicts with calm.", "follow_up": {"target": "NONE", "question": ""}, "context_requests": []}
        parsed = expand_proposal(json.dumps(value), graph, packet, {"VF1"}, set())
        self.assertTrue(parsed["_format_valid"])
        self.assertEqual(parsed["caption_premise"], graph["source_caption"])
        value["node_relations"][0].update(relation="SUPPORT", unestablished_condition="calm")
        self.assertFalse(expand_proposal(json.dumps(value), graph, packet, {"VF1"}, set())["_format_valid"])

    def test_quality_distinguishes_gate_from_judge(self):
        def row(initial, proposed, final, gold, valid=True):
            return dict(initial_prediction=initial, judge_verdict=proposed, prediction=final, ground_truth=gold, judge_format_valid=valid)
        result = summarize_tribunal([
            row("ENTAILS", "CONTRADICTS", "ENTAILS", "CONTRADICTS"),
            row("ENTAILS", "CONTRADICTS", "CONTRADICTS", "ENTAILS"),
            row("ENTAILS", "ABSTAIN", "ENTAILS", "ENTAILS", False)])
        self.assertEqual(result["proposal_precision"], .5)
        self.assertEqual(result["accepted_change_precision"], 0)
        self.assertEqual(result["failed_reviews"], 1)
        self.assertEqual(result["semantic_abstentions"], 0)


if __name__ == "__main__":
    unittest.main()
