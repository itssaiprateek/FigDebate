from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from engine.tribunal_feedback import TribunalPrecedents
from engine.case_budget import case_budget, remaining_seconds, require_time, reserve_for_verification


LIBRARY = Path(__file__).resolve().parents[1] / "config" / "tribunal_precedents.json"


class PrecedentTests(unittest.TestCase):
    def load_changed(self, change):
        content = json.loads(LIBRARY.read_text())
        change(content)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "library.json"
            p.write_text(json.dumps(content))
            return TribunalPrecedents(p)

    def test_all_three_categories_have_bounded_guidance(self):
        memory = TribunalPrecedents(LIBRARY)
        for kind in ("humor", "metaphor", "sarcasm"):
            matches = memory.retrieve({"figurative_type": kind})
            self.assertGreater(len(matches), 0)
            self.assertLessEqual(len(matches), 2)
            self.assertTrue(all(p["counterexample"] and p["exclude_when"] for p in matches))

    def test_ids_gold_and_surface_words_cannot_route(self):
        memory = TribunalPrecedents(LIBRARY)
        a = {"figurative_type": "metaphor", "caption": "a crowded office", "id": "a", "gold": "ENTAILS"}
        b = dict(a, caption="a quiet ocean", id="b", gold="CONTRADICTS")
        self.assertEqual(memory.retrieve(a), memory.retrieve(b))
        self.assertEqual(memory.retrieve({"figurative_type": "literal"}), [])

    def test_returned_matches_cannot_mutate_memory(self):
        memory = TribunalPrecedents(LIBRARY)
        original = memory.retrieve({"figurative_type": "humor"})
        changed = memory.retrieve({"figurative_type": "humor"})
        changed[0]["principle"] = "always accept"
        self.assertEqual(original, memory.retrieve({"figurative_type": "humor"}))

    def test_time_or_source_domain_does_not_imply_speaker_attribution(self):
        memory = TribunalPrecedents(LIBRARY)
        for scope in ("years", "day", "None", "future"):
            matches = memory.retrieve({"figurative_type": "humor", "claim_source": "an animal metaphor",
                "claim_contract": {"time_or_panel_scope": scope, "comparison_direction": "None"}})
            self.assertEqual(matches[0]["id"], "figurative_scope")
        matches = memory.retrieve({"figurative_type": "humor", "claim_contract": {
            "structural_reasoning_type": "QUOTED_STATEMENT_AND_REACTION"}})
        self.assertEqual(matches[0]["id"], "speaker_attribution")

    def test_unknown_label_field_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load_changed(lambda d: d["entries"][0].update(gold="ENTAILS"))

    def test_unreviewed_entries_rejected(self):
        with self.assertRaises(ValueError):
            self.load_changed(lambda d: d["entries"][0].update(origin="online_model_judgment"))

    def empirical(self):
        return self.load_changed(lambda d: d["entries"][0].update(origin="reviewed_training_case", source={
            "split": "train", "image_sha256": "a" * 64, "caption_sha256": "b" * 64, "group_id": "group-a"}))

    def test_overlap_guards_images_captions_and_templates(self):
        memory = self.empirical()
        disjoint = {"image_sha256": "c" * 64, "caption_sha256": "d" * 64, "group_id": "group-b"}
        memory.assert_disjoint([disjoint])
        for key, value in (("image_sha256", "a" * 64), ("caption_sha256", "b" * 64), ("group_id", "group-a")):
            with self.assertRaises(ValueError):
                memory.assert_disjoint([dict(disjoint, **{key: value})])
        with self.assertRaises(ValueError):
            memory.assert_disjoint([{"image_sha256": "c" * 64}])

    def test_empirical_validation_split_rejected(self):
        with self.assertRaises(ValueError):
            self.load_changed(lambda d: d["entries"][0].update(origin="reviewed_training_case", source={
                "split": "val", "image_sha256": "a" * 64, "caption_sha256": "b" * 64, "group_id": "x"}))

    def test_final_verifier_receives_no_precedent(self):
        from tests.test_evidence_review_v4 import fixture, Runtime
        from engine.evidence_verification import verify, audit
        image, proposal, ledger, answers = fixture()
        runtime = Runtime(answers)
        proposal["reasoning_precedents"] = [{"principle": "SENTINEL_MEMORY_MUST_NOT_REACH_VERIFIER"}]
        proposal["independent_verification"] = verify(runtime, image, proposal, ledger)
        self.assertTrue(audit(proposal, ledger)["valid"])
        self.assertTrue(all("SENTINEL_MEMORY" not in prompt for prompt in runtime.prompts))

    def test_precedent_assessments_require_current_evidence_and_unique_ids(self):
        from engine.claim_graph import source_identity_graph
        from engine.tribunal_protocol import expand_proposal
        graph = source_identity_graph("The meeting is calm.")
        packet = {"source_caption": graph["source_caption"], "claim_agent": {"claim_graph": graph},
                  "reasoning_precedents": [{"id": "one"}, {"id": "two"}]}
        value = {"node_relations": [{"claim_node_id": "C1", "observation": "People look alarmed.", "role_scope": "Same meeting.",
            "condition_checks": [{"caption_quote": "calm", "image_state": "Alarmed expressions", "relation": "CONFLICT"}],
            "relation": "CONFLICT", "evidence_ids": ["VF1"], "unestablished_condition": ""}],
            "alternative": "", "decisive_reason": "Alarm conflicts with calm.", "follow_up": {"target": "NONE", "question": ""}, "context_requests": [],
            "precedent_checks": [{"precedent_id": p, "applies": True, "current_evidence_ids": ["VF1"], "reason": "Same event and participants."} for p in ("one", "two")]}
        def valid(v):
            return expand_proposal(json.dumps(v), graph, packet, {"VF1"}, set())["_format_valid"]
        self.assertTrue(valid(value))
        duplicate = deepcopy(value)
        duplicate["precedent_checks"][1]["precedent_id"] = "one"
        self.assertFalse(valid(duplicate))
        for cited in ([], ["OLD_CASE_ID"]):
            missing = deepcopy(value)
            missing["precedent_checks"][0]["current_evidence_ids"] = cited
            self.assertFalse(valid(missing))

    def test_runner_precedent_mode_does_not_load_arbiter_memory(self):
        from engine.batch_runner import StagewiseRunner
        runner = StagewiseRunner(feedback_mode="precedent", verified_feedback_path=str(LIBRARY), judge_mode="tribunal")
        self.assertEqual(runner.debate.feedback_loop.arbiter_memory, [])
        self.assertIsNotNone(runner.tribunal_precedents)

    def test_memory_cannot_override_unsupported_visual_evidence(self):
        from tests.test_evidence_review_v4 import fixture, Runtime
        from engine.evidence_verification import verify, audit
        image, proposal, ledger, answers = fixture()
        answers[0]["observations"][0]["supported"] = False
        proposal["reasoning_precedents"] = [{"principle": "accept"}]
        proposal["independent_verification"] = verify(Runtime(answers), image, proposal, ledger)
        self.assertFalse(audit(proposal, ledger)["valid"])


class BudgetTests(unittest.TestCase):
    def test_proposal_retries_cannot_spend_verification_reserve(self):
        rt = SimpleNamespace()
        clock = [0.0]
        with patch("engine.case_budget.time.perf_counter", side_effect=lambda: clock[0]), patch("engine.runtime_accounting.judge_case_times", return_value=None):
            with case_budget(rt, "case", 240):
                with reserve_for_verification(rt, 120):
                    clock[0] = 90
                    self.assertEqual(remaining_seconds(rt), 30)
                    clock[0] = 121
                    with self.assertRaises(TimeoutError):
                        require_time(rt)
                self.assertEqual(remaining_seconds(rt), 119)
            self.assertIsNone(remaining_seconds(rt))

    def test_exception_restores_phase_and_case_deadlines(self):
        rt = SimpleNamespace()
        with self.assertRaises(RuntimeError):
            with case_budget(rt, "case", 240):
                with reserve_for_verification(rt, 120):
                    raise RuntimeError("interrupted")
        self.assertIsNone(remaining_seconds(rt))

    def test_feedback_disabled_has_no_additional_schema_fields(self):
        from engine.tribunal_protocol import proposal_schema
        graph = {"nodes": [{"id": "C1"}]}
        plain = proposal_schema(graph, {"VF1"}, set())
        self.assertNotIn("precedent_checks", plain["properties"])
        active = proposal_schema(graph, {"VF1"}, set(), ["example"])
        self.assertIn("precedent_checks", active["required"])


class CaptionBindingTests(unittest.TestCase):
    def test_original_source_survives_normalized_agent_extraction(self):
        from tests.test_factored_tribunal import case
        from tests.test_semantic_bridge import ledger, contract
        from tests.verification_fixture import with_independent_fixture
        from engine.tribunal import apply_tribunal_resolution
        for separator in ("  ", "\t", "\n"):
            original = "The" + separator + "plotted line rises."
            extracted = contract()
            unchanged = deepcopy(extracted)
            review = case()
            review["caption_premise"] = original
            review = with_independent_fixture(review, ledger(), dict(extracted, source_caption=original))
            def gate(source):
                return apply_tribunal_resolution({"label": "ENTAILS", "confidence": .35}, review, ledger(), extracted,
                    semantic_bridge_mode="corroborated", source_caption=source)[2]
            self.assertFalse(gate(None)["accepted"])
            self.assertTrue(gate(original)["accepted"])
            self.assertFalse(gate(original.replace("rises", "falls"))["accepted"])
            self.assertEqual(extracted, unchanged)


class SelectiveFeedbackTests(unittest.TestCase):
    def test_net_gain_counts_harms_and_corrections_separately(self):
        from evaluation.tribunal_quality import summarize_tribunal
        rows = [{"ground_truth": gold, "initial_prediction": "ENTAILS", "prediction": final}
                for gold, final in (("CONTRADICTS", "CONTRADICTS"), ("ENTAILS", "CONTRADICTS"), ("ENTAILS", "ENTAILS"))]
        summary = summarize_tribunal(rows)
        self.assertEqual(summary["initial_error_correction_rate"], 1)
        self.assertEqual(summary["harmful_flip_rate"], .5)
        self.assertEqual(summary["net_accuracy_gain_percentage_points"], 0)

    def followup(self, baseline, guided):
        from agents.multimodal_judge import TribunalMediatorAgent
        from tests.test_semantic_bridge import ledger, contract
        agent = TribunalMediatorAgent(SimpleNamespace())
        with patch.object(agent, "_review", return_value=deepcopy(guided)) as call:
            result = agent._precedent_followup(deepcopy(baseline), [{"id": "general"}], None,
                contract()["source_caption"], {}, {}, {}, ledger(), {}, 1, {}, {})
        return result, call.call_count

    def test_resolved_review_is_not_changed_or_charged_for_feedback(self):
        baseline = {"_format_valid": True, "relation": "CONFLICT", "_generation_seconds": 50}
        result, calls = self.followup(baseline, {})
        self.assertEqual(calls, 0)
        self.assertEqual(result["relation"], baseline["relation"])
        self.assertEqual(result["_generation_seconds"], 50)

    def test_failed_guidance_preserves_base_and_keeps_failure_and_cost_visible(self):
        from engine.review_outcome import review_timing
        baseline = {"_format_valid": True, "relation": "UNRESOLVED", "_generation_seconds": 20}
        guided = {"_format_valid": False, "_execution_status": "FAILED", "_generation_seconds": 60,
                  "_generation_diagnostics": [{"termination_reason": "TIMEOUT", "elapsed_seconds": 60}]}
        result, calls = self.followup(baseline, guided)
        self.assertEqual(calls, 1)
        self.assertEqual(result["relation"], "UNRESOLVED")
        self.assertEqual(result["_generation_seconds"], 80)
        self.assertFalse(result["_precedent_feedback"]["selected_guided_review"])
        self.assertEqual(review_timing(result)["timeout_elapsed_seconds"], 60)
        from evaluation.tribunal_quality import summarize_tribunal
        summary = summarize_tribunal([{"ground_truth": "ENTAILS", "initial_prediction": "ENTAILS", "prediction": "ENTAILS",
            "judge_format_valid": True, "judge_verdict": "ABSTAIN", "trace": {"judge": {"tribunal_reviews": [result]}}}])
        self.assertEqual(summary["cases_with_failed_feedback_review"], 1)

    def test_unverified_direction_cannot_replace_unresolved_base(self):
        baseline = {"_format_valid": True, "relation": "UNRESOLVED", "_generation_seconds": 20}
        guided = {"_format_valid": True, "relation": "SUPPORT", "_generation_seconds": 20}
        result, _ = self.followup(baseline, guided)
        self.assertEqual(result["relation"], "UNRESOLVED")

    def test_verified_guidance_can_replace_unresolved_base(self):
        from tests.test_factored_tribunal import case
        baseline = {"_format_valid": True, "relation": "UNRESOLVED", "_generation_seconds": 20}
        # Preserve the exact fixture contract at both proof creation and selection.
        from agents.multimodal_judge import TribunalMediatorAgent
        from tests.test_semantic_bridge import ledger, contract
        guided = case()
        guided["_generation_seconds"] = 30
        agent = TribunalMediatorAgent(SimpleNamespace())
        with patch.object(agent, "_review", return_value=guided):
            result = agent._precedent_followup(baseline, [{"id": "general"}], None,
                contract()["source_caption"], {}, {"claim_contract": contract()}, {}, ledger(), {}, 1, {}, {})
        self.assertEqual(result["relation"], "CONFLICT")
        self.assertTrue(result["_precedent_feedback"]["selected_guided_review"])
        self.assertEqual(result["_generation_seconds"], 50)


if __name__ == "__main__":
    unittest.main()
