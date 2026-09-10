"""Regression contracts for controls and reporting (not model accuracy tests)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from engine.control_conditions import aggregate, public_case
from engine.output_contracts import saturated_text_fields
from engine.runtime_accounting import begin_accounting, set_scope, record_generation, sample_accounting
from evaluation.evaluate_predictions import classification_summary, metric_row, evaluate_predictions
from evaluation.confidence_calibration import apply, METHOD_KEYS
from evaluation.human_review import agreement, blind_record, RATINGS
from engine.review_board import decision_grade_strength
from engine.final_artifact import final_artifact


class ResearchControlsTests(unittest.TestCase):
    def test_failed_phase_preserves_completed_count_and_records_failure(self):
        from run_figdebate import run_tracked_phase
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "progress.json")
            progress = dict(status="running", completed_samples=3)
            def fail():
                raise RuntimeError("finalization failed")
            with self.assertRaises(RuntimeError):
                run_tracked_phase(path, progress, "stagewise_execution", fail)
            saved = json.loads(Path(path).read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["completed_samples"], 3)
            self.assertEqual(saved["failure_type"], "RuntimeError")

    def test_interrupt_is_not_success_or_semantic_abstention(self):
        from run_figdebate import run_tracked_phase
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "progress.json")
            def interrupt():
                raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                run_tracked_phase(path, {"completed_samples": 1}, "execution", interrupt)
            self.assertEqual(json.loads(Path(path).read_text())["status"], "interrupted")

    def test_complete_cached_control_path_finishes_without_new_generation(self):
        from copy import deepcopy
        from PIL import Image
        from engine.batch_runner import StagewiseRunner
        runner = StagewiseRunner(debate_mode="disabled", judge_mode="disabled",
                                 semantic_bridge_mode="disabled", evidence_mode="disabled",
                                 control_mode="independent_vote")
        case = dict(relation="CONFLICT", _format_valid=True, _execution_status="SUCCEEDED",
                    decisive_question="none needed", claim_reading="Claim", decisive_observation="Fact", alternative="none")
        baseline = dict(label="ENTAILS", confidence=.8, explanation="Baseline")
        cached = dict(visual_output={}, language_output={}, evidence_ledger=[], initial_decision=baseline,
                      decision=baseline, timing={}, debate_level=0)
        def restore(stage, sample):
            return deepcopy({"visual_grounding": {}, "initial_reasoning": cached,
                             "visual_candidate": case, "text_candidate": case}[stage])
        captured = []
        with patch.object(runner.stage_checkpoints, "load", side_effect=restore), \
             patch("engine.batch_runner.MistralModel"), patch("engine.batch_runner.ClaimExtractionAgent"), \
             patch("engine.batch_runner.Arbiter"), patch("engine.batch_runner.GPUManager.clear"):
            timing = runner.run_samples([dict(index=0, raw={"id": "fixture"}, caption="Claim", image=Image.new("RGB", (2, 2)))],
                                        lambda _index, _raw, result, _seconds: captured.append(result))
        self.assertEqual(captured[0]["decision"]["label"], "CONTRADICTS")
        self.assertEqual(captured[0]["runtime_accounting"]["generation_requests"], 0)
        self.assertEqual(timing["stage_reuse_events"], [])

    def test_unusable_witness_text_is_withheld_from_judge(self):
        from engine.case_dossier import render_judge_dossier
        packet = render_judge_dossier(dict(source_caption="Claim", evidence_catalog=[],
            targeted_hearing={"agent2_critique": dict(requirements_valid=False,
                support_requirement="Unreliable secret", conflict_requirement="Bad secret")},
            claim_agent=dict(expected_visual_state="Unsupported state", opposite_visual_state="Bad state",
                             claim_contract={"fully_valid": False, "field_groups": {"relation": {"valid": False}}})))
        self.assertNotIn("secret", json.dumps(packet))
        self.assertNotIn("Unsupported state", json.dumps(packet))
        self.assertEqual(packet["targeted_hearing"]["agent2_critique"]["status"], "UNUSABLE_WITNESS_RECORD")

    def test_primary_metric_keeps_unanswered_and_unverified_answers(self):
        rows = pd.DataFrame([
            dict(ground_truth="ENTAILS", prediction="ENTAILS", _valid=False),
            dict(ground_truth="CONTRADICTS", prediction=None, _valid=False),
            dict(ground_truth="CONTRADICTS", prediction="ENTAILS", _valid=True)])
        result = classification_summary(rows)
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["per_label"]["CONTRADICTS"]["support"], 2)
        self.assertEqual(result["confusion_matrix"]["matrix"], [[1, 0, 0], [1, 0, 1]])
        self.assertAlmostEqual(metric_row(rows, "phenomenon", "humor")["accuracy"], 1 / 3)

    def test_all_invalid_report_is_still_written(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.csv"
            pd.DataFrame([dict(id="a", ground_truth="ENTAILS", prediction=None,
                               phenomenon="humor", final_decision_valid=False,
                               reference_explanation="reference", final_reason="unavailable")]).to_csv(path, index=False)
            result = evaluate_predictions(str(path), temp)
            self.assertEqual(result["accuracy"], 0)
            self.assertEqual(result["unanswered_count"], 1)
            self.assertIsNone(result["contract_valid_subset_accuracy"])
            self.assertEqual(pd.read_csv(Path(temp) / "confusion_matrix.csv").iloc[0, -1], 1)

    def test_bad_gold_cannot_be_scored(self):
        with self.assertRaises(ValueError):
            classification_summary(pd.DataFrame([dict(ground_truth="unknown", prediction="ENTAILS")]))

    def test_voting_requires_majority_of_all_agents(self):
        base = dict(label="ENTAILS", explanation="Original", confidence=.8)
        case = dict(relation="CONFLICT", _format_valid=True, _execution_status="SUCCEEDED")
        decision, audit = aggregate(base, [case, {}], "independent_vote")
        self.assertEqual(decision, base)
        self.assertTrue(audit["baseline_fallback"])
        decision, audit = aggregate(base, [case, case], "independent_vote")
        self.assertEqual(decision["label"], "CONTRADICTS")
        self.assertFalse(decision["_final_decision_valid"])
        self.assertIsNone(decision["confidence"])
        self.assertFalse(audit["compute_match_established"])
        self.assertEqual(base["label"], "ENTAILS")

    def test_peer_case_hides_private_fields(self):
        case = dict(relation="SUPPORT", _format_valid=True, _execution_status="SUCCEEDED",
                    ground_truth="secret", initial_prediction="secret", confidence=.99)
        self.assertNotIn("ground_truth", public_case(case))
        self.assertNotIn("confidence", public_case(case))

    def test_accounting_resets_and_failed_tokens_are_unknown(self):
        class Runtime:
            @record_generation("fixture")
            def generate(self, fail=False):
                if fail:
                    raise RuntimeError("fixture failure")
                self._last_generation_diagnostics = dict(input_tokens=3, generated_tokens=5)
                return "{}", .1
        begin_accounting()
        set_scope("a", "test")
        runtime = Runtime()
        runtime.generate()
        with self.assertRaises(RuntimeError):
            runtime.generate(True)
        audit = sample_accounting("a")
        self.assertEqual(audit["generation_requests"], 2)
        self.assertEqual(audit["output_tokens_known"], 5)
        self.assertFalse(audit["generation_tokens_complete"])
        self.assertIsNone(audit["events"][1]["output_tokens"])
        begin_accounting()
        self.assertEqual(sample_accounting("a")["generation_requests"], 0)

    def test_saturated_unfinished_clause_is_not_valid_prose(self):
        schema = {"type": "object", "properties": {"reason": {"type": "string", "maxLength": 20}}}
        self.assertEqual(saturated_text_fields(json.dumps({"reason": "x" * 18 + " a"}), schema), ["reason"])
        self.assertEqual(saturated_text_fields(json.dumps({"reason": "A complete claim."}), schema), [])

    def test_calibrator_uses_exported_score_and_never_changes_label(self):
        config = {key: "fixture" for key in METHOD_KEYS}
        artifact = dict(method_identity=config, calibration_image_groups=["train"], coefficient=1., intercept=0.)
        record = dict(id="a", image_group_id="heldout", final_confidence=.5, prediction="ENTAILS")
        result = apply([record], config, artifact)[0]
        self.assertEqual(result["uncalibrated_score"], .5)
        self.assertFalse(result["prediction_changed"])
        with self.assertRaises(ValueError):
            apply([dict(record, image_group_id="train")], config, artifact)

    def test_human_packet_contains_no_gold_or_method(self):
        packet = blind_record(dict(id="a", ground_truth="secret", decision_method="secret", trace={}), "Caption", "a.png")
        self.assertNotIn("secret", json.dumps(packet))
        self.assertTrue(all(value is None for value in packet["ratings"].values()))
        with self.assertRaisesRegex(ValueError, "raters"):
            agreement([packet], [packet])
        first = dict(packet, rater_id="one", ratings={key: True for key in RATINGS})
        second = dict(packet, rater_id="two", ratings={key: False for key in RATINGS})
        self.assertEqual(agreement([first], [second])["roles_preserved"]["agreement"], 0)

    def test_new_ids_cannot_multiply_exact_same_observation(self):
        root = dict(text="The line rises.", grounded=True, relation="NEUTRAL")
        proof = dict(text="The rise supports the claim.", grounded=True, relation="SUPPORT", decision_grade=True, reliability=.8)
        ledger = [dict(root, id="A"), dict(root, id="B"),
                  dict(proof, id="C", derived_from_ids=["A"]), dict(proof, id="D", derived_from_ids=["B"])]
        self.assertEqual(decision_grade_strength(ledger, "SUPPORT"), .8)
        ledger[0]["panel"] = "left"
        ledger[1]["panel"] = "right"
        self.assertEqual(decision_grade_strength(ledger, "SUPPORT"), 1.6)

    def test_final_trace_contains_ancestors_and_loses_superseded_chain(self):
        root = dict(id="A", text="The line rises.", grounded=True)
        proof = dict(id="B", text="Supports rise.", relation="SUPPORT", grounded=True,
                     decision_grade=True, derived_from_ids=["A"])
        result = dict(decision=dict(label="ENTAILS", _model_cited_evidence_ids=["B"]), evidence_ledger=[root, proof])
        artifact = final_artifact("The line rises.", result)
        self.assertEqual([row["id"] for row in artifact["supporting_ancestors"]], ["A"])
        root["lifecycle_status"] = "SUPERSEDED"
        self.assertFalse(final_artifact("The line rises.", result)["evidence_chain_complete"])

    def test_typed_witness_preserves_raw_claim_for_each_semantic_dimension(self):
        from agents.claim_extraction import ClaimExtractionAgent
        from engine.claim_witness import audit_claim_witness
        from engine.claim_semantics import CHECKS
        witness = dict(stance="CHALLENGE", support_requirement="The complete claim holds.",
                       conflict_requirement="The contrary state holds.", figurative_mechanism="unclear", ambiguity="none", reason="Audit.")
        # Mocked audit tests routing and repair, not semantic accuracy of these captions.
        captions = ["The cat chases the dog.", "The dog does not chase the cat.",
                    "Exactly three cups remain.", "She is taller than him.",
                    "In the right panel, he has already left.", "What a wonderfully easy failure!",
                    "That was a very difficult failure."]
        for caption in captions:
            agent = object.__new__(ClaimExtractionAgent)
            audit = dict(**{key: True for key in CHECKS}, _format_valid=True)
            with patch.object(agent, "_generate_section", return_value=(witness, json.dumps(witness), .1, {"schema_valid": True})) as generation, \
                 patch("engine.claim_witness.audit_core", return_value=audit) as verifier:
                from tests.critical_fixture import core
                result = audit_claim_witness(agent, caption, {"claim_fields": core(caption)})
            self.assertTrue(result["requirements_valid"])
            self.assertEqual(generation.call_args.args[1], caption)
            self.assertEqual(verifier.call_args.args[2]["caption_proposition"], caption)

    def test_witness_semantic_failure_stays_failure_after_bounded_retry(self):
        from agents.claim_extraction import ClaimExtractionAgent
        from engine.claim_witness import audit_claim_witness
        from engine.claim_semantics import CHECKS
        agent = object.__new__(ClaimExtractionAgent)
        fields = dict(stance="ENDORSE", support_requirement="It recovered.", conflict_requirement="It declined.")
        audit = dict(**{key: True for key in CHECKS}, _format_valid=True)
        audit["modifiers_scope_preserved"] = False
        with patch.object(agent, "_generate_section", return_value=(fields, "fixture", .1, {"schema_valid": True})) as generation, \
             patch("engine.claim_witness.audit_core", return_value=audit):
            result = audit_claim_witness(agent, "It recovered easily.", "")
        self.assertEqual(generation.call_count, 2)
        self.assertFalse(result["requirements_valid"])
        self.assertFalse(result["deterministic_requirement_repair"])


if __name__ == "__main__":
    unittest.main()
