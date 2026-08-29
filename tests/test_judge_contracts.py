import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from agents.multimodal_judge import (
    MultimodalJudgeAgent,
    MultimodalMediatorAgent,
    build_judge_packet,
    build_judge_prompt,
    build_mediation_packet,
    build_mediation_prompt,
)
from engine.debate import DebateEngine
from engine.judge_review import (
    apply_judge_review,
    judge_feedback_candidate,
    judge_request_reasons,
)
from engine.review_board import review_revision
from evaluation.evaluate_predictions import evaluate_predictions
from utils.judge_parser import parse_judge_response, parse_mediation_response
from models.hub_source import cached_snapshot_or_hub


def judgment(verdict="CONTRADICTS", confidence=0.9, evidence_ids=None):
    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence_ids": list(evidence_ids or []),
        "visual_observations": ["The plotted line falls from left to right."],
        "reason": "The cited direction directly conflicts with the caption.",
        "_format_valid": True,
        "_format_error": "",
        "_valid_evidence_ids": list(evidence_ids or []),
        "_invalid_evidence_ids": [],
    }


class JudgeParserTests(unittest.TestCase):
    def test_accepts_exact_json_contract(self):
        raw = json.dumps({
            "verdict": "ENTAILS",
            "confidence": 0.82,
            "evidence_ids": ["vf001", "VF001"],
            "visual_observations": ["A line rises."],
            "reason": "The visible trend matches the claim.",
        })
        parsed = parse_judge_response(raw)
        self.assertTrue(parsed["_format_valid"])
        self.assertEqual(parsed["verdict"], "ENTAILS")
        self.assertEqual(parsed["evidence_ids"], ["VF001"])

    def test_rejects_explanation_outside_json(self):
        parsed = parse_judge_response(
            '{"verdict":"ABSTAIN","confidence":0.5,"evidence_ids":[],'
            '"visual_observations":[],"reason":"unclear"} extra'
        )
        self.assertFalse(parsed["_format_valid"])

    def test_rejects_unexpected_fields(self):
        parsed = parse_judge_response(json.dumps({
            "verdict": "ENTAILS",
            "confidence": 0.9,
            "evidence_ids": ["VF001"],
            "visual_observations": ["A line rises."],
            "reason": "supported",
            "current_label": "ENTAILS",
        }))
        self.assertFalse(parsed["_format_valid"])

    def test_accepts_exact_mediation_contract(self):
        parsed = parse_mediation_response(json.dumps({
            "status": "MEDIATE",
            "provisional_verdict": "CONTRADICTS",
            "confidence": 0.84,
            "evidence_ids": ["vf001"],
            "issue": "The object-to-text binding is unresolved.",
            "agent1_question": "Which label belongs to the left object?",
            "agent2_question": "Which polarity does the caption assert?",
            "verification_request": "Verify the left/right binding.",
        }))
        self.assertTrue(parsed["_format_valid"])
        self.assertEqual(parsed["evidence_ids"], ["VF001"])

    def test_mediation_abstention_cannot_hide_directional_vote(self):
        parsed = parse_mediation_response(json.dumps({
            "status": "ABSTAIN",
            "provisional_verdict": "ENTAILS",
            "confidence": 0.5,
            "evidence_ids": [],
            "issue": "Unclear.",
            "agent1_question": "",
            "agent2_question": "",
            "verification_request": "",
        }))
        self.assertFalse(parsed["_format_valid"])

    def test_mediation_requires_a_targeted_question(self):
        parsed = parse_mediation_response(json.dumps({
            "status": "MEDIATE",
            "provisional_verdict": "ABSTAIN",
            "confidence": 0.5,
            "evidence_ids": [],
            "issue": "Unresolved relation",
            "agent1_question": "",
            "agent2_question": "",
            "verification_request": "",
        }))
        self.assertFalse(parsed["_format_valid"])


class JudgePacketTests(unittest.TestCase):
    def test_packet_has_no_primary_or_gold_label(self):
        packet = build_judge_packet(
            "Sales rose.",
            {"visual_facts": ["A chart is visible."]},
            {"caption_proposition": "Sales rose."},
            {"recommendation": "semantic review"},
            [{
                "id": "VF001", "source": "agent1", "type": "visual_fact",
                "text": "A chart is visible.", "relation": "NEUTRAL",
                "grounded": True,
            }],
            {},
        )
        serialized = json.dumps(packet).casefold()
        self.assertNotIn("ground_truth", serialized)
        self.assertNotIn("primary_decision", serialized)
        self.assertNotIn("current_label", serialized)
        self.assertIn("missing support alone is not contradiction", build_judge_prompt(packet).casefold())

    def test_agent_marks_unknown_evidence_ids(self):
        class Runtime:
            @staticmethod
            def generate(_image, _prompt):
                return json.dumps({
                    "verdict": "ENTAILS",
                    "confidence": 0.8,
                    "evidence_ids": ["VF001", "FAKE999"],
                    "visual_observations": ["A rising line is visible."],
                    "reason": "The line rises.",
                }), 0.1

        result = MultimodalJudgeAgent(Runtime()).analyze(
            object(), "Sales rose.", {}, {}, {},
            [{"id": "VF001"}], {},
        )
        self.assertEqual(result["_valid_evidence_ids"], ["VF001"])
        self.assertEqual(result["_invalid_evidence_ids"], ["FAKE999"])

    def test_mediator_receives_complete_ledger_but_no_labels(self):
        ledger = [
            {
                "id": f"VF{index:03d}", "source": "agent1",
                "type": "visual_fact", "text": f"Fact {index}",
                "relation": "NEUTRAL", "grounded": True,
            }
            for index in range(1, 46)
        ]
        packet = build_mediation_packet(
            "Caption", {}, {}, {}, ledger
        )
        self.assertEqual(len(packet["evidence_ledger"]), 45)
        serialized = json.dumps(packet).casefold()
        self.assertNotIn("ground_truth", serialized)
        self.assertNotIn("primary_decision", serialized)

    def test_mediator_marks_unknown_evidence_ids(self):
        class Runtime:
            @staticmethod
            def generate(_image, _prompt, max_new_tokens=None):
                return json.dumps({
                    "status": "MEDIATE",
                    "provisional_verdict": "ENTAILS",
                    "confidence": 0.8,
                    "evidence_ids": ["VF001", "FAKE999"],
                    "issue": "Direction",
                    "agent1_question": "Does the line rise?",
                    "agent2_question": "Does the caption assert a rise?",
                    "verification_request": "Verify the slope.",
                }), 0.1

        result = MultimodalMediatorAgent(Runtime()).analyze(
            object(), "Sales rose.", {}, {}, {}, [{"id": "VF001"}],
        )
        self.assertEqual(result["_valid_evidence_ids"], ["VF001"])
        self.assertEqual(result["_invalid_evidence_ids"], ["FAKE999"])
        self.assertFalse(result["_usable"])

    def test_debate_prompts_hide_mediator_vote_and_reason(self):
        plan = {
            "status": "MEDIATE", "provisional_verdict": "CONTRADICTS",
            "agent1_questions": ["Inspect UNIQUE_VISUAL_QUESTION."],
            "agent2_questions": ["Resolve UNIQUE_CLAIM_QUESTION."],
            "disputed_issues": [], "verification_requests": [],
            "reason": "SECRET_MEDIATOR_REASON",
            "_format_valid": True, "_usable": True,
        }
        engine = DebateEngine.__new__(DebateEngine)
        visual_prompt = engine.build_agent1_challenge_prompt(
            {}, {}, {}, plan
        )
        claim_prompt = engine.build_agent2_challenge_prompt({}, {}, plan)
        self.assertIn("UNIQUE_VISUAL_QUESTION", visual_prompt)
        self.assertIn("UNIQUE_CLAIM_QUESTION", claim_prompt)
        self.assertNotIn("SECRET_MEDIATOR_REASON", visual_prompt + claim_prompt)


class JudgeGateTests(unittest.TestCase):
    def setUp(self):
        self.ledger = [
            {
                "id": "VS001", "source": "agent1", "type": "visual_relation",
                "text": "The plotted line rises.", "relation": "SUPPORT",
                "grounded": True, "decision_grade": True,
            },
            {
                "id": "VC001", "source": "comparator", "type": "direct_conflict",
                "text": "The plotted line falls.", "relation": "CONFLICT",
                "grounded": True, "decision_grade": True,
            },
            {
                "id": "VC002", "source": "targeted_region_verifier",
                "type": "verified_relation", "text": "End value is below start value.",
                "relation": "CONFLICT", "grounded": True,
                "decision_grade": True,
            },
        ]
        self.current = {
            "label": "ENTAILS",
            "confidence": 0.8,
            "explanation": "Original decision.",
            "decision_method": "position_balanced_semantic",
            "_model_cited_evidence_ids": ["VS001"],
            "_final_decision_valid": True,
        }

    def test_shadow_mode_never_changes_decision(self):
        output, review = apply_judge_review(
            self.current, judgment(evidence_ids=["VC001", "VC002"]),
            self.ledger, mode="shadow",
        )
        self.assertEqual(output, self.current)
        self.assertFalse(review["accepted"])

    def test_appellate_rejects_unresolved_equal_evidence(self):
        output, review = apply_judge_review(
            self.current, judgment(evidence_ids=["VC001"]),
            self.ledger, mode="appellate",
        )
        self.assertEqual(output["label"], "ENTAILS")
        self.assertEqual(review["reason"], "unresolved_opposing_decision_grade_evidence")

    def test_appellate_accepts_stronger_existing_directional_evidence(self):
        output, review = apply_judge_review(
            self.current, judgment(evidence_ids=["VC001", "VC002"]),
            self.ledger, mode="appellate",
        )
        self.assertEqual(output["label"], "CONTRADICTS")
        self.assertEqual(output["decision_method"], "multimodal_judge_appellate")
        self.assertTrue(output["_review_board"]["directionally_grounded"])
        self.assertTrue(review["accepted"])

    def test_appellate_rejects_low_confidence(self):
        output, review = apply_judge_review(
            self.current,
            judgment(confidence=0.7, evidence_ids=["VC001", "VC002"]),
            self.ledger,
            mode="appellate",
        )
        self.assertEqual(output["label"], "ENTAILS")
        self.assertEqual(review["reason"], "judge_confidence_below_appellate_threshold")

    def test_routing_escalates_unsupported_final_decision(self):
        result = {
            "decision": {
                "label": "ENTAILS", "confidence": 0.55,
                "_final_decision_valid": True,
                "_review_board": {"directionally_grounded": False},
            },
            "comparison": {},
            "debate_triggered": False,
        }
        reasons = judge_request_reasons(result)
        self.assertIn("final_decision_not_directionally_grounded", reasons)
        self.assertIn("low_final_confidence", reasons)

    def test_judge_disagreement_is_review_candidate_not_memory_update(self):
        candidate = judge_feedback_candidate(
            judgment(verdict="CONTRADICTS", evidence_ids=["VC001"]),
            "ENTAILS",
        )
        self.assertTrue(candidate["recorded"])
        self.assertEqual(candidate["role"], "human_or_gold_verified_review_only")
        self.assertFalse(candidate["memory_update_applied"])

    def test_mediated_gate_accepts_only_verified_directional_tie(self):
        ledger = [
            {
                "id": "VS001", "source": "comparator", "type": "direct_support",
                "text": "The line rises.", "relation": "SUPPORT",
                "grounded": True, "decision_grade": True,
            },
            {
                "id": "VF001", "source": "agent1", "type": "visual_fact",
                "text": "The right endpoint is lower.", "relation": "NEUTRAL",
                "grounded": True, "decision_grade": False,
            },
            {
                "id": "TV001", "source": "targeted_region_verifier",
                "type": "verified_region_relation",
                "text": "The right endpoint is lower than the left endpoint.",
                "relation": "CONFLICT", "grounded": True,
                "decision_grade": True,
            },
        ]
        candidate = {
            **self.current,
            "label": "CONTRADICTS",
            "_model_cited_evidence_ids": ["TV001"],
        }
        visual_review = {
            "recommendation": "CONTRADICTS", "specific_evidence": True,
            "reason": "The right endpoint is lower than the left endpoint.",
        }
        mediation = {
            "status": "MEDIATE", "provisional_verdict": "CONTRADICTS",
            "confidence": 0.85, "_format_valid": True, "_usable": True,
            "_valid_evidence_ids": ["VF001"], "_invalid_evidence_ids": [],
        }
        accepted, reason, _ = review_revision(
            self.current, candidate, ledger, visual_review=visual_review,
            claim_contract={"safe_for_directional_reasoning": True},
            mediation=mediation,
        )
        self.assertTrue(accepted)
        self.assertTrue(reason.startswith("accepted_mediated_verified_tiebreak:"))

        rejected, reason, _ = review_revision(
            self.current, candidate, ledger, visual_review=visual_review,
            claim_contract={"safe_for_directional_reasoning": True},
            mediation=None,
        )
        self.assertFalse(rejected)
        self.assertEqual(reason, "unresolved_opposing_decision_grade_evidence")


class HubSourceTests(unittest.TestCase):
    def test_cached_snapshot_uses_local_path_without_revision_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = (
                Path(temporary) / "models--org--model" / "snapshots" / "abc"
            )
            snapshot.mkdir(parents=True)
            for name in (
                "config.json", "model.safetensors.index.json",
                "tokenizer_config.json",
            ):
                (snapshot / name).write_text("{}", encoding="utf-8")
            with patch("models.hub_source.hub_cache_root", return_value=temporary):
                source, kwargs = cached_snapshot_or_hub("org/model", "abc")
            self.assertEqual(source, str(snapshot))
            self.assertEqual(kwargs, {"local_files_only": True})

    def test_missing_snapshot_preserves_online_first_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("models.hub_source.hub_cache_root", return_value=temporary):
                source, kwargs = cached_snapshot_or_hub("org/model", "abc")
            self.assertEqual(source, "org/model")
            self.assertEqual(kwargs, {"revision": "abc"})


class JudgeEvaluationTests(unittest.TestCase):
    def test_mediated_verdicts_replace_empty_judge_column(self):
        rows = pd.DataFrame({
            "ground_truth": ["ENTAILS", "CONTRADICTS"],
            "prediction": ["ENTAILS", "CONTRADICTS"],
            "phenomenon": ["literal", "metaphor"],
            "judge_requested": [True, True],
            "judge_mode": ["mediated", "mediated"],
            "judge_verdict": ["", ""],
            "mediator_format_valid": [True, True],
            "mediator_provisional_verdict": ["ABSTAIN", "CONTRADICTS"],
            "mediator_usable": [True, True],
        })
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "predictions.csv"
            output_path = Path(temporary) / "metrics"
            rows.to_csv(input_path, index=False)
            metrics = evaluate_predictions(input_path, output_path)
        self.assertEqual(metrics["judge"]["contract_valid_rate"], 1.0)
        self.assertEqual(
            metrics["judge"]["verdict_distribution"],
            {"ABSTAIN": 1, "CONTRADICTS": 1},
        )

    def test_tribunal_metrics_record_rounds_and_verified_resolution(self):
        rows = pd.DataFrame({
            "ground_truth": ["CONTRADICTS"],
            "prediction": ["CONTRADICTS"],
            "phenomenon": ["metaphor"],
            "judge_requested": [True],
            "judge_mode": ["tribunal"],
            "judge_verdict": ["CONTRADICTS"],
            "judge_format_valid": [True],
            "judge_revision_accepted": [True],
            "pre_judge_prediction": ["ENTAILS"],
            "tribunal_state": ["RESOLVED"],
            "tribunal_round_count": [2],
            "tribunal_review_status": ["RESOLVE"],
            "tribunal_verified_evidence_id": ["AV001"],
        })
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "predictions.csv"
            output_path = Path(temporary) / "metrics"
            rows.to_csv(input_path, index=False)
            metrics = evaluate_predictions(input_path, output_path)
        self.assertEqual(metrics["judge"]["tribunal_requested_count"], 1)
        self.assertEqual(metrics["judge"]["tribunal_mean_rounds"], 2.0)
        self.assertEqual(
            metrics["judge"]["tribunal_state_distribution"],
            {"RESOLVED": 1},
        )
        self.assertEqual(
            metrics["judge"]["tribunal_verified_relation_count"], 1
        )

    def test_tribunal_harm_is_not_counted_as_debate_harm(self):
        rows = pd.DataFrame({
            "ground_truth": ["ENTAILS"],
            "prediction": ["CONTRADICTS"],
            "final_prediction": ["CONTRADICTS"],
            "phenomenon": ["metaphor"],
            "debate_triggered": [True],
            "initial_prediction": ["ENTAILS"],
            "pre_judge_prediction": ["ENTAILS"],
            "judge_requested": [True],
            "judge_mode": ["tribunal"],
            "judge_verdict": ["CONTRADICTS"],
            "judge_format_valid": [True],
            "judge_revision_accepted": [True],
        })
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "predictions.csv"
            output_path = Path(temporary) / "metrics"
            rows.to_csv(input_path, index=False)
            metrics = evaluate_predictions(input_path, output_path)
        self.assertEqual(metrics["debate"]["harm_count"], 0)
        self.assertEqual(metrics["judge"]["accepted_harms"], 1)


if __name__ == "__main__":
    unittest.main()
