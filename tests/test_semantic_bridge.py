from tests.verification_fixture import with_independent_fixture
import json
import unittest

from agents.multimodal_judge import (
    _run_structured_generation,
    build_tribunal_repair_prompt,
)

from engine.evidence_ledger import add_semantic_bridge_evidence
from engine.semantic_bridge import build_semantic_bridge
from engine.semantic_bridge_verifier import verify_semantic_bridge
from utils.judge_parser import parse_tribunal_review_response
from engine.tribunal import apply_tribunal_resolution, repair_followup_plan


def ledger():
    return [
        {
            "id": "VF001", "source": "agent1", "type": "visual_fact",
            "text": "The plotted line visibly falls from left to right.",
            "relation": "NEUTRAL", "grounded": True, "decision_grade": False,
            "lifecycle_status": "ACTIVE", "derived_from_ids": [],
        },
        {
            "id": "LC001", "source": "agent2", "type": "caption_proposition",
            "text": "The plotted line rises.", "relation": "NEUTRAL",
            "grounded": False, "decision_grade": False,
            "lifecycle_status": "ACTIVE", "derived_from_ids": [],
        },
    ]


def contract():
    return {
        "schema_version": "3.0", "caption_proposition": "The plotted line rises.",
        "source_caption": "The plotted line rises.", "proposition_preserved": True,
        "entity_frame_preserved": True, "entity_checks": {"claim_subject": True},
        "safe_for_tribunal_reasoning": True, "structural_reasoning_type": "COMPARATIVE_LAYOUT",
        "time_or_panel_scope": "", "requires_normative_reasoning": False,
    }


class SemanticBridgeTests(unittest.TestCase):
    def _review(self):
        review = {
            "relation": "CONFLICT", "admissibility": "CORROBORATED",
            "visual_premise": "The plotted line visibly falls from left to right.",
            "caption_premise": "The plotted line rises.",
            "semantic_bridge_type": "COMPARISON_DIRECTION",
            "semantic_bridge": "A falling line is opposite to a rising line.",
            "counter_interpretation": "", "counter_interpretation_strength": 0.1,
            "confidence": 0.9, "_valid_evidence_ids": ["VF001"],
        }
        return with_independent_fixture(review, ledger(), contract())

    def test_bridge_requires_existing_visual_premise(self):
        review = self._review()
        review["_valid_evidence_ids"] = []
        proposal = build_semantic_bridge(review, ledger(), contract())
        self.assertEqual(proposal["verification_status"], "INSUFFICIENT")

    def test_corroborated_bridge_is_auditable_and_decision_grade(self):
        proposal = build_semantic_bridge(self._review(), ledger(), contract())
        proposal, verification = verify_semantic_bridge(proposal, ledger(), contract(), True)
        self.assertTrue(verification["corroborated"])
        updated, record = add_semantic_bridge_evidence(ledger(), proposal, verification)
        self.assertTrue(record["promoted"])
        self.assertTrue(updated[-1]["decision_grade"])
        self.assertEqual(updated[-1]["evidence_level"], "BRIDGE_CORROBORATED")

    def test_missing_evidence_cannot_form_conflict(self):
        review = self._review()
        review["visual_premise"] = "No evidence is visible."
        proposal = build_semantic_bridge(review, ledger(), contract())
        _, verification = verify_semantic_bridge(proposal, ledger(), contract(), True)
        self.assertFalse(verification["corroborated"])

    def test_depiction_language_cannot_be_verified_as_conflict(self):
        review = self._review()
        review["semantic_bridge_type"] = "SYMBOL_TARGET_ATTACHMENT"
        review["semantic_bridge"] = (
            "The visual metaphor maps the distorted figure to the caption's "
            "description."
        )
        proposal = build_semantic_bridge(review, ledger(), contract())
        _, verification = verify_semantic_bridge(
            proposal, ledger(), contract(), True
        )
        self.assertFalse(verification["corroborated"])
        self.assertIn(
            "bridge_statement_relation_consistent",
            verification["failed_checks"],
        )

    def test_humor_incongruity_is_not_directional_evidence(self):
        review = self._review()
        review["semantic_bridge_type"] = "HUMOR_INCONGRUITY"
        review["semantic_bridge"] = (
            "The unusual pairing creates a humorous incongruity."
        )
        humor_contract = contract() | {
            "structural_reasoning_type": "BACKGROUND_REQUIRED"
        }
        proposal = build_semantic_bridge(review, ledger(), humor_contract)
        _, verification = verify_semantic_bridge(
            proposal, ledger(), humor_contract, True
        )
        self.assertFalse(verification["corroborated"])
        self.assertIn("directional_bridge_family", verification["failed_checks"])

    def test_v2_parser_separates_best_judgment_and_admissibility(self):
        payload = {
            "best_semantic_judgment": "CONTRADICTS", "relation": "CONFLICT",
            "admissibility": "PLAUSIBLE", "visual_premise": "The line falls.",
            "caption_premise": "The line rises.",
            "semantic_bridge_type": "COMPARISON_DIRECTION",
            "semantic_bridge": "Falling opposes rising.", "evidence_ids": ["VF001"],
            "counter_interpretation": "Perspective may distort slope.",
            "counter_interpretation_strength": 0.4, "confidence": 0.7,
            "requested_follow_up": "VISUAL_PREMISE", "reason": "Slope needs confirmation.",
        }
        parsed = parse_tribunal_review_response(json.dumps(payload))
        self.assertTrue(parsed["_format_valid"])
        self.assertEqual(parsed["best_semantic_judgment"], "CONTRADICTS")
        self.assertEqual(parsed["status"], "FOLLOW_UP")

    def test_v2_parser_preserves_symmetric_cases(self):
        payload = {
            "best_semantic_judgment": "CONTRADICTS", "relation": "CONFLICT",
            "admissibility": "CORROBORATED", "visual_premise": "The line falls.",
            "caption_premise": "The line rises.",
            "semantic_bridge_type": "COMPARISON_DIRECTION",
            "semantic_bridge": "Falling is opposite to rising.",
            "evidence_ids": ["VF001"],
            "support_case": "No observed rise.", "support_evidence_ids": [],
            "support_strength": 0.1,
            "conflict_case": "The observed line falls.",
            "conflict_evidence_ids": ["VF001"], "conflict_strength": 0.9,
            "counter_interpretation": "Perspective may distort slope.",
            "counter_interpretation_strength": 0.1, "confidence": 0.9,
            "requested_follow_up": "NONE", "reason": "Direction is opposite.",
        }
        parsed = parse_tribunal_review_response(json.dumps(payload))
        self.assertTrue(parsed["_format_valid"])
        self.assertTrue(parsed["symmetric_cases_present"])
        self.assertEqual(parsed["conflict_strength"], 0.9)

    def test_compact_alias_contract_is_rejected(self):
        payload = {
            "j": "CONTRADICTS", "r": "CONFLICT", "a": "CORROBORATED",
            "vp": "The plotted line falls.", "cp": "The line rises.",
            "bt": "COMPARISON_DIRECTION", "b": "Falling opposes rising.",
            "ids": ["VF001"], "sc": "No observed rise.", "si": [],
            "ss": 0.1, "cc": "The line visibly falls.", "ci": ["VF001"],
            "cs": 0.9, "alt": "Perspective may distort slope.", "as": 0.1,
            "c": 0.9, "fu": "NONE", "why": "Direction is opposite.",
        }
        parsed = parse_tribunal_review_response(json.dumps(payload))
        self.assertFalse(parsed["_format_valid"])

    def test_truncated_contract_uses_short_targeted_rewrite(self):
        valid = json.dumps({
            "best_semantic_judgment": "CONTRADICTS",
            "relation": "CONFLICT", "admissibility": "CORROBORATED",
            "visual_premise": "The plotted line falls.",
            "caption_premise": "The line rises.",
            "semantic_bridge_type": "COMPARISON_DIRECTION",
            "semantic_bridge": "Falling opposes rising.",
            "evidence_ids": ["VF001"], "counter_interpretation": "",
            "counter_interpretation_strength": 0.0, "confidence": 0.9,
            "requested_follow_up": "NONE", "reason": "Direction is opposite.",
        })

        class Runtime:
            hardware_profile = None

            def __init__(self):
                self.prompts = []
                self.outputs = [
                    '{"best_semantic_judgment":"CONTRADICTS","relation":"CONFLICT"',
                    valid,
                ]
                self._last_generation_diagnostics = {}

            def generate(self, image, prompt, max_new_tokens=None):
                self.prompts.append(prompt)
                output = self.outputs.pop(0)
                self._last_generation_diagnostics = {
                    "generated_tokens": max_new_tokens if len(self.prompts) == 1 else 180,
                    "max_new_tokens": max_new_tokens,
                    "hit_token_limit": len(self.prompts) == 1,
                }
                return output, 0.1

        runtime = Runtime()
        parsed = _run_structured_generation(
            runtime, object(), "ORIGINAL CASE PACKET " * 300,
            parse_tribunal_review_response, max_new_tokens=384,
            contract_name="tribunal_review",
            repair_prompt_builder=build_tribunal_repair_prompt,
        )
        self.assertTrue(parsed["_format_valid"])
        self.assertTrue(parsed["_format_retry_success"])
        self.assertEqual(parsed["_format_retry_strategy"], "targeted_contract_rewrite")
        self.assertIn("ORIGINAL CASE PACKET", runtime.prompts[1])
        self.assertIn("identified defect", runtime.prompts[1])

    def test_judgment_relation_mismatch_is_repaired_not_normalized(self):
        mismatch = json.dumps({
            "best_semantic_judgment": "CONTRADICTS", "relation": "SUPPORT",
            "admissibility": "CORROBORATED", "visual_premise": "The line rises.",
            "caption_premise": "The line rises.",
            "semantic_bridge_type": "COMPARISON_DIRECTION",
            "semantic_bridge": "Both directions match.",
            "evidence_ids": ["VF001"], "counter_interpretation": "",
            "counter_interpretation_strength": 0.0, "confidence": 0.9,
            "requested_follow_up": "NONE", "reason": "The visual direction agrees.",
        })
        repaired = json.loads(mismatch)
        repaired["best_semantic_judgment"] = "ENTAILS"

        class Runtime:
            def __init__(self):
                self.outputs = [mismatch, json.dumps(repaired)]
                self.prompts = []
                self._last_generation_diagnostics = {}

            def generate(self, image, prompt, max_new_tokens=None):
                self.prompts.append(prompt)
                return self.outputs.pop(0), 0.1

        runtime = Runtime()
        parsed = _run_structured_generation(
            runtime, object(), "full prompt", parse_tribunal_review_response,
            max_new_tokens=384, contract_name="tribunal_review",
            repair_prompt_builder=build_tribunal_repair_prompt,
        )
        self.assertTrue(parsed["_format_valid"])
        self.assertEqual(parsed["provisional_verdict"], "ENTAILS")
        self.assertIn("judgment_relation_mismatch", runtime.prompts[1])

    def test_repairable_contract_routes_to_followup(self):
        plan = repair_followup_plan(
            {
                "semantic_bridge_type": "HUMOR_INCONGRUITY",
                "counter_interpretation_strength": 0.8,
                "admissibility": "PLAUSIBLE",
                "_valid_evidence_ids": ["VF001"],
            },
            {"claim_contract": contract()},
            {"agent2_critique": {"requirements_valid": False}},
        )
        self.assertTrue(plan["_usable"])
        self.assertIn("AGENT2_DIRECTIONAL_CONTRACT", plan["repair_reasons"])

    def test_corroborated_mode_can_revise_only_through_review_board(self):
        review = self._review() | {
            "status": "RESOLVE", "provisional_verdict": "CONTRADICTS",
            "best_semantic_judgment": "CONTRADICTS", "_format_valid": True,
            "_invalid_evidence_ids": [], "visual_observations": [
                "The plotted line visibly falls from left to right."
            ], "reason": "The observed direction is opposite.",
        }
        decision, updated, metadata = apply_tribunal_resolution(
            {"label": "ENTAILS", "confidence": 0.4}, review, ledger(),
            contract(), agent2_requirements_valid=True,
            semantic_bridge_mode="corroborated",
        )
        self.assertTrue(metadata["accepted"])
        self.assertEqual(decision["label"], "CONTRADICTS")
        self.assertTrue(any(item.get("evidence_level") == "BRIDGE_CORROBORATED" for item in updated))

    def test_shadow_mode_records_without_using_bridge_to_change_label(self):
        review = self._review() | {
            "status": "ABSTAIN", "provisional_verdict": "CONTRADICTS",
            "best_semantic_judgment": "CONTRADICTS", "_format_valid": True,
            "_invalid_evidence_ids": [], "visual_observations": [
                "The plotted line visibly falls from left to right."
            ], "reason": "The observed direction is opposite.",
        }
        decision, updated, metadata = apply_tribunal_resolution(
            {"label": "ENTAILS", "confidence": 0.4}, review, ledger(),
            contract(), agent2_requirements_valid=True,
            semantic_bridge_mode="shadow",
        )
        self.assertFalse(metadata["accepted"])
        self.assertEqual(decision["label"], "ENTAILS")
        self.assertEqual(updated, ledger())
        self.assertTrue(any(item.get("type") == "semantic_bridge"
                            for item in metadata["shadow_evidence"]))

    def test_humor_conflict_is_rejected_without_directional_family(self):
        review = self._review() | {
            "status": "RESOLVE", "provisional_verdict": "CONTRADICTS",
            "best_semantic_judgment": "CONTRADICTS", "_format_valid": True,
            "semantic_bridge_type": "HUMOR_INCONGRUITY",
            "semantic_bridge": "The pairing creates a humorous incongruity.",
            "_invalid_evidence_ids": [], "visual_observations": [
                "The plotted line visibly falls from left to right."
            ], "reason": "The pairing is incongruous.",
        }
        humor_contract = contract() | {
            "structural_reasoning_type": "BACKGROUND_REQUIRED"
        }
        decision, _, metadata = apply_tribunal_resolution(
            {"label": "ENTAILS", "confidence": 0.4}, review, ledger(),
            humor_contract, agent2_requirements_valid=True,
            semantic_bridge_mode="corroborated",
        )
        self.assertFalse(metadata["accepted"])
        self.assertEqual(decision["label"], "ENTAILS")
        self.assertEqual(
            metadata["reason"],
            "figurative_incongruity_requires_directional_relation_proof",
        )


if __name__ == "__main__":
    unittest.main()
