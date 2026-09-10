import json
import unittest

from agents.multimodal_judge import TribunalMediatorAgent
from engine.case_dossier import build_case_dossier, render_judge_dossier
from engine.claim_contract import assess_tribunal_claim_dependencies
from engine.evidence_ledger import classify_evidence
from engine.tribunal import MAX_TRIBUNAL_ROUNDS
from utils.judge_parser import parse_tribunal_review_response


class SupervisorDossierTests(unittest.TestCase):
    def _ledger(self):
        return [
            {
                "id": "VF001", "source": "agent1", "type": "visual_fact",
                "text": "The plotted line falls.", "relation": "NEUTRAL",
                "grounded": True, "decision_grade": False,
                "lifecycle_status": "ACTIVE", "derived_from_ids": [],
                "evidence_level": "OBSERVATION",
            },
            {
                "id": "TV001", "source": "targeted_region_verifier",
                "type": "verified_region_relation",
                "text": "The fall conflicts with the claimed rise.",
                "relation": "CONFLICT", "grounded": True,
                "decision_grade": True, "lifecycle_status": "ACTIVE",
                "derived_from_ids": ["VF001"],
                "evidence_level": "VERIFIED_RELATION",
            },
        ]

    def test_dossier_hides_initial_label_and_preserves_every_active_id(self):
        dossier = build_case_dossier(
            "The line rises.", {"visual_facts": ["The plotted line falls."]},
            {"caption_proposition": "The line rises."}, {},
            {"label": "ENTAILS", "confidence": 0.4}, self._ledger(),
            {"review_status": "hearing_recorded"},
            {"issue_type": "COMPARISON_OR_OUTCOME", "reasons": ["uncertain"]},
        )
        rendered = render_judge_dossier(dossier, detailed_evidence_limit=1)
        self.assertTrue(rendered["initial_decision_audit"]["label_hidden"])
        self.assertNotIn("label", rendered["initial_decision_audit"])
        visible_ids = {
            item["id"] for item in rendered["evidence_ledger"]
        } | {
            item["id"] for item in rendered["remaining_evidence_index"]
        }
        self.assertEqual(visible_ids, {"VF001", "TV001"})
        self.assertTrue(rendered["evidence_rendering"]["all_active_ids_visible"])

    def test_dossier_removes_raw_generation_duplication(self):
        dossier = build_case_dossier(
            "Caption", {"visual_facts": ["Fact"], "_raw_response": "raw"},
            {"caption_proposition": "Caption", "_raw_output": "raw"}, {},
            {}, [],
        )
        self.assertNotIn("_raw_response", dossier["visual_agent"])
        self.assertNotIn("_raw_output", dossier["claim_agent"])


class TolerantContractTests(unittest.TestCase):
    def test_optional_contract_defects_do_not_erase_semantic_verdict(self):
        parsed = parse_tribunal_review_response(json.dumps({
            "best_semantic_judgment": "CONTRADICTS",
            "relation": "CONFLICT",
            "confidence": "0.88",
            "evidence_ids": "VF-001",
            "reason": "The visible line falls.",
            "semantic_bridge_type": "ENTITY_BINDING",
        }))
        self.assertTrue(parsed["_format_valid"])
        self.assertEqual(parsed["provisional_verdict"], "CONTRADICTS")
        self.assertEqual(parsed["confidence"], 0.88)
        self.assertIn("unknown_bridge_type_ignored", parsed["_normalized_fields"])

    def test_unknown_evidence_is_not_normalized_to_a_real_id(self):
        class Runtime:
            hardware_profile = None
            _last_generation_diagnostics = {}

            @staticmethod
            def generate(_image, _prompt, max_new_tokens=None):
                return json.dumps({
                    "best_semantic_judgment": "CONTRADICTS",
                    "relation": "CONFLICT", "confidence": 0.9,
                    "evidence_ids": ["MADE-UP-999"],
                    "reason": "Unsupported evidence.",
                }), 0.01

        review = TribunalMediatorAgent(Runtime()).review(
            object(), "The line rises.", {}, {}, {},
            [{
                "id": "VF001", "source": "agent1", "type": "visual_fact",
                "text": "The line falls.", "grounded": True,
                "relation": "NEUTRAL", "lifecycle_status": "ACTIVE",
            }], {},
        )
        self.assertEqual(review["_valid_evidence_ids"], [])
        self.assertEqual(review["_invalid_evidence_ids"], ["MADE-UP-999"])


class DependencyAwareGateTests(unittest.TestCase):
    def test_optional_reasoning_profile_does_not_block_verified_direction(self):
        audit = assess_tribunal_claim_dependencies(
            {
                "proposition_preserved": True,
                "entity_frame_preserved": True,
                "field_groups": {
                    "immutable_proposition": {"valid": True},
                    "relation": {"valid": False},
                    "reasoning_profile": {"valid": False},
                },
                "warnings": [],
            },
            cited_evidence=[{
                "relation": "CONFLICT", "grounded": True,
                "decision_grade": True,
            }],
            proposed_relation="CONFLICT",
        )
        self.assertTrue(audit["safe"])
        self.assertEqual(audit["ignored_optional_groups"], ["reasoning_profile"])

    def test_changed_negation_remains_fatal(self):
        audit = assess_tribunal_claim_dependencies(
            {
                "field_groups": {
                    "immutable_proposition": {"valid": True},
                    "relation": {"valid": True},
                },
                "warnings": ["caption_negation_changed_or_dropped"],
            },
            agent2_requirements_valid=True,
            proposed_relation="CONFLICT",
        )
        self.assertFalse(audit["safe"])

    def test_evidence_classes_are_general_and_two_round_cap_is_enforced(self):
        self.assertEqual(
            classify_evidence("agent1", "visible_text"),
            "DIRECT_TEXT_OR_COMPARISON",
        )
        self.assertEqual(
            classify_evidence("agent1", "symbolic_anchor"),
            "FIGURATIVE_OR_SYMBOLIC_MAPPING",
        )
        self.assertEqual(MAX_TRIBUNAL_ROUNDS, 2)


if __name__ == "__main__":
    unittest.main()
