"""Dataset-independent regressions for the research audit's counterexamples."""
import copy
import json
import unittest

from agents.multimodal_judge import _run_structured_generation
from engine.case_dossier import build_case_dossier, render_judge_dossier
from engine.claim_contract import audit_claim_contract
from engine.evidence_ledger import audit_decision
from engine.review_board import decision_grade_strength
from engine.tribunal import apply_tribunal_resolution
from utils.judge_parser import parse_tribunal_review_response


class IntegrityRepairTests(unittest.TestCase):
    def test_execution_failure_is_not_semantic_abstention(self):
        class Runtime:
            _last_generation_diagnostics = {"failed_attempts": [{"error": "OOM"}]}

            def generate(self, *args, **kwargs):
                raise RuntimeError("synthetic out of memory")

        review = _run_structured_generation(
            Runtime(), None, "prompt", parse_tribunal_review_response,
            max_new_tokens=100, contract_name="tribunal_review",
        )
        _, _, meta = apply_tribunal_resolution(
            {"label": "ENTAILS"}, review, [], semantic_bridge_mode="disabled",
        )
        self.assertFalse(meta["semantic_abstained"])
        self.assertEqual(review.get("_execution_status"), "FAILED")
        self.assertEqual(review.get("_execution_error_type"), "OUT_OF_MEMORY")
        self.assertTrue(review["_generation_diagnostics"])
        self.assertGreater(review["_generation_seconds"], 0)

    def test_malformed_output_is_not_semantic_abstention(self):
        review = parse_tribunal_review_response("not json")
        _, _, meta = apply_tribunal_resolution({"label": "ENTAILS"}, review, [])
        self.assertFalse(meta["semantic_abstained"])
        self.assertFalse(meta["semantic_judgment_valid"])

    def test_dossier_does_not_copy_prior_decision_or_free_form_route(self):
        dossier = build_case_dossier(
            "A line rises.", {"visual_facts": ["A line falls."]}, {},
            {"recommendation": "SECRET_PRIOR", "claim_direction": "SECRET_PRIOR"},
            {"label": "SECRET_PRIOR"}, [],
            {"agent1_critique": {"recommendation": "SECRET_PRIOR"}},
            {"current_label": "SECRET_PRIOR", "reasons": ["SECRET_PRIOR"]},
        )
        packet = render_judge_dossier(dossier)
        self.assertNotIn("SECRET_PRIOR", json.dumps(packet))

    def test_superseded_parent_invalidates_child_and_strength(self):
        ledger = [
            {"id": "OLD", "source": "agent1", "text": "Old observation",
             "grounded": True, "decision_grade": True, "relation": "SUPPORT",
             "reliability": 1.0, "lifecycle_status": "SUPERSEDED"},
            {"id": "CHILD", "source": "targeted_region_verifier", "text": "Derived",
             "grounded": True, "decision_grade": True, "relation": "SUPPORT",
             "reliability": 0.9, "lifecycle_status": "ACTIVE",
             "derived_from_ids": ["OLD"]},
        ]
        self.assertEqual(decision_grade_strength(ledger, "SUPPORT"), 0)
        audit = audit_decision({"label": "ENTAILS", "_model_cited_evidence_ids": ["CHILD"]}, ledger)
        self.assertFalse(audit["valid"])

    def test_missing_or_cyclic_parent_is_not_admissible(self):
        from engine.evidence_ledger import is_admissible_evidence
        orphan = {"id": "A", "derived_from_ids": ["MISSING"]}
        cycle = [{"id": "A", "derived_from_ids": ["B"]},
                 {"id": "B", "derived_from_ids": ["A"]}]
        self.assertFalse(is_admissible_evidence([orphan], orphan))
        self.assertFalse(is_admissible_evidence(cycle, cycle[0]))

    def test_shadow_never_changes_the_operational_ledger(self):
        ledger = [
            {"id": "VF001", "source": "agent1", "type": "visual_fact",
             "text": "The line falls.", "grounded": True, "relation": "NEUTRAL"},
            {"id": "LC001", "source": "agent2", "type": "caption_proposition",
             "text": "The line rises.", "grounded": False, "relation": "NEUTRAL"},
        ]
        review = {"_format_valid": True, "status": "RESOLVE", "relation": "CONFLICT",
                  "provisional_verdict": "CONTRADICTS", "confidence": 0.9,
                  "visual_premise": "The line falls.", "caption_premise": "The line rises.",
                  "semantic_bridge": "Falling is opposite to rising.",
                  "_valid_evidence_ids": ["VF001"], "visual_observations": ["The line falls."]}
        original = copy.deepcopy(ledger)
        _, shadow, _ = apply_tribunal_resolution(
            {"label": "ENTAILS"}, review, ledger, {}, semantic_bridge_mode="shadow",
        )
        _, disabled, _ = apply_tribunal_resolution(
            {"label": "ENTAILS"}, review, ledger, {}, semantic_bridge_mode="disabled",
        )
        self.assertEqual(shadow, disabled)
        self.assertEqual(ledger, original)

    def test_swapped_roles_cannot_pass_directional_contract(self):
        contract = audit_claim_contract("The dog bites the man.", {
            "caption_proposition": "The dog bites the man.",
            "claim_subject": "the man", "claim_object": "the dog", "claim_predicate": "bites",
            "expected_visual_state": "The man bites the dog.",
            "opposite_visual_state": "The man does not bite the dog.",
        })
        self.assertFalse(contract["safe_for_directional_reasoning"])


if __name__ == "__main__":
    unittest.main()
