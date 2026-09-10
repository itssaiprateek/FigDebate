import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch
from agents.claim_extraction import ClaimExtractionAgent
from engine.stage_checkpoint import StageCheckpointStore
from engine.deliberation import evidence_signature
from engine.final_artifact import final_artifact
from engine.review_outcome import failed_review, classify_review
from engine.evidence_ledger import is_admissible_evidence
from evaluation.grouped_metrics import grouped_accuracy_difference, verify_pair_identity


class IntegratedProtocolTests(unittest.TestCase):
    def test_changed_caption_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StageCheckpointStore(directory, enabled=True)
            sample = {"raw": {"id": "case"}, "caption": "The line rises."}
            store.save("visual", sample, {})
            sample["caption"] = "The line falls."
            with self.assertRaisesRegex(RuntimeError, "input mismatch"):
                store.load("visual", sample)

    def test_new_id_is_not_new_evidence(self):
        row = {"id": "A", "text": "A line rises.", "relation": "SUPPORT"}
        self.assertEqual(evidence_signature([row]), evidence_signature([dict(row, id="B")]))

    def test_duplicate_ledger_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            is_admissible_evidence([{"id": "A"}, {"id": "A"}], "A")

    def test_failed_review_is_not_semantic_abstention(self):
        outcome = classify_review(failed_review(RuntimeError("model missing"), "load", 1))
        self.assertFalse(outcome["semantic_abstained"])
        self.assertEqual(outcome["execution_status"], "FAILED")

    def test_explanation_does_not_adopt_rejected_judge_reason(self):
        result = {"decision": {"label": "ENTAILS", "explanation": "Baseline reason"},
                  "judge": {"requested": True, "tribunal_resolution": {
                      "accepted": False, "reason": "invalid_contract"},
                      "tribunal_reviews": [{"reason": "Rejected reason", "provisional_verdict": "CONTRADICTS"}]}}
        artifact = final_artifact("Caption", result)
        self.assertEqual("Baseline reason", artifact["accepted_reason"])
        self.assertIn("baseline_retained_after_review", artifact["delivered_explanation"])
        self.assertNotIn("Rejected reason", artifact["delivered_explanation"])
        self.assertFalse(artifact["evidence_chain_complete"])

    def test_grouped_inference_is_deterministic_and_clusters_pairs(self):
        rows = [{"image_group_id": "a", "control_correct": False, "treatment_correct": True},
                {"image_group_id": "a", "control_correct": True, "treatment_correct": False},
                {"image_group_id": "b", "control_correct": True, "treatment_correct": True}]
        first = grouped_accuracy_difference(rows, iterations=100)
        self.assertEqual(first, grouped_accuracy_difference(rows, iterations=100))
        self.assertEqual(first["groups"], 2)
        self.assertEqual(first["accuracy_delta"], 0)

    def test_comparison_rejects_changed_input(self):
        row = {"image_sha256": "x", "caption_sha256": "y", "image_group_id": "z"}
        with self.assertRaisesRegex(ValueError, "identity differs"):
            verify_pair_identity({"a": row}, {"a": dict(row, caption_sha256="new")})

    def test_interpretation_cannot_overwrite_claim_core(self):
        agent = object.__new__(ClaimExtractionAgent)
        core = {"caption_proposition": "The line rises.", "claim_subject": "line",
                "claim_predicate": "rises", "claim_object": "None",
                "claim_source": "None", "claim_target": "None",
                "asserted_property": "rises", "expected_visual_state": "The line rises.",
                "opposite_visual_state": "The line falls.", "relation_family": "trajectory"}
        interpretation = {"figurative_type": "literal", "claim_subject": "malicious overwrite",
                          "underlying_message": "A fall", "literal_polarity": "positive",
                          "intended_polarity": "positive", "structural_reasoning_type": "direct_state"}
        diag = {"hit_token_limit": False, "generated_tokens": 30}
        from engine.claim_semantics import CHECKS
        audit = {**{key: True for key in CHECKS}, "defective_fields": [], "reason": "fixture"}
        with patch("engine.claim_semantics.audit_core", return_value=dict(audit, _generation_seconds=0,
                _generation_diagnostics=[], independent_source_answers=[])), patch.object(agent, "_generate_section", side_effect=[
                (core, "core", .1, diag), (interpretation, "interpretation", .1, diag),
                (audit, "audit", .1, dict(diag, schema_valid=True))]):
            output = agent.analyze("The line rises.")
        self.assertEqual(output["claim_subject"], "line")
        self.assertEqual(output["caption_proposition"], "The line rises.")
        self.assertEqual(output["intended_meaning"], "A fall")
