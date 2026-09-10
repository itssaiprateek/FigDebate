import unittest

from agents.claim_extraction import ClaimExtractionAgent

from engine.claim_contract import audit_claim_contract


class TypedClaimGroupTests(unittest.TestCase):
    def _base(self):
        return {
            "caption_proposition": "The line rises",
            "claim_subject": "line",
            "claim_object": "None",
            "claim_source": "None",
            "claim_target": "line",
            "claim_predicate": "rises",
            "expected_visual_state": "the line rises",
            "opposite_visual_state": "the line falls",
            "structural_reasoning_type": "TRAJECTORY",
            "literal_polarity": "positive",
            "intended_polarity": "positive",
        }

    def test_groups_are_audited_independently(self):
        contract = audit_claim_contract("The line rises", self._base())
        self.assertTrue(contract["field_groups"]["immutable_proposition"]["valid"])
        self.assertTrue(contract["field_groups"]["relation"]["valid"])
        self.assertTrue(contract["field_groups"]["reasoning_profile"]["valid"])

    def test_absence_only_conflict_invalidates_relation_group(self):
        value = self._base()
        value["opposite_visual_state"] = "the line is not visible"
        contract = audit_claim_contract("The line rises", value)
        self.assertFalse(contract["field_groups"]["relation"]["valid"])
        self.assertIn("relation", contract["invalid_field_groups"])



if __name__ == "__main__":
    unittest.main()
