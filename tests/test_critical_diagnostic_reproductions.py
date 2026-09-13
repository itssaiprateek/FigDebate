"""Known defects expressed as expected failures, not claimed repaired behavior."""
import json
import unittest
from unittest.mock import patch
from agents.claim_extraction import ClaimExtractionAgent
from engine.claim_semantics import CHECKS
from engine.claim_witness import audit_claim_witness
from engine.debate import DebateEngine
from engine.output_contracts import object_schema, saturated_text_fields


class CriticalDiagnosticReproductions(unittest.TestCase):
    def test_placeholder_requirements_must_fail_even_if_model_approves(self):
        agent = object.__new__(ClaimExtractionAgent)
        witness = dict(stance="ENDORSE", support_requirement="None", conflict_requirement="None",
                       figurative_mechanism="unknown", ambiguity="none", reason="fixture")
        audit = dict(**{key: True for key in CHECKS}, _format_valid=True)
        with patch.object(agent, "_generate_section", return_value=(witness, json.dumps(witness), 0, {"schema_valid": True})), \
             patch("engine.claim_witness.audit_core", return_value=audit):
            result = audit_claim_witness(agent, "The line rises.", "Claim subject: line")
        self.assertFalse(result["requirements_valid"])

    def test_complete_unpunctuated_clause_is_not_truncation(self):
        schema = object_schema({"reason": {"type": "string", "maxLength": 20}})
        self.assertEqual(saturated_text_fields('{"reason":"The sky appears blue"}', schema), [])

    def test_short_dangling_clause_is_not_complete(self):
        schema = object_schema({"reason": {"type": "string", "maxLength": 20}})
        self.assertEqual(saturated_text_fields('{"reason":"because the"}', schema), ["reason"])

    def test_empty_list_survives_hearing_prompt_roundtrip(self):
        import ast
        prior = DebateEngine.build_agent2_challenge_prompt(None, {"negation": []}, {})
        parsed = ast.literal_eval(ClaimExtractionAgent._audit_field(prior, "negation"))
        self.assertEqual(parsed, [])


if __name__ == "__main__":
    unittest.main()
