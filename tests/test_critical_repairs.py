import json
import unittest
from engine.claim_semantics import audit_core, CHECKS
from engine.claim_validation import requirement_errors
from engine.output_contracts import schema_for_contract
from utils.judge_parser import parse_tribunal_review_response
from evaluation.qualify_decoder import example
from tests.critical_fixture import core


class FactoredAuditTests(unittest.TestCase):
    def test_source_answers_are_blind_and_failure_has_repair_fields(self):
        prompts = []
        class Agent:
            def _generate_section(self, instruction, caption, budget, schema):
                prompts.append(instruction)
                if "source_ids" in schema["properties"]:
                    value = dict(answer="The meeting is calm.", source_ids=["C1"], unknown=False)
                elif "relation" in schema["properties"]:
                    value = dict(relation="SUPPORT", reason="The condition matches the claim.")
                else:
                    value = dict(status="FAIL" if len(prompts) == 5 else "PASS", reason="The obligation fails." if len(prompts) == 5 else "The obligation holds.")
                return value, json.dumps(value), .1, {"schema_valid": True}
        fields = core()
        fields["opposite_visual_state"] = "PROPOSAL_ONLY_MARKER"
        result = audit_core(Agent(), fields["caption_proposition"], fields)
        self.assertEqual(len(prompts), 5)
        self.assertTrue(all("PROPOSAL_ONLY_MARKER" not in prompt for prompt in prompts[:2]))
        self.assertTrue(all("opposite_visual_state" not in prompt for prompt in prompts[3:]))
        self.assertEqual(result["defective_fields"], ["opposite_visual_state"])
        self.assertFalse(result[CHECKS[-1]])
        self.assertTrue(result["_format_valid"])

    def test_invalid_input_cannot_be_approved_by_a_model(self):
        class Agent:
            def _generate_section(self, *args, **kwargs):
                raise AssertionError("Must fail before model call")
        for support, conflict in [("None", "None"), ("same", "same")]:
            fields = core()
            fields.update(expected_visual_state=support, opposite_visual_state=conflict)
            result = audit_core(Agent(), "Caption", fields)
            self.assertFalse(any(result[key] for key in CHECKS))
            self.assertTrue(result["input_errors"])

    def test_wrong_list_type_fails_without_coercion(self):
        fields = core()
        fields["negation"] = None
        self.assertIn("INVALID_CORE_TYPES", audit_core(None, "Caption", fields)["input_errors"])

    def test_unanchored_source_answer_is_unknown_not_pass(self):
        class Agent:
            def _generate_section(self, *args, **kwargs):
                value = dict(answer="invented", source_quotes=["NOT IN SOURCE"], unknown=False)
                return value, json.dumps(value), 0, {"schema_valid": True}
        result = audit_core(Agent(), "The meeting was calm.", core())
        self.assertFalse(any(result[key] for key in CHECKS))
        self.assertTrue(all(item["status"] == "UNKNOWN" for item in result["obligations"]))

    def test_literal_none_in_sentence_is_not_placeholder(self):
        self.assertEqual(requirement_errors('The sign says "None".', 'The sign says "All".'), [])

    def test_judge_has_one_authoritative_relation(self):
        schema = schema_for_contract("tribunal_review")
        self.assertNotIn("best_semantic_judgment", schema["properties"])
        value = example(schema)
        parsed = parse_tribunal_review_response(json.dumps(value))
        self.assertTrue(parsed["_format_valid"])
        self.assertEqual(parsed["provisional_verdict"], "ENTAILS")
        value["confidence"] = "0.5"
        self.assertFalse(parse_tribunal_review_response(json.dumps(value))["_format_valid"])
