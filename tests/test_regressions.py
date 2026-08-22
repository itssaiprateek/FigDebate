import unittest

from agents.visual_grounding import VisualGroundingAgent
from agents.claim_extraction import ClaimExtractionAgent
from comparators.evidence_comparator import compare
from engine.claim_contract import attach_claim_contract
from engine.debate import DebateEngine
from engine.evidence_ledger import add_visual_reinspection_evidence
from engine.relation_schema import attach_claim_relation, build_claim_relation
from engine.review_board import review_revision
from utils.visual_parser import parse_visual_response


class VisualGroundingRegressionTests(unittest.TestCase):
    def test_bare_nothing_and_template_relations_are_not_evidence(self):
        raw = """Literal Scene: A man and woman stand by a sign.
Objects: Man, Woman, Sign
Scene Type: Street Scene
Visual Facts: The man and woman face opposite directions.
Visual Relations: Left/Right, Top/Bottom
Visible Text: Nothing
Symbolic Elements: Arrow pointing right, Black Hole of It
Possible Visual Metaph
"""
        parsed = parse_visual_response(raw)
        output = VisualGroundingAgent._to_spec_schema(parsed, raw)
        self.assertEqual(output["visible_text"], [])
        self.assertEqual(output["visual_relations"], [])
        self.assertEqual(
            parsed["symbolic_elements"],
            ["Arrow pointing right, Black Hole of It"],
        )
        self.assertEqual(parsed["possible_visual_metaphors"], [])
        self.assertIn(
            "visual_relations_are_placeholders", output["schema_issues"]
        )

    def test_singular_visual_metaphor_heading_does_not_leak(self):
        raw = """Literal Scene: A man displays a heart and wings.
Objects: Man, Heart, Wings
Visual Facts: The man displays a heart.
Visual Relations: The heart is attached to the man.
Visible Text: None
Symbolic Elements: Wings may conventionally suggest freedom.
Possible Visual Metaphor: A winged heart may represent hopeful love.
Scene Type: Symbolic illustration
Confidence: 0.7
"""
        parsed = parse_visual_response(raw)
        self.assertEqual(
            parsed["symbolic_elements"],
            ["Wings may conventionally suggest freedom."],
        )
        self.assertEqual(
            parsed["possible_visual_metaphors"],
            ["A winged heart may represent hopeful love."],
        )

    def test_unstructured_grounding_is_not_schema_complete(self):
        raw = "All written text"
        output = VisualGroundingAgent._to_spec_schema(
            parse_visual_response(raw), raw
        )
        self.assertFalse(output["schema_complete"])
        self.assertFalse(output["schema_format_valid"])
        self.assertFalse(output["factual_grounding_present"])


class RecoveryRoutingRegressionTests(unittest.TestCase):
    def test_insufficient_visual_evidence_always_routes_level_two(self):
        decision = {
            "label": "ENTAILS",
            "confidence": 0.55,
            "_final_decision_valid": True,
        }
        assessment = DebateEngine.debate_assessment(
            decision,
            {
                "required_evidence_status": "INSUFFICIENT_VISUAL_EVIDENCE",
                "visual_schema_complete": False,
            },
        )
        self.assertTrue(assessment["trigger"])
        self.assertEqual(assessment["level"], 2)
        self.assertEqual(assessment["reason"], "insufficient_visual_evidence")

    def test_text_surface_without_ocr_requires_binding_review(self):
        caption = "A black hole absorbs anything in front of it."
        language = attach_claim_relation(
            attach_claim_contract(
                {
                    "original_caption": caption,
                    "caption_proposition": caption,
                    "claim_subject": "black hole",
                    "claim_predicate": "absorbs",
                    "claim_object": "anything in front of it",
                    "relation_family": "outcome",
                    "expected_visual_state": "black hole absorbing an object",
                    "opposite_visual_state": "black hole releasing an object",
                    "figurative_type": "literal",
                },
                caption,
            ),
            caption,
        )
        visual = {
            "visual_description": "Two people stand beside a sign.",
            "objects": ["people", "sign"],
            "scene_type": "street meme",
            "visible_text": [],
            "visual_facts": ["Two people stand beside a sign."],
            "visual_relations": [],
            "schema_complete": True,
        }
        comparison = compare(visual, language, caption)
        self.assertTrue(comparison["has_text_surface"])
        self.assertTrue(comparison["text_surface_without_ocr"])
        self.assertTrue(comparison["relation_binding_required"])
        self.assertFalse(comparison["relation_binding_observed"])
        self.assertFalse(comparison["region_pair_verifier_eligible"])
        assessment = DebateEngine.debate_assessment(
            {
                "label": "CONTRADICTS",
                "confidence": 0.6,
                "_final_decision_valid": True,
            },
            comparison,
        )
        self.assertEqual(
            assessment["reason"], "unresolved_text_relation_semantics"
        )

    def test_generic_ocr_placeholders_are_not_region_evidence(self):
        for value in ("Text", "object label", "unknown", "unreadable"):
            self.assertFalse(VisualGroundingAgent._usable_region_text(value))
        self.assertTrue(
            VisualGroundingAgent._usable_region_text("gone in a week")
        )


class SemanticDebateRegressionTests(unittest.TestCase):
    def test_visual_critique_parser_rejects_prompt_echo(self):
        parsed = VisualGroundingAgent._parse_critique_response(
            "Observed Entity: Man\nObserved State: Heart\n"
            "Expected State: sad\nThe man has a heart.",
            "symbolic_visual_reinspection",
        )
        self.assertFalse(parsed["_format_valid"])
        self.assertFalse(parsed["specific_evidence"])

    def test_visual_critique_parser_accepts_directional_contract(self):
        parsed = VisualGroundingAgent._parse_critique_response(
            "Recommendation: CONTRADICTS\n"
            "Observed Entity: the man's heart\n"
            "Observed State: whole and bright with white wings\n"
            "Image Region: center of the man's chest\n"
            "Claim Relation: CONFLICT\n"
            "Reason: The whole bright heart has positive symbolism opposite to a rotten corrupt heart.",
            "symbolic_visual_reinspection",
        )
        self.assertTrue(parsed["_format_valid"])
        self.assertTrue(parsed["specific_evidence"])

    def test_linguistic_requirements_support_role_equivalent_images(self):
        support, conflict = ClaimExtractionAgent._ground_visual_requirements(
            "A literal black hole absorbs an object",
            "A literal black hole releases an object",
            "literal",
            "Claim subject: black hole\n"
            "Expected visual state: a black hole absorbs matter\n"
            "Opposite visual state: a black hole releases matter",
        )
        self.assertIn("explicitly bound label", support)
        self.assertIn("opposite role relation", conflict)

    def test_metaphor_none_requirements_fall_back_to_claim_contract(self):
        support, conflict = ClaimExtractionAgent._ground_visual_requirements(
            "None (negative sentiment)",
            "None",
            "metaphor",
            "Claim subject: the man\n"
            "Asserted property: rotten heart\n"
            "Intended meaning: the man's character is corrupt\n"
            "Expected visual state: a rotten or damaged heart\n"
            "Opposite visual state: a whole healthy heart",
        )
        self.assertIn("rotten or damaged heart", support)
        self.assertIn("visible symbol attached to the man", support)
        self.assertIn("whole healthy heart", conflict)

    def test_rotten_heart_polarity_is_negative(self):
        polarity = ClaimExtractionAgent._normalize_caption_polarity(
            "neutral",
            "a rotten heart",
            "the man's character is corrupt",
        )
        self.assertEqual(polarity, "negative")

    def test_symbolic_sentiment_review_outranks_spurious_text_binding(self):
        comparison = {
            "required_evidence_status": "GROUNDED_REVIEW_REQUIRED",
            "visual_schema_complete": True,
            "relation_binding_required": True,
            "relation_binding_observed": False,
            "region_pair_verifier_eligible": False,
            "has_symbolic_evidence": True,
            "claim_relation": {"relation_family": "sentiment"},
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "grounded_anchor_evidence": ["an intact heart is attached to the man"],
        }
        assessment = DebateEngine.debate_assessment(
            {"label": "ENTAILS", "confidence": 0.7, "_final_decision_valid": True},
            comparison,
        )
        self.assertNotEqual(
            assessment["reason"], "unresolved_text_relation_semantics"
        )
        engine = DebateEngine.__new__(DebateEngine)
        prompt = engine.build_agent1_challenge_prompt(
            {}, {"label": "ENTAILS"}, comparison
        )
        self.assertIn("FIGURATIVE_SYMBOL_REINSPECTION", prompt)

    def test_structured_symbolic_reinspection_can_prove_conflict(self):
        caption = "His heart within him is fully rotten."
        language = attach_claim_contract(
            {
                "original_caption": caption,
                "caption_proposition": "The man's moral core is rotten.",
                "claim_subject": "the man",
                "claim_predicate": "has a rotten heart",
                "asserted_property": "rotten heart",
                "relation_family": "sentiment",
                "expected_visual_state": "the man has a rotten corrupt heart",
                "opposite_visual_state": "the man has a healthy loving heart",
                "intended_meaning": "The man's character is negative and corrupt.",
                "figurative_type": "metaphor",
            },
            caption,
        )
        relation = build_claim_relation(caption, language)
        ledger = add_visual_reinspection_evidence(
            [],
            {
                "recommendation": "CONTRADICTS",
                "claim_relation": "CONFLICT",
                "specific_evidence": True,
                "observed_entity": "the man's heart",
                "observed_state": "is intact and conventionally represents love",
                "image_region": "center of the man's chest",
                "reason": (
                    "The intact heart is attached to the man and conventionally "
                    "suggests love rather than corruption."
                ),
            },
            {"claim_relation": relation},
        )
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["relation"], "CONFLICT")
        self.assertTrue(ledger[0]["decision_grade"])

    def test_debate_can_keep_a_label_and_strengthen_its_evidence(self):
        ledger = [{
            "id": "DV001",
            "source": "debate_visual_reinspection",
            "type": "verified_reinspection_relation",
            "text": "The man is visibly smiling in the center.",
            "relation": "SUPPORT",
            "grounded": True,
            "decision_grade": True,
            "verification": {"decision_grade": True},
        }]
        accepted, reason, audit = review_revision(
            {"label": "ENTAILS", "confidence": 0.4},
            {
                "label": "ENTAILS",
                "confidence": 0.7,
                "_model_cited_evidence_ids": ["DV001"],
            },
            ledger,
            visual_review={
                "recommendation": "ENTAILS",
                "specific_evidence": True,
                "reason": "The man is visibly smiling in the center.",
            },
        )
        self.assertTrue(accepted)
        self.assertIn("accepted_evidence_backed_confirmation", reason)
        self.assertTrue(audit["valid"])

    def test_unopposed_visual_evidence_controls_revision_proposal(self):
        decision = DebateEngine._apply_unopposed_visual_evidence(
            {"label": "ENTAILS", "confidence": 0.65},
            [{
                "id": "DV001",
                "relation": "CONFLICT",
                "grounded": True,
                "decision_grade": True,
            }],
            {
                "recommendation": "CONTRADICTS",
                "specific_evidence": True,
                "_format_valid": True,
                "reason": "A whole bright heart visibly opposes the rotten-heart claim.",
            },
        )
        self.assertEqual(decision["label"], "CONTRADICTS")
        self.assertEqual(
            decision["decision_method"], "visual_reinspection_consensus"
        )
        self.assertEqual(decision["_model_cited_evidence_ids"], ["DV001"])

    def test_mixed_visual_evidence_does_not_control_revision(self):
        decision = DebateEngine._apply_unopposed_visual_evidence(
            {"label": "ENTAILS", "confidence": 0.65},
            [
                {"id": "DV001", "relation": "CONFLICT", "grounded": True, "decision_grade": True},
                {"id": "DV002", "relation": "SUPPORT", "grounded": True, "decision_grade": True},
            ],
            {
                "recommendation": "CONTRADICTS",
                "specific_evidence": True,
                "_format_valid": True,
                "reason": "The image has conflicting visual evidence.",
            },
        )
        self.assertEqual(decision["label"], "ENTAILS")
        self.assertFalse(decision["_visual_evidence_consensus_applied"])


if __name__ == "__main__":
    unittest.main()
