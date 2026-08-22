import unittest
import os
import tempfile
from PIL import Image

from comparators.evidence_comparator import compare
from engine.claim_contract import attach_claim_contract
from engine.relation_schema import attach_claim_relation
from agents.claim_extraction import ClaimExtractionAgent
from agents.visual_grounding import VisualGroundingAgent
from arbiter.arbiter import Arbiter
from engine.debate import DebateEngine
from engine.evidence_ledger import (
    attach_evidence_audit,
    audit_decision,
    build_evidence_ledger,
)
from engine.evidence_verifier import AtomicEvidenceVerifier, merge_verified_evidence
from engine.feedback_loop import FeedbackLoop
from utils.arbiter_parser import parse_arbiter_response
from utils.claim_parser import parse_claim_response
from utils.visual_parser import parse_list, parse_visual_response
from utils.decision_scoring import (
    evidence_adjusted_confidence,
    position_balanced_relation_scores,
)


def safe_language(caption, **values):
    output = attach_claim_contract({
        "caption_proposition": caption,
        "claim_subject": values.pop("claim_subject", "subject"),
        "relation_family": values.pop("relation_family", "other"),
        "expected_visual_state": values.pop(
            "expected_visual_state", "subject matches expected state"
        ),
        "opposite_visual_state": values.pop(
            "opposite_visual_state", "subject matches opposite state"
        ),
        **values,
    }, caption)
    return attach_claim_relation(output, caption)


class ParserContractTests(unittest.TestCase):
    def test_arbiter_accepts_only_exact_final_label(self):
        response = """Evidence Summary:
Clear evidence.
Final Decision:
ENTAILS
Confidence:
0.80"""
        self.assertEqual(parse_arbiter_response(response)["label"], "ENTAILS")

    def test_arbiter_rejects_template_text_as_a_label(self):
        response = """Final Decision:
ENTAILS or CONTRADICTS
Confidence:
0.20"""
        self.assertIsNone(parse_arbiter_response(response)["label"])

    def test_arbiter_requires_numeric_confidence(self):
        parsed = parse_arbiter_response("Final Decision:\nENTAILS")
        output = Arbiter._to_spec_schema(parsed, "Final Decision:\nENTAILS")
        self.assertFalse(output["_final_decision_valid"])
        self.assertTrue(output["_confidence_was_invalid"])

    def test_arbiter_rejects_out_of_range_confidence(self):
        parsed = parse_arbiter_response("Final Decision:\nENTAILS\nConfidence:\n1.20")
        self.assertIsNone(parsed["confidence"])

    def test_arbiter_choice_prompt_prioritizes_region_bound_review(self):
        prompt = Arbiter._build_relation_choice_prompt(
            "Disliked products disappear quickly.",
            "[TARGETED VISUAL REINSPECTION - region_ocr] "
            "Left object label reads disliked; left bottom outcome reads lasts months.",
            "The caption claims a short duration.",
            "Older OCR was unbound.",
            "The paired outcome conflicts with the caption.",
        )
        self.assertIn("targeted region-bound reinspection", prompt)
        self.assertIn("left bottom outcome reads lasts months", prompt)
        self.assertIn("displayed outcome are not the same", prompt)

    def test_targeted_region_verifier_uses_explicit_pair_binding(self):
        arbiter = Arbiter.__new__(Arbiter)
        caption = (
            "The product I love lasts long while the product I dislike is short."
        )
        language = safe_language(
            caption,
            claim_subject="product",
            claim_target="product",
            relation_family="outcome",
            expected_visual_state=(
                "product love lasts long; product dislike is short"
            ),
            opposite_visual_state=(
                "product love is short; product dislike lasts long"
            ),
        )
        label, confidence, scores, _, _, verification = (
            arbiter._verify_targeted_region_relation(
                caption,
                [
                    {
                        "side": "left",
                        "object_text": "product love",
                        "outcome_text": "short",
                    },
                    {
                        "side": "right",
                        "object_text": "product dislike",
                        "outcome_text": "long",
                    },
                ],
                language["claim_relation"],
            )
        )
        self.assertEqual(label, "CONTRADICTS")
        self.assertGreater(confidence, 0.5)
        self.assertTrue(verification["decision_grade"])
        self.assertEqual(
            scores["verification_method"],
            "deterministic_structured_region_binding",
        )

    def test_type_only_retry_parser_accepts_only_one_type(self):
        self.assertEqual(ClaimExtractionAgent._parse_type_retry("humor"), "humor")
        self.assertEqual(ClaimExtractionAgent._parse_type_retry("humor."), "humor")
        self.assertEqual(ClaimExtractionAgent._parse_type_retry("literal"), "literal")
        self.assertIsNone(ClaimExtractionAgent._parse_type_retry("Type: humor because"))

    def test_visual_list_preserves_commas(self):
        self.assertEqual(parse_list("- Pittsburgh, Pennsylvania"), ["Pittsburgh, Pennsylvania"])

    def test_agent1_reclassifies_object_descriptions_out_of_visible_text(self):
        output = VisualGroundingAgent._to_spec_schema(
            {
                "literal_scene": "A man sits in a boat.",
                "objects": ["Money bags"],
                "visible_text": [
                    "Arrow pointing upwards",
                    "Money bags",
                    "The value is $226",
                ],
                "visual_relations": ["No arrow direction provided"],
                "visual_facts": [],
            },
            "",
        )
        self.assertEqual(output["visible_text"], ["The value is $226"])
        self.assertIn("Arrow pointing upwards", output["visual_relations"])
        self.assertIn("Money bags", output["visual_facts"])

    def test_visual_list_drops_prompt_placeholders_and_progressive_duplicates(self):
        self.assertEqual(
            parse_list(
                "- Directly observed fact\n"
                "- Product you are trying\n"
                "- Product you are trying to use quickly"
            ),
            ["Product you are trying to use quickly"],
        )

    def test_visual_parser_maps_heading_alias_without_leaking(self):
        parsed = parse_visual_response(
            "Visible Text: None\n"
            "Object Names and Phrases (list):\n- boat\n"
            "Visible Relations (observed):\n- Arrow points upward\n"
            "Visual Relations:\n- Arrow: 0\n"
            "Visual Facts:\n- A man sits in a boat.\n"
            "Literal Scene: A man sits in a boat.\n"
            "Uncertain Observations"
        )
        self.assertEqual(parsed["visible_text"], [])
        self.assertNotIn("object_names_and_phrases", parsed)
        self.assertEqual(parsed["objects"], ["boat"])
        self.assertEqual(
            parsed["visual_relations"],
            ["Arrow points upward", "Arrow: 0"],
        )
        self.assertNotIn("Uncertain Observations", parsed["literal_scene"])

    def test_agent1_drops_zero_relation_placeholders(self):
        output = VisualGroundingAgent._to_spec_schema(
            {
                "literal_scene": "A man sits in a boat.",
                "objects": ["boat"],
                "visual_relations": [
                    "Arrow: 0", "Trend: None", "Left/Right: Yes",
                    "Tilted/", "Boat above water",
                ],
                "visual_facts": ["A man sits in a boat."],
            },
            "",
        )
        self.assertEqual(output["visual_relations"], ["Boat above water"])

    def test_comparison_crops_choose_axis_and_overlap(self):
        landscape = Image.new("RGB", (800, 600))
        landscape_crops = VisualGroundingAgent._comparison_crops(landscape)
        self.assertEqual(
            [name for name, _ in landscape_crops],
            [
                "left object label",
                "left bottom outcome",
                "right object label",
                "right bottom outcome",
            ],
        )
        self.assertGreater(landscape_crops[0][1].width, 400)
        self.assertGreater(landscape_crops[0][1].height, 200)
        self.assertLess(landscape_crops[1][1].height, 300)

        portrait = Image.new("RGB", (400, 800))
        portrait_crops = VisualGroundingAgent._comparison_crops(portrait)
        self.assertEqual(len(portrait_crops), 4)
        self.assertGreater(portrait_crops[3][1].height, 200)

    def test_region_review_pairs_each_object_with_same_side_outcome(self):
        agent = VisualGroundingAgent.__new__(VisualGroundingAgent)
        texts = {
            "left object label": "product disliked",
            "left bottom outcome": "lasts for months",
            "right object label": "product loved",
            "right bottom outcome": "gone in a week",
        }
        agent._read_region_text = lambda _crop, name: texts[name]
        result = agent.critique(
            Image.new("RGB", (800, 600)),
            "UNRESOLVED_TEXT_LAYOUT_BINDING",
        )
        self.assertEqual(result["review_method"], "region_ocr")
        self.assertIn(
            "LEFT bottom outcome = 'lasts for months' for object label 'product disliked'",
            result["reason"],
        )
        self.assertIn(
            "RIGHT bottom outcome = 'gone in a week' for object label 'product loved'",
            result["reason"],
        )
        self.assertEqual(
            result["region_pairs"],
            [
                {
                    "side": "left",
                    "object_text": "product disliked",
                    "outcome_text": "lasts for months",
                },
                {
                    "side": "right",
                    "object_text": "product loved",
                    "outcome_text": "gone in a week",
                },
            ],
        )

    def test_object_label_cleanup_removes_pronoun_led_predicate(self):
        self.assertEqual(
            VisualGroundingAgent._clean_object_label(
                "Product you hate and you're trying"
            ),
            "Product you hate",
        )
        self.assertEqual(
            VisualGroundingAgent._clean_object_label("Research and development"),
            "Research and development",
        )

    def test_visual_parser_compacts_long_repeated_ocr_runs(self):
        parsed = parse_visual_response(
            "Visible Text:\ncloudy, sunny, cloudy, rainy, sunny, clear, "
            "windy, foggy, humid, dry, stormy, snowy\n"
            "Literal Scene: A weather forecast."
        )
        self.assertEqual(
            parsed["visible_text"],
            ["cloudy, sunny, rainy, clear, windy, foggy, humid, dry"],
        )

    def test_debate_does_not_accept_absence_based_label_flip(self):
        original = {
            "label": "CONTRADICTS",
            "_binary_resolution_scores": {"ENTAILS": 0.30},
        }
        revised = {
            "label": "ENTAILS",
            "confidence": 0.70,
            "_binary_resolution_scores": {"ENTAILS": 0.70},
            "_debate_agent1_critique": {
                "stance": "CHALLENGE",
                "specific_evidence": True,
                "reason": "A man has money, but there is no clear indication of recovery.",
            },
        }
        accepted, reason = DebateEngine._accept_revision(original, revised, {})
        self.assertFalse(accepted)
        self.assertEqual(reason, "absence_is_not_decision_evidence")

    def test_claim_parser_keeps_sections_separate(self):
        response = """Linguistic Notes:
- Uses irony
Figurative Type:
sarcasm
Background Knowledge:
Context matters.
Confidence:
0.80"""
        parsed = parse_claim_response(response)
        self.assertEqual(parsed["figurative_type"], "sarcasm")
        self.assertEqual(parsed["background_knowledge"], "Context matters.")
        self.assertEqual(parsed["linguistic_notes"], ["Uses irony"])

    def test_claim_parser_keeps_same_line_non_literal_values(self):
        parsed = parse_claim_response("""Non-literal Expressions:
Phrase: more appealing
Literal interpretation: more desirable
Contextual interpretation: more enjoyable
Reason: It is a preference.
Figurative Type:
humor""")
        self.assertEqual(parsed["non_literal_expressions"][0]["expression"], "more appealing")


class ComparatorContractTests(unittest.TestCase):
    def test_downward_visual_trend_contradicts_growth_caption(self):
        visual = {
            "visual_description": "A chart is displayed.",
            "objects": ["chart"], "scene_type": "financial chart",
            "visual_facts": ["The economy chart shows a downward trend."], "symbolic_tone": "None",
        }
        caption = "The economy is recovering."
        language = safe_language(
            caption,
            claim_subject="economy",
            claim_predicate="recovering",
            relation_family="trajectory",
            expected_visual_state="economy chart rising upward",
            opposite_visual_state="economy chart falling downward",
            intended_meaning=caption,
            figurative_type="metaphor",
            explicit_claims=["The economy recovered."],
        )
        result = compare(visual, language, caption=caption)
        self.assertEqual(result["recommendation"], "LEAN_CONTRADICTS")
        self.assertTrue(result["contradicting_evidence"])
        self.assertIn("[VISUAL]", result["contradicting_evidence"][0])

    def test_unrelated_domains_require_semantic_review_without_false_conflict(self):
        visual = {
            "visual_description": "A doctor stands in a hospital.", "objects": ["doctor"],
            "scene_type": "hospital", "visual_facts": ["A doctor is visible."], "symbolic_tone": "None",
        }
        language = {
            "intended_meaning": "The stock market is unstable.", "figurative_type": "metaphor",
            "explicit_claims": ["The market is unstable."],
        }
        result = compare(visual, language)
        self.assertEqual(result["contradicting_evidence"], [])
        self.assertEqual(result["recommendation"], "SEMANTIC_REVIEW")
        self.assertEqual(result["required_evidence_status"], "SEMANTIC_REVIEW_REQUIRED")
        self.assertEqual(result["missing_evidence"], [])

    def test_shared_anchors_require_polarity_review(self):
        visual = {
            "visual_description": "A weather app for Pittsburgh is visible.",
            "objects": ["weather app"], "scene_type": "phone screen",
            "visual_facts": ["The forecast is for Pittsburgh."], "symbolic_tone": "None",
        }
        language = {
            "intended_meaning": "The speaker dislikes Pittsburgh weather.",
            "figurative_type": "sarcasm",
            "explicit_claims": ["Pittsburgh weather is terrible."],
        }
        result = compare(visual, language)
        self.assertEqual(result["required_evidence_status"], "GROUNDED_REVIEW_REQUIRED")
        self.assertTrue(result["grounded_anchor_evidence"])

    def test_caption_is_included_as_a_grounding_source(self):
        visual = {
            "visible_text": ["226.00001"],
            "visual_description": "A financial chart shows 226.00001.",
            "objects": ["chart"],
            "scene_type": "financial chart",
            "visual_facts": ["The chart is rising."],
        }
        language = {
            "intended_meaning": "The speaker is disappointed with the profit.",
            "figurative_type": "sarcasm",
        }
        result = compare(visual, language, caption="$226 ain't enough")
        self.assertIn("number:226", result["direct_evidence_terms"])

    def test_recovery_after_recession_has_growth_direction_only(self):
        visual = {
            "visual_description": "A financial image.",
            "objects": ["chart"],
            "scene_type": "financial chart",
            "visual_facts": [],
            "visual_relations": ["An economy arrow points upward."],
        }
        caption = "The economy recovered relatively easily."
        language = safe_language(
            caption,
            claim_subject="economy",
            claim_predicate="recovered",
            relation_family="trajectory",
            expected_visual_state="economy arrow rising upward",
            opposite_visual_state="economy arrow falling downward",
            surface_meaning="Recovery following a downturn.",
            intended_meaning=caption,
            figurative_type="literal",
        )
        result = compare(
            visual,
            language,
            caption=caption,
        )
        self.assertEqual(result["claim_direction"], "growth")
        self.assertEqual(result["required_evidence_status"], "SUPPORTED")
        self.assertEqual(result["contradicting_evidence"], [])

    def test_unbound_contrast_text_requires_layout_review(self):
        visual = {
            "visual_description": "Two bottles are side by side.",
            "objects": ["two bottles"],
            "scene_type": "comparison meme",
            "visible_text": ["product you love", "gone in a week"],
            "visual_facts": ["The left bottle is white and the right bottle is red."],
            "visual_relations": ["The bottles are side by side."],
        }
        caption = "Disliked products disappear, while loved products last."
        language = safe_language(
            caption,
            claim_subject="products",
            claim_target="products",
            relation_family="outcome",
            expected_visual_state=(
                "disliked products disappear; loved products last"
            ),
            opposite_visual_state=(
                "disliked products last; loved products disappear"
            ),
            surface_meaning=caption,
            intended_meaning=caption,
            figurative_type="literal",
        )
        result = compare(
            visual,
            language,
            caption=caption,
        )
        self.assertTrue(result["relation_binding_required"])
        self.assertFalse(result["relation_binding_observed"])
        self.assertEqual(result["evidence_quality"], 0.55)

    def test_explicit_two_sided_text_binding_is_observed(self):
        visual = {
            "visual_description": "Two bottles are side by side.",
            "objects": ["white bottle", "red bottle"],
            "scene_type": "comparison meme",
            "visible_text": [
                "product you hate", "lasts for months",
                "product you love", "gone in a week",
            ],
            "visual_facts": [
                "The left white bottle says product you hate and lasts for months; "
                "the right red bottle says product you love and gone in a week."
            ],
            "visual_relations": ["The white bottle is left of the red bottle."],
        }
        language = {
            "intended_meaning": "Preference changes product duration.",
            "caption_proposition": "Loved products last while disliked products disappear.",
            "figurative_type": "literal",
        }
        result = compare(
            visual,
            language,
            caption="Disliked products disappear, while loved products last.",
        )
        self.assertTrue(result["relation_binding_required"])
        self.assertTrue(result["relation_binding_observed"])


class DecisionScoringTests(unittest.TestCase):
    def test_arbiter_extracts_only_supported_evidence_id_formats(self):
        ids = Arbiter._extract_evidence_ids(
            "Evidence IDs: VF001, CC002, VF001, MADEUP7"
        )
        self.assertEqual(ids, ["VF001", "CC002"])

    def test_arbiter_resolves_near_verbatim_visual_quote_to_current_id(self):
        ids = Arbiter._resolve_evidence_ids(
            "Visual Evidence: A graph falls sharply from left to right.\n"
            "Evidence IDs: NONE",
            {"grounded_evidence_catalog": [{
                "id": "VF001",
                "text": "The graph falls sharply from left to right.",
            }]},
        )
        self.assertEqual(ids, ["VF001"])

    def test_arbiter_does_not_resolve_broad_semantic_similarity(self):
        ids = Arbiter._resolve_evidence_ids(
            "Visual Evidence: The situation appears economically difficult.",
            {"grounded_evidence_catalog": [{
                "id": "VF001",
                "text": "A person holds coins beside a boat.",
            }]},
        )
        self.assertEqual(ids, [])

    def test_position_balancing_cancels_pure_option_bias(self):
        scores = position_balanced_relation_scores(
            {"A": 2.0, "B": 0.0},
            {"A": 2.0, "B": 0.0},
        )
        self.assertAlmostEqual(scores["ENTAILS"], 0.5)
        self.assertAlmostEqual(scores["CONTRADICTS"], 0.5)

    def test_position_balancing_retains_semantic_support(self):
        scores = position_balanced_relation_scores(
            {"A": 3.0, "B": 0.0},
            {"A": 2.0, "B": 1.0},
        )
        self.assertGreater(scores["ENTAILS"], scores["CONTRADICTS"])

    def test_weak_evidence_shrinks_confidence_toward_half(self):
        self.assertAlmostEqual(evidence_adjusted_confidence(0.9, 0.25), 0.6)


class DebateContractTests(unittest.TestCase):
    def test_invalid_arbiter_output_does_not_start_debate(self):
        engine = DebateEngine()
        self.assertFalse(engine.should_debate({"_final_decision_valid": False}, {}))

    def test_low_confidence_alone_does_not_start_debate(self):
        engine = DebateEngine()
        decision = {"_final_decision_valid": True, "label": "ENTAILS", "confidence": 0.1}
        self.assertFalse(engine.should_debate(decision, {"recommendation": "UNCERTAIN"}))

    def test_uncorroborated_structured_relation_routes_to_level_two(self):
        assessment = DebateEngine.debate_assessment(
            {"label": "ENTAILS", "_final_decision_valid": True},
            {
                "required_evidence_status": "SEMANTIC_REVIEW_REQUIRED",
                "claim_relation": {"resolved": True},
                "structured_relation_candidates": [{"proposed_relation": "CONFLICT"}],
                "atomic_evidence_verification": {"verified_count": 0},
            },
        )
        self.assertTrue(assessment["trigger"])
        self.assertEqual(assessment["level"], 2)
        self.assertIn("uncorroborated_structured_relation", assessment["signals"])

    def test_memory_warning_can_raise_a_relation_case_to_level_one(self):
        assessment = DebateEngine.debate_assessment(
            {
                "label": "CONTRADICTS", "_final_decision_valid": True,
                "explanation": "The relation is insufficient.",
            },
            {
                "required_evidence_status": "SEMANTIC_REVIEW_REQUIRED",
                "feedback_warning": {"failure_patterns": ["unsupported_contradiction"]},
            },
        )
        self.assertTrue(assessment["trigger"])
        self.assertEqual(assessment["level"], 1)

    def test_direct_evidence_disagreement_starts_debate(self):
        engine = DebateEngine()
        decision = {"_final_decision_valid": True, "label": "ENTAILS", "confidence": 0.8}
        comparison = {
            "recommendation": "LEAN_CONTRADICTS",
            "required_evidence_status": "CONFLICTING",
            "contradicting_evidence": ["[VISUAL] A falling graph conflicts with growth."],
        }
        self.assertTrue(engine.should_debate(decision, comparison))

    def test_direct_support_disagreement_starts_debate(self):
        engine = DebateEngine()
        decision = {"_final_decision_valid": True, "label": "CONTRADICTS", "confidence": 0.8}
        comparison = {
            "recommendation": "LEAN_ENTAILS",
            "required_evidence_status": "SUPPORTED",
            "supporting_evidence": ["[VISUAL] The price 226 matches the caption claim."],
        }
        self.assertTrue(engine.should_debate(decision, comparison))

    def test_unresolved_layout_binding_starts_debate(self):
        engine = DebateEngine()
        decision = {
            "_final_decision_valid": True,
            "label": "ENTAILS",
            "confidence": 0.9,
        }
        comparison = {
            "relation_binding_required": True,
            "relation_binding_observed": False,
        }
        self.assertEqual(
            engine.debate_trigger_reason(decision, comparison),
            "unresolved_text_layout_binding",
        )

    def test_layout_debate_prompt_requests_both_bindings(self):
        engine = DebateEngine()
        prompt = engine.build_agent1_challenge_prompt(
            {"visible_text": ["left phrase", "right phrase"]},
            {"label": "ENTAILS", "explanation": "Draft relation."},
            {
                "relation_binding_required": True,
                "relation_binding_observed": False,
            },
        )
        self.assertIn("Include both sides", prompt)
        self.assertIn("object-to-text bindings", prompt)

    def test_visible_text_semantic_case_routes_to_level_two(self):
        assessment = DebateEngine.debate_assessment(
            {"label": "ENTAILS", "_final_decision_valid": True},
            {
                "required_evidence_status": "SEMANTIC_REVIEW_REQUIRED",
                "has_visible_text": True,
                "has_visual_relations": True,
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "claim_relation": {"resolved": False},
            },
        )
        self.assertTrue(assessment["trigger"])
        self.assertEqual(assessment["level"], 2)
        self.assertIn("unresolved_text_relation_semantics", assessment["signals"])

    def test_symbolic_relation_case_routes_to_level_two(self):
        assessment = DebateEngine.debate_assessment(
            {"label": "ENTAILS", "_final_decision_valid": True},
            {
                "required_evidence_status": "GROUNDED_REVIEW_REQUIRED",
                "has_symbolic_evidence": True,
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "claim_relation": {"resolved": True},
            },
        )
        self.assertTrue(assessment["trigger"])
        self.assertEqual(assessment["level"], 2)

    def test_linguistic_debate_prompt_excludes_nested_debug_payloads(self):
        prompt = DebateEngine().build_agent2_challenge_prompt(
            {
                "original_caption": "His heart is rotten.",
                "caption_proposition": "The man's emotional core is rotten.",
                "claim_subject": "the man",
                "claim_contract": {"warnings": []},
                "raw_output": "x" * 5000,
                "_internal": {"raw_output": "y" * 5000},
            },
            {"label": "ENTAILS"},
        )
        self.assertIn("Caption proposition", prompt)
        self.assertNotIn("raw_output", prompt)
        self.assertNotIn("_internal", prompt)
        self.assertLess(len(prompt), 2500)

    def test_round_three_is_disabled_by_default(self):
        self.assertFalse(DebateEngine().enable_round3)

    def test_generated_revision_text_cannot_replace_grounded_provenance(self):
        original = {"label": "ENTAILS"}
        revised = {
            "label": "CONTRADICTS",
            "contradictions": ["The revised answer claims a strong conflict."],
            "_debate_agent1_critique": {
                "stance": "CHALLENGE",
                "specific_evidence": True,
                "reason": "The chart falls from left to right.",
            },
        }
        accepted, reason = DebateEngine._accept_revision(
            original, revised, {}, evidence_ledger=[]
        )
        self.assertFalse(accepted)
        self.assertEqual(
            reason, "revision_did_not_cite_current_image_evidence"
        )

    def test_grounded_conflict_can_support_a_debate_revision(self):
        original = {"label": "ENTAILS", "confidence": 0.6}
        revised = {
            "label": "CONTRADICTS",
            "confidence": 0.7,
            "_evidence_audit": {
                "source_cited_evidence_ids": ["CC001"]
            },
            "_debate_agent1_critique": {
                "stance": "CHALLENGE",
                "specific_evidence": True,
                "reason": "The chart falls from left to right.",
            },
        }
        ledger = [{
            "id": "CC001",
            "source": "comparator",
            "type": "direct_conflict",
            "text": "A falling graph conflicts with growth.",
            "relation": "CONFLICT",
            "grounded": True,
            "decision_grade": True,
        }]
        accepted, reason = DebateEngine._accept_revision(
            original, revised, {}, evidence_ledger=ledger
        )
        self.assertTrue(accepted)
        self.assertIn("CC001", reason)

    def test_targeted_directional_upgrade_can_override_unsupported_confidence(self):
        ledger = [{
            "id": "TV001",
            "source": "targeted_region_verifier",
            "type": "verified_region_relation",
            "text": "The disliked product lasts months; the loved product is gone.",
            "relation": "CONFLICT",
            "grounded": True,
            "decision_grade": True,
        }]
        original = {
            "label": "ENTAILS",
            "confidence": 0.78,
            "_evidence_audit": {"valid": False},
        }
        revised = {
            "label": "CONTRADICTS",
            "confidence": 0.53,
            "_evidence_audit": {
                "valid": True,
                "source_cited_evidence_ids": ["TV001"],
            },
            "_debate_agent1_critique": {
                "stance": "CHALLENGE",
                "specific_evidence": True,
                "reason": (
                    "The disliked product lasts months; the loved product is gone."
                ),
            },
        }
        accepted, reason = DebateEngine._accept_revision(
            original, revised, {}, ledger
        )
        self.assertTrue(accepted)
        self.assertIn("TV001", reason)

    def test_atomic_evidence_must_match_the_visual_challenge(self):
        revised = {
            "label": "CONTRADICTS",
            "confidence": 0.7,
            "_evidence_audit": {
                "source_cited_evidence_ids": ["VF001"]
            },
            "_debate_agent1_critique": {
                "stance": "CHALLENGE",
                "specific_evidence": True,
                "reason": "A smiling person is holding a cup.",
            },
        }
        ledger = [{
            "id": "VF001", "source": "agent1", "type": "visual_fact",
            "text": "A graph falls sharply from left to right.",
            "relation": "CONFLICT", "grounded": True,
            "decision_grade": True,
            "verification": {"decision_grade": True},
        }]
        accepted, reason = DebateEngine._accept_revision(
            {"label": "ENTAILS", "confidence": 0.6}, revised, {}, ledger
        )
        self.assertFalse(accepted)
        self.assertEqual(
            reason, "revision_not_linked_to_visual_review"
        )

    def test_level_one_debate_reuses_loaded_language_runtime(self):
        class FakeAgent2:
            @staticmethod
            def critique(_caption, _prompt):
                return {
                    "stance": "ENDORSE",
                    "reason": "The caption claim frame preserves its polarity.",
                }

        class FakeArbiter:
            @staticmethod
            def analyze(*_args, **_kwargs):
                return {
                    "label": "CONTRADICTS",
                    "confidence": 0.75,
                    "explanation": "[CC001] The current image conflicts.",
                    "_model_cited_evidence_ids": ["CC001"],
                    "_final_decision_valid": True,
                    "_timing": {"total_seconds": 0.01},
                }

        engine = DebateEngine()
        results = engine.run_debate_batch(
            [{
                "key": "case",
                "caption": "The value rises.",
                "image": object(),
                "visual_output": {},
                "language_output": {},
                "comparison": {},
                "decision": {
                    "label": "ENTAILS", "confidence": 0.6,
                    "_final_decision_valid": True,
                },
                "evidence_ledger": [{
                    "id": "CC001", "source": "comparator",
                    "type": "direct_conflict",
                    "text": "The current image visibly conflicts.",
                    "relation": "CONFLICT", "grounded": True,
                    "decision_grade": True,
                }],
                "debate_level": 1,
                "debate_score": 5,
                "debate_signals": ["direct_evidence_disagreement"],
            }],
            language_runtime={
                "agent2": FakeAgent2(), "arbiter": FakeArbiter()
            },
        )
        self.assertEqual(results["case"]["label"], "CONTRADICTS")
        self.assertTrue(results["case"]["_debate"]["revision_accepted"])
        self.assertEqual(
            engine.last_batch_timing["language_model_load_seconds"], 0.0
        )


class EvidenceLedgerContractTests(unittest.TestCase):
    def test_ledger_preserves_direct_support_provenance(self):
        ledger = build_evidence_ledger(
            {"visual_facts": ["A graph rises."]},
            {"caption_proposition": "The value increased."},
            {"supporting_evidence": ["The rising graph supports increased value."]},
        )
        audit = audit_decision({
            "label": "ENTAILS",
            "_model_cited_evidence_ids": ["CS001"],
        }, ledger)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["cited_evidence_ids"], ["CS001"])

    def test_neutral_visual_fact_is_valid_source_but_not_directional_proof(self):
        ledger = build_evidence_ledger(
            {"visual_facts": ["A person smiles beside a damaged bicycle."]},
            {"caption_proposition": "The situation is going well."},
            {},
        )
        audit = audit_decision({
            "label": "ENTAILS",
            "_model_cited_evidence_ids": ["VF001"],
        }, ledger)
        self.assertTrue(audit["source_valid"])
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["status"], "GROUNDED_SOURCE_CITED")

    def test_symbolic_output_is_an_anchor_not_directional_proof(self):
        ledger = build_evidence_ledger(
            {
                "symbolic_tone": (
                    "A winged heart may conventionally suggest love or hope."
                )
            },
            {"caption_proposition": "His heart is rotten."},
            {},
        )
        self.assertEqual(ledger[0]["type"], "symbolic_anchor")
        self.assertEqual(ledger[0]["relation"], "ANCHOR")
        self.assertFalse(ledger[0]["decision_grade"])

    def test_targeted_verifier_is_source_valid_and_directionally_valid(self):
        ledger = [{
            "id": "TV001",
            "source": "targeted_region_verifier",
            "type": "verified_region_relation",
            "text": "Bound region outcomes conflict with the caption.",
            "relation": "CONFLICT",
            "grounded": True,
            "decision_grade": True,
        }]
        audit = audit_decision({
            "label": "CONTRADICTS",
            "decision_method": "targeted_region_verifier",
        }, ledger)
        self.assertTrue(audit["source_valid"])
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["source_cited_evidence_ids"], ["TV001"])

    def test_attach_evidence_audit_does_not_mutate_original_decision(self):
        decision = {"label": "ENTAILS"}
        output = attach_evidence_audit(decision, [])
        self.assertNotIn("_evidence_audit", decision)
        self.assertFalse(output["_evidence_audit"]["valid"])


class AtomicEvidenceVerifierContractTests(unittest.TestCase):
    def test_atomic_verifier_abstains_below_probability_threshold(self):
        relation, _, _ = AtomicEvidenceVerifier._resolve_relation({
            "entailment": 0.55,
            "contradiction": 0.20,
            "neutral": 0.25,
        })
        self.assertEqual(relation, "NEUTRAL")

    def test_atomic_verifier_accepts_high_margin_conflict(self):
        relation, selected, margin = AtomicEvidenceVerifier._resolve_relation({
            "entailment": 0.05,
            "contradiction": 0.88,
            "neutral": 0.07,
        })
        self.assertEqual(relation, "CONFLICT")
        self.assertGreaterEqual(selected, 0.70)
        self.assertGreaterEqual(margin, 0.15)

    def test_uncorroborated_nli_candidate_is_not_merged(self):
        ledger = [{
            "id": "VF001",
            "source": "agent1",
            "type": "visual_fact",
            "text": "The graph falls.",
            "relation": "CONFLICT",
            "grounded": True,
            "verification": {
                "decision_grade": False,
                "candidate_relation": "CONFLICT",
            },
        }]
        output = merge_verified_evidence(
            {
                "required_evidence_status": "SEMANTIC_REVIEW_REQUIRED",
                "recommendation": "SEMANTIC_REVIEW",
            },
            ledger,
            {"verified_count": 1},
        )
        self.assertEqual(
            output["required_evidence_status"], "SEMANTIC_REVIEW_REQUIRED"
        )
        self.assertEqual(
            output["_pre_verification_status"], "SEMANTIC_REVIEW_REQUIRED"
        )
        self.assertEqual(output["contradicting_evidence"], [])

    def test_structured_and_nli_agreement_remains_diagnostic(self):
        class StubNli:
            MODEL_ID = "stub"
            REVISION = "test"

            @staticmethod
            def predict_batch(_pairs):
                return [{"entailment": 0.90, "contradiction": 0.04, "neutral": 0.06}]

        verifier = AtomicEvidenceVerifier(nli_verifier=StubNli())
        ledger, summary = verifier.verify(
            [{
                "id": "VF001", "source": "agent1", "type": "visual_fact",
                "text": "The graph is rising.", "relation": "NEUTRAL",
                "grounded": True,
            }],
            {"caption_proposition": "The value is rising."},
            {"structured_relation_candidates": [{
                "text": "The graph is rising.",
                "proposed_relation": "SUPPORT",
                "relation_family": "trajectory",
                "matched_cues": ["rising"],
            }]},
        )
        self.assertEqual(ledger[0]["relation"], "NEUTRAL")
        self.assertFalse(ledger[0]["verification"]["decision_grade"])
        self.assertEqual(summary["verified_count"], 0)
        self.assertTrue(ledger[0]["verification"]["diagnostic_agreement"])

    def test_nli_without_structured_nomination_remains_neutral(self):
        class StubNli:
            MODEL_ID = "stub"
            REVISION = "test"

            @staticmethod
            def predict_batch(_pairs):
                return [{"entailment": 0.90, "contradiction": 0.04, "neutral": 0.06}]

        verifier = AtomicEvidenceVerifier(nli_verifier=StubNli())
        ledger, summary = verifier.verify(
            [{
                "id": "VF001", "source": "agent1", "type": "visual_fact",
                "text": "A statue is visible.", "relation": "NEUTRAL",
                "grounded": True,
            }],
            {"caption_proposition": "Justice continues."},
            {},
        )
        self.assertEqual(ledger[0]["relation"], "NEUTRAL")
        self.assertFalse(ledger[0]["verification"]["decision_grade"])
        self.assertEqual(summary["verified_count"], 0)


class FeedbackContractTests(unittest.TestCase):
    def test_feedback_without_ground_truth_does_not_create_a_candidate(self):
        loop = FeedbackLoop()
        event = loop.generate_feedback(
            {"visual_description": "A dog is visible."},
            {"figurative_type": "humor"},
            {},
            {"label": "ENTAILS"},
        )
        self.assertFalse(event["candidate_recorded"])

    def test_verified_error_is_only_a_candidate_not_a_prompt_update(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = FeedbackLoop(log_file=os.path.join(temp_dir, "feedback.json"))
            event = loop.record_verified_error(
                {"visual_description": "A dog is visible."},
                {"figurative_type": "humor", "intended_meaning": "A joke."},
                {},
                {"label": "CONTRADICTS"},
                "ENTAILS",
                "humor",
            )
        self.assertTrue(event["candidate_recorded"])
        self.assertEqual(loop.agent1_memory, [])
        self.assertEqual(loop.agent2_memory, [])

    def test_calibration_adds_only_a_preapproved_general_rule(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = FeedbackLoop(log_file=os.path.join(temp_dir, "feedback.json"))
            caption = "The profit of $226 isn't enough."
            language = safe_language(
                caption,
                claim_subject="profit",
                relation_family="quantity",
                expected_visual_state="profit 226 is insufficient",
                opposite_visual_state="profit 226 is sufficient",
                figurative_type="sarcasm",
                intended_meaning="The profit is insufficient.",
            )
            event = loop.generate_feedback(
                {"visual_description": "A chart shows price 226."},
                language,
                {
                    "required_evidence_status": "SUPPORTED",
                    "supporting_evidence": ["[VISUAL] price 226 overlaps the claim."],
                    "contradicting_evidence": [],
                },
                {"label": "CONTRADICTS"},
                "ENTAILS",
                "sarcasm",
                apply_calibration=True,
                evidence_ledger=[{
                    "id": "CS001", "source": "comparator",
                    "type": "direct_support", "text": "The price matches.",
                    "relation": "SUPPORT", "grounded": True,
                    "decision_grade": True,
                }],
            )
        self.assertTrue(event["update_applied"])
        self.assertEqual(event["target_agent"], "arbiter")
        self.assertEqual(len(loop.arbiter_memory), 1)

    def test_calibration_never_memorizes_a_correct_prediction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = FeedbackLoop(log_file=os.path.join(temp_dir, "feedback.json"))
            event = loop.generate_feedback(
                {"visual_facts": ["A weather forecast is visible."]},
                {"figurative_type": "sarcasm"},
                {"required_evidence_status": "GROUNDED_REVIEW_REQUIRED"},
                {"label": "ENTAILS"},
                "ENTAILS",
                "sarcasm",
                apply_calibration=True,
                evidence_ledger=[{
                    "id": "VF001", "source": "agent1",
                    "type": "visual_fact",
                    "text": "A weather forecast is visible.",
                    "relation": "NEUTRAL", "grounded": True,
                }],
            )
        self.assertFalse(event["update_applied"])
        self.assertIsNone(event["failure_type"])
        self.assertEqual(loop.arbiter_memory, [])

    def test_verified_feedback_is_inactive_without_matching_case_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = FeedbackLoop(log_file=os.path.join(temp_dir, "feedback.json"))
            loop.add_verified_example(
                "arbiter",
                loop.CALIBRATED_RULES["missed_grounded_support"],
                "missed_grounded_support",
            )
            prompt = loop.build_prompt(
                "arbiter",
                "Decide.",
                context={"comparison": {"required_evidence_status": "CONFLICTING"}},
            )
        self.assertIsNone(prompt)

    def test_verified_feedback_activates_only_for_matching_grounded_signal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = FeedbackLoop(log_file=os.path.join(temp_dir, "feedback.json"))
            loop.add_verified_example(
                "arbiter",
                loop.CALIBRATED_RULES["missed_grounded_support"],
                "missed_grounded_support",
            )
            context = {
                "comparison": {
                    "required_evidence_status": "SUPPORTED",
                    "supporting_evidence": ["A visible number matches the claim."],
                }
            }
            prompt = loop.build_prompt("arbiter", "Decide.", context=context)
        self.assertIn("missed_grounded_support", prompt)

    def test_case_memory_requires_matching_relation_and_similarity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = FeedbackLoop(log_file=os.path.join(temp_dir, "feedback.json"))
            caption = "The value increased sharply."
            language = safe_language(
                caption,
                claim_subject="value",
                relation_family="trajectory",
                expected_visual_state="value graph rises sharply",
                opposite_visual_state="value graph falls sharply",
                figurative_type="metaphor",
                intended_meaning=caption,
            )
            context = {
                "visual_output": {"visual_facts": ["A graph falls sharply."]},
                "language_output": language,
                "comparison": {"required_evidence_status": "CONFLICTING"},
                "evidence_ledger": [{
                    "id": "CC001", "source": "comparator", "type": "direct_conflict",
                    "text": "A graph falls sharply.", "relation": "CONFLICT",
                    "grounded": True, "decision_grade": True, "verification": {
                        "decision_grade": True, "selected_probability": 0.9
                    },
                }],
            }
            self.assertTrue(loop.add_verified_case(
                context,
                "CONTRADICTS",
                "missed_grounded_conflict",
                loop.CALIBRATED_RULES["missed_grounded_conflict"],
                {"sample_id": "dev_case_1"},
            ))
            matches = loop.matching_rules("arbiter", context)
        self.assertEqual(matches[0]["memory_id"], "dev_case_1")
        self.assertGreaterEqual(matches[0]["_match_score"], loop.MIN_CASE_SIMILARITY)

    def test_procedural_memory_does_not_store_source_case_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = FeedbackLoop(log_file=os.path.join(temp_dir, "feedback.json"))
            ledger = [{
                "id": "VF001", "source": "agent1", "type": "visual_fact",
                "text": "A graph falls sharply.", "relation": "NEUTRAL",
                "grounded": True,
            }]
            decision = attach_evidence_audit({
                "label": "ENTAILS",
                "_model_cited_evidence_ids": ["VF001"],
            }, ledger)
            caption = "The value increased sharply."
            language = safe_language(
                caption,
                claim_subject="value",
                relation_family="trajectory",
                expected_visual_state="value graph rises sharply",
                opposite_visual_state="value graph falls sharply",
                figurative_type="metaphor",
                intended_meaning=caption,
            )
            context = {
                "visual_output": {"visual_facts": ["A graph falls sharply."]},
                "language_output": language,
                "comparison": {
                    "required_evidence_status": "SEMANTIC_REVIEW_REQUIRED"
                },
                "decision": decision,
                "evidence_ledger": ledger,
            }
            added = loop.add_verified_case(
                context,
                "CONTRADICTS",
                "missed_grounded_conflict",
                loop.CALIBRATED_RULES["missed_grounded_conflict"],
                {"sample_id": "dev_source_case"},
            )
        self.assertTrue(added)
        self.assertNotIn("source_evidence_ids", loop.arbiter_memory[0])
        self.assertNotIn("verified_relation", loop.arbiter_memory[0])

    def test_feedback_revision_needs_stronger_grounded_direction(self):
        ledger = [
            {
                "id": "CC001", "source": "comparator", "type": "direct_conflict",
                "text": "A graph falls.", "relation": "CONFLICT", "grounded": True,
                "verification": {
                    "decision_grade": True, "selected_probability": 0.88
                },
            }
        ]
        accepted, reason = FeedbackLoop.accept_feedback_revision(
            {"label": "ENTAILS", "confidence": 0.6},
            {
                "label": "CONTRADICTS",
                "confidence": 0.7,
                "_evidence_audit": {
                    "source_cited_evidence_ids": ["CC001"]
                },
                "_binary_resolution_scores": {
                    "ENTAILS": 0.3, "CONTRADICTS": 0.7
                },
            },
            ledger,
            [{"memory_type": "procedural_case", "failure_type": "review"}],
        )
        self.assertTrue(accepted)
        self.assertIn("CC001", reason)


if __name__ == "__main__":
    unittest.main()
