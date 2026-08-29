import unittest
import os
import tempfile

from comparators.evidence_comparator import compare
from engine.claim_contract import attach_claim_contract, audit_claim_contract
from engine.evidence_ledger import (
    add_visual_reinspection_evidence,
    audit_decision,
)
from engine.feedback_loop import FeedbackLoop
from engine.relation_schema import build_claim_relation, nominate_visual_relations
from engine.region_verifier import verify_region_pairs
from engine.review_board import attach_final_review, review_revision
from utils.claim_parser import parse_claim_response
from utils.decision_scoring import (
    evidence_adjusted_confidence,
    position_balanced_relation_scores,
)
from utils.visual_parser import parse_list, parse_visual_response


class PureParserTests(unittest.TestCase):
    def test_visual_relations_are_parsed(self):
        parsed = parse_visual_response(
            "Visible Text:\nNone\n"
            "Visual Relations:\n- The entire meeting is tilted clockwise.\n"
            "Visual Facts:\n- Eight people are visible.\n"
            "Literal Scene:\nA tilted meeting scene.\n"
            "Objects:\n- table\nScene Type:\nmeeting\nConfidence:\n0.8"
        )
        self.assertEqual(
            parsed["visual_relations"],
            ["The entire meeting is tilted clockwise."],
        )

    def test_progressive_lines_are_compacted(self):
        self.assertEqual(
            parse_list(
                "- Product you love\n"
                "- Product you love lasts\n"
                "- Product you love lasts a long time"
            ),
            ["Product you love lasts a long time"],
        )

    def test_caption_proposition_is_parsed(self):
        parsed = parse_claim_response(
            "Figurative Type: literal\n"
            "Caption Proposition: The economy recovered.\n"
            "Confidence: 0.9"
        )
        self.assertEqual(parsed["caption_proposition"], "The economy recovered.")

    def test_claim_relation_sections_are_parsed(self):
        parsed = parse_claim_response(
            "Caption Proposition: The crowd moves slowly.\n"
            "Claim Subject: the crowd\n"
            "Claim Predicate: moves slowly\n"
            "Claim Object: None\n"
            "Relation Family: pace\n"
            "Expected Visual State: people walking slowly\n"
            "Opposite Visual State: people running quickly\n"
            "Confidence: 0.9"
        )
        self.assertEqual(parsed["relation_family"], "pace")
        self.assertEqual(parsed["opposite_visual_state"], "people running quickly")

    def test_extended_claim_frame_is_parsed(self):
        parsed = parse_claim_response(
            "Caption Proposition: Kanye blames Jewish people.\n"
            "Claim Subject: Kanye\n"
            "Claim Source: Same as subject\n"
            "Claim Target: Jewish people\n"
            "Asserted Property: responsible for his problems\n"
            "Caption Polarity: negative\n"
            "Confidence: 0.9"
        )
        self.assertEqual(parsed["claim_source"], "Same as subject")
        self.assertEqual(parsed["claim_target"], "Jewish people")

    def test_parenthetical_claim_headings_are_parsed(self):
        parsed = parse_claim_response(
            "Alternative Interpretation (if literal): a physical heart.\n"
            "Expected Visual State: the man appears emotionally damaged.\n"
            "Opposite Visual State (if literal): the man appears happy.\n"
        )
        self.assertEqual(
            parsed["alternative_interpretation"], "a physical heart."
        )
        self.assertEqual(
            parsed["opposite_visual_state"], "the man appears happy."
        )


class ClaimContractTests(unittest.TestCase):
    def test_complete_claim_frame_is_safe(self):
        audit = audit_claim_contract(
            "Kanye blames Jewish people for his problems.",
            {
                "caption_proposition": (
                    "Kanye blames Jewish people for his problems."
                ),
                "claim_subject": "Kanye",
                "claim_source": "Same as subject",
                "claim_target": "Jewish people",
                "expected_visual_state": "Kanye assigns blame to Jewish people",
                "opposite_visual_state": "Kanye does not assign that blame",
            },
        )
        self.assertTrue(audit["safe_for_directional_reasoning"])

    def test_dropped_negation_is_not_safe(self):
        audit = audit_claim_contract(
            "$226 isn't enough.",
            {
                "caption_proposition": "$226 is enough.",
                "claim_subject": "$226",
                "expected_visual_state": "226 is insufficient",
                "opposite_visual_state": "226 is sufficient",
            },
        )
        self.assertFalse(audit["safe_for_directional_reasoning"])
        self.assertIn("caption_negation_changed_or_dropped", audit["warnings"])

    def test_replaced_entity_is_not_safe(self):
        audit = audit_claim_contract(
            "Kanye blames Jewish people.",
            {
                "caption_proposition": "Kanye blames Jewish people.",
                "claim_subject": "Kanye",
                "claim_target": "record producers",
                "expected_visual_state": "Kanye blames Jewish people",
                "opposite_visual_state": "Kanye blames someone else",
            },
        )
        self.assertFalse(audit["safe_for_directional_reasoning"])
        self.assertIn("claim_target_not_grounded_in_caption", audit["warnings"])

    def test_unverified_entity_aliases_do_not_make_contract_safe(self):
        caption = (
            "Products one dislikes disappear quickly, while products one "
            "loves last a long time."
        )
        audit = audit_claim_contract(
            caption,
            {
                "caption_proposition": (
                    "People consume disliked items more quickly than loved items."
                ),
                "claim_subject": "Items or products",
                "claim_object": "disliked items, loved items",
                "claim_source": "Unspecified",
                "claim_target": "Items or products",
                "expected_visual_state": "disliked products disappear quickly",
                "opposite_visual_state": "loved products disappear quickly",
            },
        )
        self.assertFalse(audit["safe_for_directional_reasoning"])
        self.assertIsNone(audit["entity_checks"]["claim_source"])

    def test_human_pronoun_coreference_preserves_entity_frame(self):
        audit = audit_claim_contract(
            "His heart within him is fully rotten.",
            {
                "caption_proposition": "The man's emotional core is rotten.",
                "claim_subject": "The man",
                "claim_source": "Unknown (implicitly the speaker)",
                "claim_target": "The man",
                "expected_visual_state": "the man appears emotionally damaged",
                "opposite_visual_state": "the man appears emotionally healthy",
            },
        )
        self.assertTrue(audit["safe_for_directional_reasoning"])

    def test_explicit_human_entity_substitution_remains_blocked(self):
        audit = audit_claim_contract(
            "The woman is worried.",
            {
                "caption_proposition": "The woman is worried.",
                "claim_subject": "the man",
                "expected_visual_state": "the man appears worried",
                "opposite_visual_state": "the man appears calm",
            },
        )
        self.assertFalse(audit["safe_for_directional_reasoning"])
        self.assertIn("claim_subject_not_grounded_in_caption", audit["warnings"])

    def test_unrelated_proposition_without_caption_anchor_is_blocked(self):
        audit = audit_claim_contract(
            "Justice is an ongoing process.",
            {
                "caption_proposition": "A bicycle was repaired yesterday.",
                "claim_subject": "Justice",
                "expected_visual_state": "justice continues",
                "opposite_visual_state": "justice stops",
            },
        )
        self.assertFalse(audit["proposition_preserved"])
        self.assertIn(
            "caption_proposition_has_no_source_anchor", audit["warnings"]
        )


class PureComparatorTests(unittest.TestCase):
    def test_semantic_gap_is_review_not_contradiction(self):
        result = compare(
            {
                "visual_description": "People and a table are tilted clockwise.",
                "visual_relations": ["The entire meeting is tilted clockwise."],
                "visual_facts": ["People sit around a table."],
                "objects": ["table"],
                "scene_type": "meeting",
                "schema_complete": True,
            },
            {
                "surface_meaning": "A welcome at an orientation.",
                "intended_meaning": "A welcome at an orientation.",
                "figurative_type": "literal",
                "caption_proposition": "The audience is welcomed to orientation.",
            },
            caption="Welcome to orientation.",
        )
        self.assertEqual(result["recommendation"], "SEMANTIC_REVIEW")
        self.assertEqual(result["required_evidence_status"], "SEMANTIC_REVIEW_REQUIRED")
        self.assertFalse(result["contradicting_evidence"])

    def test_caption_number_creates_a_grounded_anchor(self):
        result = compare(
            {
                "visual_description": "A chart displays 226.",
                "visible_text": ["226"],
                "visual_facts": ["A chart is visible."],
                "objects": ["chart"],
                "scene_type": "financial chart",
            },
            {
                "intended_meaning": "The speaker is disappointed with the profit.",
                "figurative_type": "sarcasm",
            },
            caption="$226 ain't enough",
        )
        self.assertIn("number:226", result["direct_evidence_terms"])

    @staticmethod
    def _growth_language():
        caption = "The economy is recovering."
        return attach_claim_contract(
            {
                "caption_proposition": caption,
                "claim_subject": "economy",
                "relation_family": "trajectory",
                "expected_visual_state": "economy rising and improving",
                "opposite_visual_state": "economy falling and declining",
            },
            caption,
        )

    def test_negated_chart_direction_is_not_direct_support(self):
        result = compare(
            {"visual_facts": ["The chart shows no upward trend."]},
            self._growth_language(),
            caption="The economy is recovering.",
        )
        self.assertEqual(result["supporting_evidence"], [])

    def test_opposing_chart_directions_are_reported_as_candidates(self):
        result = compare(
            {
                "visual_facts": [
                    "The chart has an upward trend.",
                    "The chart also has a downward trend.",
                ]
            },
            self._growth_language(),
            caption="The economy is recovering.",
        )
        self.assertTrue(result["relation_support_candidates"])
        self.assertTrue(result["relation_conflict_candidates"])
        self.assertFalse(result["supporting_evidence"])
        self.assertFalse(result["contradicting_evidence"])
        self.assertEqual(
            result["required_evidence_status"],
            "MIXED_RELATION_CANDIDATES",
        )

    def test_structured_relation_nominates_explicit_opposite(self):
        language = attach_claim_contract(
            {
                "caption_proposition": "The people move at a leisurely pace.",
                "claim_subject": "people",
                "claim_predicate": "move at a leisurely pace",
                "relation_family": "pace",
                "expected_visual_state": "people walking slowly",
                "opposite_visual_state": "people running quickly",
            },
            "The people move at a leisurely pace.",
        )
        relation = build_claim_relation(
            "The people move at a leisurely pace.",
            language,
        )
        nominations = nominate_visual_relations(
            {"visual_facts": ["Many people are running and rushing down the street."]},
            relation,
        )
        self.assertTrue(relation["resolved"])
        self.assertEqual(relation["relation_family"], "pace")
        self.assertEqual(nominations[0]["proposed_relation"], "CONFLICT")

    def test_directional_nomination_requires_a_caption_entity(self):
        language = attach_claim_contract(
            {
                "caption_proposition": "The people move at a leisurely pace.",
                "claim_subject": "people",
                "relation_family": "pace",
                "expected_visual_state": "people walking slowly",
                "opposite_visual_state": "people running quickly",
            },
            "The people move at a leisurely pace.",
        )
        relation = build_claim_relation(
            "The people move at a leisurely pace.", language
        )
        self.assertEqual(
            nominate_visual_relations(
                {"visual_facts": ["A vehicle is moving quickly."]}, relation
            ),
            [],
        )

    def test_two_entity_bound_cues_become_conflict_candidate(self):
        caption = "The people move at a leisurely pace."
        language = attach_claim_contract(
            {
                "caption_proposition": caption,
                "claim_subject": "people",
                "relation_family": "pace",
                "expected_visual_state": "people walking slowly",
                "opposite_visual_state": "people running and rushing quickly",
            },
            caption,
        )
        language["claim_relation"] = build_claim_relation(caption, language)
        result = compare(
            {
                "visual_facts": [
                    "The people are running and rushing through the street."
                ],
                "schema_complete": True,
            },
            language,
            caption=caption,
        )
        self.assertEqual(
            result["recommendation"], "VERIFY_CONFLICT_CANDIDATE"
        )
        self.assertEqual(
            result["required_evidence_status"], "CONFLICT_CANDIDATE"
        )

    def test_negated_directional_cues_do_not_become_direct_evidence(self):
        caption = "The people move at a leisurely pace."
        language = attach_claim_contract(
            {
                "caption_proposition": caption,
                "claim_subject": "people",
                "relation_family": "pace",
                "expected_visual_state": "people walking slowly",
                "opposite_visual_state": "people running quickly",
            },
            caption,
        )
        language["claim_relation"] = build_claim_relation(caption, language)
        result = compare(
            {
                "visual_facts": ["The people are not running quickly."],
                "schema_complete": True,
            },
            language,
            caption=caption,
        )
        self.assertFalse(result["contradicting_evidence"])

    def test_structured_relation_abstains_without_known_relation(self):
        relation = build_claim_relation(
            "Justice is an ongoing process.",
            {"caption_proposition": "Justice is an ongoing process."},
        )
        self.assertFalse(relation["resolved"])
        self.assertEqual(nominate_visual_relations(
            {"visual_facts": ["A statue holds a scale."]}, relation
        ), [])

    def test_relation_family_normalization_does_not_bypass_invalid_pair(self):
        caption = (
            "Products one dislikes disappear quickly, while products one "
            "loves last a long time."
        )
        language = attach_claim_contract(
            {
                "caption_proposition": caption,
                "claim_subject": "products",
                "relation_family": "usage, consumption, preference",
                "expected_visual_state": "disliked products disappear quickly",
                "opposite_visual_state": "loved products disappear quickly",
            },
            caption,
        )
        relation = build_claim_relation(caption, language)
        self.assertEqual(relation["relation_family"], "pace")
        self.assertFalse(relation["resolved"])


class PureDecisionScoringTests(unittest.TestCase):
    def test_position_bias_cancels(self):
        result = position_balanced_relation_scores(
            {"A": 2.0, "B": 0.0}, {"A": 2.0, "B": 0.0}
        )
        self.assertAlmostEqual(result["ENTAILS"], 0.5)
        self.assertAlmostEqual(result["CONTRADICTS"], 0.5)

    def test_evidence_quality_shrinks_probability(self):
        self.assertAlmostEqual(evidence_adjusted_confidence(0.9, 0.25), 0.6)


class EvidenceGradeTests(unittest.TestCase):
    @staticmethod
    def _pace_relation():
        caption = "The people move at a leisurely pace."
        language = attach_claim_contract(
            {
                "caption_proposition": caption,
                "claim_subject": "people",
                "relation_family": "pace",
                "expected_visual_state": "people walking slowly",
                "opposite_visual_state": (
                    "people running and rushing quickly"
                ),
            },
            caption,
        )
        return build_claim_relation(caption, language)

    def test_single_model_reinspection_is_only_a_candidate(self):
        ledger = add_visual_reinspection_evidence(
            [],
            {
                "recommendation": "CONTRADICTS",
                "specific_evidence": True,
                "reason": "The people are running and rushing down the street.",
                "observed_entity": "people",
                "observed_state": "running and rushing down the street",
                "image_region": "street",
                "claim_relation": "CONFLICT",
            },
            {"claim_relation": self._pace_relation()},
        )
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["id"], "DV001")
        self.assertFalse(ledger[0]["decision_grade"])
        self.assertEqual(ledger[0]["evidence_level"], "RELATION_CANDIDATE")
        audit = audit_decision(
            {
                "label": "CONTRADICTS",
                "_model_cited_evidence_ids": ["DV001"],
            },
            ledger,
        )
        self.assertFalse(audit["valid"])

    def test_weak_or_negated_reinspection_is_not_promoted(self):
        comparison = {"claim_relation": self._pace_relation()}
        weak = add_visual_reinspection_evidence(
            [],
            {
                "recommendation": "CONTRADICTS",
                "specific_evidence": True,
                "reason": "The people are running.",
            },
            comparison,
        )
        negated = add_visual_reinspection_evidence(
            [],
            {
                "recommendation": "CONTRADICTS",
                "specific_evidence": True,
                "reason": "The people are not running and not rushing.",
            },
            comparison,
        )
        self.assertEqual(weak, [])
        self.assertEqual(negated, [])

    def test_grounded_source_is_not_automatically_decision_grade(self):
        decision = {
            "label": "ENTAILS",
            "_model_cited_evidence_ids": ["VF001"],
        }
        audit = audit_decision(
            decision,
            [{
                "id": "VF001",
                "source": "agent1",
                "type": "visual_fact",
                "text": "People are visible.",
                "relation": "SUPPORT",
                "grounded": True,
                "decision_grade": False,
            }],
        )
        self.assertTrue(audit["source_valid"])
        self.assertFalse(audit["valid"])

    def test_explicit_verified_relation_is_decision_grade(self):
        decision = {
            "label": "CONTRADICTS",
            "_model_cited_evidence_ids": ["CC001"],
        }
        audit = audit_decision(
            decision,
            [{
                "id": "CC001",
                "source": "comparator",
                "type": "direct_conflict",
                "text": "People are running rather than walking slowly.",
                "relation": "CONFLICT",
                "grounded": True,
                "decision_grade": True,
            }],
        )
        self.assertTrue(audit["valid"])

    def test_diagnostic_nli_evidence_cannot_change_a_label(self):
        ledger = [{
            "id": "TV001",
            "source": "targeted_region_verifier",
            "type": "nli_region_relation_candidate",
            "text": "The model proposed a conflicting relation.",
            "relation": "CONFLICT",
            "grounded": True,
            "decision_grade": False,
        }]
        accepted, reason, _ = review_revision(
            {"label": "ENTAILS"},
            {
                "label": "CONTRADICTS",
                "_model_cited_evidence_ids": ["TV001"],
            },
            ledger,
            visual_review={
                "recommendation": "CONTRADICTS",
                "specific_evidence": True,
                "reason": "The paired text proposes a conflicting relation.",
            },
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "revision_lacks_decision_grade_direction")

    def test_decision_grade_revision_is_accepted(self):
        ledger = [{
            "id": "TV001",
            "source": "targeted_region_verifier",
            "type": "verified_region_relation",
            "text": "The loved product is gone in a week.",
            "relation": "CONFLICT",
            "grounded": True,
            "decision_grade": True,
        }]
        accepted, reason, _ = review_revision(
            {"label": "ENTAILS"},
            {
                "label": "CONTRADICTS",
                "_model_cited_evidence_ids": ["TV001"],
            },
            ledger,
            visual_review={
                "recommendation": "CONTRADICTS",
                "specific_evidence": True,
                "reason": "The loved product is gone in a week.",
            },
        )
        self.assertTrue(accepted)
        self.assertIn("TV001", reason)

    def test_review_board_caps_unsupported_confidence_without_changing_label(self):
        decision = attach_final_review(
            {"label": "ENTAILS", "confidence": 0.9}, [], {}
        )
        self.assertEqual(decision["label"], "ENTAILS")
        self.assertEqual(decision["confidence"], 0.35)
        self.assertTrue(
            decision["_review_board"]["confidence_cap_applied"]
        )

    def test_review_board_keeps_directionally_grounded_confidence(self):
        ledger = [{
            "id": "CS001", "source": "comparator",
            "type": "direct_support", "text": "The relation is explicit.",
            "relation": "SUPPORT", "grounded": True,
            "decision_grade": True,
        }]
        decision = attach_final_review(
            {
                "label": "ENTAILS", "confidence": 0.9,
                "_model_cited_evidence_ids": ["CS001"],
            },
            ledger,
            {"safe_for_directional_reasoning": True},
        )
        self.assertEqual(decision["confidence"], 0.79)
        self.assertEqual(
            decision["_review_board"]["independent_evidence_strength"], 0.6
        )


class RegionVerifierTests(unittest.TestCase):
    @staticmethod
    def _relation():
        caption = (
            "The product I love lasts for months, while the product I dislike "
            "is gone in a week."
        )
        language = attach_claim_contract(
            {
                "caption_proposition": caption,
                "claim_subject": "product",
                "claim_target": "product",
                "relation_family": "outcome",
                "expected_visual_state": (
                    "product love lasts for months; product dislike gone in a week"
                ),
                "opposite_visual_state": (
                    "product love gone in a week; product dislike lasts for months"
                ),
            },
            caption,
        )
        return build_claim_relation(caption, language)

    def test_explicit_pair_binding_resolves_conflict(self):
        result = verify_region_pairs(
            [
                {
                    "side": "left", "object_text": "product dislike",
                    "outcome_text": "lasts for months",
                },
                {
                    "side": "right", "object_text": "product love",
                    "outcome_text": "gone in a week",
                },
            ],
            self._relation(),
        )
        self.assertTrue(result["decision_grade"])
        self.assertEqual(result["label"], "CONTRADICTS")

    def test_incomplete_pair_binding_abstains(self):
        result = verify_region_pairs(
            [{
                "side": "left", "object_text": "product",
                "outcome_text": "visible",
            }],
            self._relation(),
        )
        self.assertFalse(result["decision_grade"])

    def test_unverified_cross_vocabulary_binding_abstains(self):
        caption = (
            "Products one dislikes disappear quickly, while products one "
            "loves last a long time."
        )
        language = attach_claim_contract(
            {
                "caption_proposition": (
                    "People use disliked items quickly and loved items slowly."
                ),
                "claim_subject": "items or products",
                "claim_object": "disliked items and loved items",
                "claim_source": "Unspecified",
                "claim_target": "items or products",
                "relation_family": "usage, consumption, preference",
                "expected_visual_state": (
                    "disliked items are used more frequently than loved items"
                ),
                "opposite_visual_state": (
                    "loved items are used at the same rate as disliked items"
                ),
            },
            caption,
        )
        relation = build_claim_relation(caption, language)
        result = verify_region_pairs(
            [
                {
                    "side": "left", "object_text": "Product you hate",
                    "outcome_text": "lasts for months",
                },
                {
                    "side": "right", "object_text": "Product you love",
                    "outcome_text": "gone in a week",
                },
            ],
            relation,
        )
        self.assertFalse(result["decision_grade"])
        self.assertEqual(result["reason"], "claim_relation_unresolved")


class PureFeedbackAttributionTests(unittest.TestCase):
    def test_dataset_phenomenon_mismatch_is_not_blamed_on_caption_agent(self):
        failure = FeedbackLoop.classify_verified_error(
            {"figurative_type": "literal"},
            {"label": "CONTRADICTS"},
            "ENTAILS",
            phenomenon="metaphor",
            visual_output={"schema_complete": True},
            comparison={"required_evidence_status": "SEMANTIC_REVIEW_REQUIRED"},
        )
        self.assertEqual(failure, "cross_modal_reasoning_candidate")

    def test_procedural_memory_contains_no_gold_direction(self):
        caption = "The people move at a leisurely pace."
        language = attach_claim_contract(
            {
                "caption_proposition": caption,
                "claim_subject": "people",
                "figurative_type": "metaphor",
                "relation_family": "pace",
                "expected_visual_state": "people walking slowly",
                "opposite_visual_state": "people running quickly",
            },
            caption,
        )
        language["claim_relation"] = build_claim_relation(caption, language)
        context = {
            "language_output": language,
            "comparison": {
                "required_evidence_status": "SEMANTIC_REVIEW_REQUIRED"
            },
            "decision": {"label": "CONTRADICTS"},
            "evidence_ledger": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = FeedbackLoop(
                log_file=os.path.join(temp_dir, "feedback.json")
            )
            self.assertTrue(loop.add_verified_case(
                context,
                "ENTAILS",
                "missed_grounded_support",
                loop.CALIBRATED_RULES["missed_grounded_support"],
                {"sample_id": "dev_1"},
            ))
        item = loop.arbiter_memory[0]
        self.assertEqual(item["memory_type"], "procedural_case")
        self.assertNotIn("verified_relation", item)
        self.assertNotIn("initial_label", item["signature"])

    def test_procedural_memory_is_selective(self):
        caption = "The people move at a leisurely pace."
        language = attach_claim_contract(
            {
                "caption_proposition": caption,
                "claim_subject": "people",
                "figurative_type": "metaphor",
                "relation_family": "pace",
                "expected_visual_state": "people walking slowly",
                "opposite_visual_state": "people running quickly",
            },
            caption,
        )
        language["claim_relation"] = build_claim_relation(caption, language)
        context = {
            "language_output": language,
            "comparison": {
                "required_evidence_status": "SEMANTIC_REVIEW_REQUIRED"
            },
            "decision": {"label": "ENTAILS"},
            "evidence_ledger": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = FeedbackLoop(
                log_file=os.path.join(temp_dir, "feedback.json")
            )
            loop.add_verified_case(
                context,
                "CONTRADICTS",
                "missed_grounded_conflict",
                loop.CALIBRATED_RULES["missed_grounded_conflict"],
                {"sample_id": "dev_2"},
            )
            self.assertEqual(len(loop.matching_rules("arbiter", context)), 1)
            unrelated = {
                **context,
                "language_output": {
                    **language,
                    "caption_proposition": "A completely unrelated safety claim.",
                    "intended_meaning": "A completely unrelated safety claim.",
                },
            }
            self.assertEqual(
                loop.matching_rules("arbiter", unrelated), []
            )


if __name__ == "__main__":
    unittest.main()
