import unittest
from unittest.mock import patch

from engine.claim_contract import audit_claim_contract, audit_relation_pair
from engine.decision_trace import (
    append_decision_checkpoint,
    attach_decision_trace,
)
from engine.evidence_ledger import (
    add_cross_agent_verified_relation,
    add_tribunal_corroborated_relation,
    promote_verified_relation,
)
from engine.evidence_ledger import build_evidence_ledger
from comparators.evidence_comparator import compare
from engine.claim_contract import attach_claim_contract
from engine.relation_schema import attach_claim_relation
from engine.question_router import build_question_plan, compile_visual_question
from engine.reasoning_schema import (
    attach_reasoning_profile,
    mechanism_candidates,
)
from engine.review_board import decision_grade_strength
from engine.structured_evidence import build_structured_observations
from engine.review_board import review_revision
from engine.tribunal import (
    apply_tribunal_resolution,
    followup_plan,
    new_tribunal_session,
    record_tribunal_round,
)
from engine.batch_runner import StagewiseRunner
from engine.debate import DebateEngine
from engine.pre_hearing import build_pre_hearing_audit
from engine.relation_semantics import (
    is_missing_evidence_only,
    relation_from_evidence_status,
)
from utils.judge_parser import parse_tribunal_review_response
import json


class TypedClaimContractTests(unittest.TestCase):
    def test_missing_evidence_is_neutral_in_shared_semantics(self):
        self.assertTrue(is_missing_evidence_only("No evidence is visible."))
        self.assertFalse(
            is_missing_evidence_only("No upward line is shown, but the line falls.")
        )
        self.assertEqual(
            relation_from_evidence_status("INSUFFICIENT_VISUAL_EVIDENCE"),
            "NEUTRAL",
        )

    def test_pre_hearing_closes_grounded_high_confidence_case(self):
        audit = build_pre_hearing_audit(
            {
                "label": "ENTAILS", "confidence": 0.82,
                "_review_board": {"directionally_grounded": True},
            },
            {
                "claim_contract": {"safe_for_directional_reasoning": True},
                "required_evidence_status": "SUPPORTED",
                "relation_binding_required": False,
            },
            {"trigger": False, "signals": []},
        )
        self.assertFalse(audit["requires_live_hearing"])

    def test_pre_hearing_escalates_unresolved_binding_without_gold(self):
        audit = build_pre_hearing_audit(
            {"label": "ENTAILS", "confidence": 0.6},
            {
                "claim_contract": {"safe_for_directional_reasoning": True},
                "required_evidence_status": "SEMANTIC_REVIEW_REQUIRED",
                "relation_binding_required": True,
                "relation_binding_observed": False,
            },
            {"trigger": True, "signals": ["unresolved_text_relation_semantics"]},
        )
        self.assertTrue(audit["requires_live_hearing"])
        self.assertNotIn("phenomenon", audit)

    def test_explicit_ocr_region_binding_is_typed_without_direction(self):
        records = build_structured_observations({
            "visible_text": ["left bottle => BEFORE", "right bottle => AFTER"],
            "visual_relations": [], "visual_facts": [],
        })
        self.assertEqual(len(records), 2)
        self.assertTrue(all(
            item["record_type"] == "OCR_REGION_BINDING" for item in records
        ))
        self.assertTrue(all(item["binding_complete"] for item in records))
        self.assertTrue(all(not item["directional"] for item in records))

    def test_panel_event_is_typed_but_not_promoted(self):
        records = build_structured_observations({
            "visible_text": [],
            "visual_relations": [
                "In the first panel a man pushes a vase; in the second panel it falls."
            ],
            "visual_facts": [],
        })
        self.assertEqual(records[0]["record_type"], "PANEL_EVENT_OR_COMPARISON")
        self.assertFalse(records[0]["directional"])

    def test_heading_contamination_cannot_be_direction_safe(self):
        caption = "The new product is preferred."
        contract = audit_claim_contract(caption, {
            "caption_proposition": caption,
            "claim_subject": "the new product",
            "claim_predicate": "is preferred",
            "expected_visual_state": (
                "the product is chosen Opposite Visual State: the product is rejected"
            ),
            "opposite_visual_state": "the product is rejected",
        })
        self.assertFalse(contract["safe_for_directional_reasoning"])
        self.assertIn("expected_visual_state", contract["contaminated_fields"])

    def test_absence_only_opposite_is_not_direction_safe(self):
        caption = "The relationship is volatile."
        contract = audit_claim_contract(caption, {
            "caption_proposition": caption,
            "claim_subject": "the relationship",
            "claim_predicate": "is volatile",
            "expected_visual_state": "the couple appears tense and distressed",
            "opposite_visual_state": "no signs of volatility are visible",
        })
        self.assertFalse(contract["opposite_state_is_affirmative"])
        self.assertFalse(contract["safe_for_directional_reasoning"])

    def test_generic_predicate_is_rejected_when_generated(self):
        caption = "The meeting is calm."
        contract = audit_claim_contract(caption, {
            "caption_proposition": caption,
            "claim_subject": "the meeting",
            "claim_predicate": "took place",
            "expected_visual_state": "people are calm and relaxed",
            "opposite_visual_state": "people are tense and distressed",
        })
        self.assertFalse(contract["predicate_specific"])
        self.assertFalse(contract["safe_for_directional_reasoning"])

    def test_typed_subject_binds_short_opposing_states(self):
        audit = audit_relation_pair("rotten", "healthy", "the heart")
        self.assertTrue(audit["valid"])

    def test_general_emotional_opposition_is_valid(self):
        audit = audit_relation_pair(
            "sadness or disappointment", "happiness or elation", "person"
        )
        self.assertTrue(audit["valid"])
    def test_general_pace_opposition_is_valid(self):
        audit = audit_relation_pair(
            "people walking slowly",
            "people running quickly",
            "people",
        )
        self.assertTrue(audit["valid"])

    def test_nonempty_but_nonopposing_states_are_not_automatic(self):
        caption = "Offering free beer in this way is disrespectful."
        contract = audit_claim_contract(caption, {
            "caption_proposition": caption,
            "claim_subject": "offering free beer",
            "expected_visual_state": "women are wearing shirts",
            "opposite_visual_state": "the offer occurs without consent",
            "reasoning_requirement": "normative",
        })
        self.assertFalse(contract["relation_pair_valid"])
        self.assertFalse(
            contract["safe_for_automatic_directional_reasoning"]
        )
        self.assertTrue(contract["requires_normative_reasoning"])

    def test_background_claim_is_not_lexically_directional(self):
        caption = "The hero's bike is unnecessary."
        contract = audit_claim_contract(caption, {
            "caption_proposition": caption,
            "claim_subject": "hero's bike",
            "expected_visual_state": "the bike is unused",
            "opposite_visual_state": "the bike is used",
            "background_knowledge": "The hero can run extremely fast.",
        })
        self.assertTrue(contract["requires_background_knowledge"])
        self.assertFalse(
            contract["safe_for_automatic_directional_reasoning"]
        )


class DecisionTraceTests(unittest.TestCase):
    def test_checkpoints_preserve_raw_proposals(self):
        original = {"label": "ENTAILS", "confidence": 0.7}
        raw = {
            "label": "CONTRADICTS",
            "_unconstrained_proposed_label": "CONTRADICTS",
        }
        trace = append_decision_checkpoint([], "initial", original)
        trace = append_decision_checkpoint(trace, "raw_debate", raw)
        raw["label"] = "ENTAILS"
        attached = attach_decision_trace(original, trace)
        self.assertEqual(
            attached["_decision_trace"][1]["raw_proposed_label"],
            "CONTRADICTS",
        )
        self.assertEqual(attached["_decision_trace_schema"], "1.0")


class EvidenceLifecycleTests(unittest.TestCase):
    def test_shared_witness_root_is_not_double_counted(self):
        ledger = [
            {
                "id": "DW001", "source": "debate_visual_witness",
                "relation": "NEUTRAL", "grounded": True,
                "decision_grade": False, "derived_from_ids": [],
            },
            {
                "id": "AV001", "source": "cross_agent_relation_verifier",
                "relation": "SUPPORT", "grounded": True,
                "decision_grade": True, "reliability": 0.78,
                "derived_from_ids": ["DW001"],
            },
            {
                "id": "IV001", "source": "tribunal_relation_verifier",
                "relation": "SUPPORT", "grounded": True,
                "decision_grade": True, "reliability": 0.74,
                "derived_from_ids": ["DW001"],
            },
        ]
        self.assertEqual(decision_grade_strength(ledger, "SUPPORT"), 0.78)

    def test_normative_relation_requires_three_source_corroboration(self):
        ledger = [
            {
                "id": "DW001", "source": "debate_visual_witness",
                "type": "entity_bound_observation", "text": "Gendered offer text is visible.",
                "relation": "NEUTRAL", "grounded": True,
                "decision_grade": False, "question_id": "Q1",
                "lifecycle_status": "ACTIVE",
            },
            {
                "id": "LC001", "source": "agent2",
                "type": "caption_proposition", "text": "The offer is disrespectful.",
                "relation": "NEUTRAL", "grounded": False,
                "decision_grade": False,
            },
        ]
        promoted, metadata = add_tribunal_corroborated_relation(
            ledger,
            {
                "status": "RESOLVE", "relation": "SUPPORT",
                "confidence": 0.86, "_format_valid": True,
                "_valid_evidence_ids": ["DW001"],
            },
            {
                "safe_for_directional_reasoning": True,
                "safe_for_automatic_directional_reasoning": False,
                "requires_normative_reasoning": True,
            },
            {"question_id": "Q1", "claim_relation": "UNRESOLVED"},
            {
                "_format_valid": True, "requirements_valid": True,
                "stance": "ENDORSE",
            },
        )
        self.assertTrue(metadata["promoted"])
        self.assertTrue(promoted[-1]["decision_grade"])
        self.assertEqual(promoted[-1]["relation"], "SUPPORT")
        self.assertEqual(
            promoted[-1]["verification_method"],
            "tribunal_normative_corroboration",
        )

    def test_tribunal_cannot_promote_without_current_round_witness(self):
        promoted, metadata = add_tribunal_corroborated_relation(
            [],
            {
                "status": "RESOLVE", "relation": "CONFLICT",
                "confidence": 0.90, "_format_valid": True,
                "_valid_evidence_ids": [],
            },
            {"safe_for_directional_reasoning": True},
            {},
            {
                "_format_valid": True, "requirements_valid": True,
                "stance": "ENDORSE",
            },
        )
        self.assertFalse(metadata["promoted"])
        self.assertEqual(promoted, [])

    def test_tribunal_cannot_promote_without_caption_provenance(self):
        ledger = [{
            "id": "DW001", "source": "debate_visual_witness",
            "type": "entity_bound_observation", "text": "The line falls.",
            "relation": "NEUTRAL", "grounded": True,
            "decision_grade": False, "question_id": "Q1",
            "lifecycle_status": "ACTIVE",
        }]
        promoted, metadata = add_tribunal_corroborated_relation(
            ledger,
            {
                "status": "RESOLVE", "relation": "CONFLICT",
                "confidence": 0.90, "_format_valid": True,
                "_valid_evidence_ids": ["DW001"],
            },
            {"safe_for_directional_reasoning": True},
            {"question_id": "Q1", "claim_relation": "UNRESOLVED"},
            {
                "_format_valid": True, "requirements_valid": True,
                "stance": "ENDORSE",
            },
        )
        self.assertFalse(metadata["promoted"])
        self.assertEqual(
            metadata["reason"], "caption_proposition_not_recorded"
        )
        self.assertEqual(promoted, ledger)

    def test_lexical_sentiment_matches_never_become_decision_grade(self):
        caption = "The meeting was calm and relaxing."
        language = attach_claim_contract({
            "caption_proposition": caption,
            "claim_subject": "meeting",
            "claim_object": "table",
            "expected_visual_state": "people around the table are calm",
            "opposite_visual_state": "people around the table are tense",
            "relation_family": "sentiment",
        }, caption)
        language = attach_claim_relation(language, caption)
        comparison = compare({
            "visual_facts": ["People are seated around a table."],
            "visual_relations": ["A person is looking toward the table."],
            "schema_complete": True,
        }, language, caption)
        ledger = build_evidence_ledger({}, language, comparison)
        self.assertFalse(any(item["decision_grade"] for item in ledger))

    def test_only_approved_independent_method_promotes_relation(self):
        ledger = promote_verified_relation(
            [],
            source="targeted_region_verifier",
            text="The right endpoint is lower than the left endpoint.",
            relation="CONFLICT",
            method="deterministic_numeric_or_geometric_relation",
            derived_from_ids=["VF001"],
        )
        self.assertTrue(ledger[0]["decision_grade"])
        self.assertEqual(ledger[0]["evidence_level"], "VERIFIED_RELATION")
        self.assertEqual(ledger[0]["derived_from_ids"], ["VF001"])

    def test_unapproved_method_cannot_promote_relation(self):
        with self.assertRaises(ValueError):
            promote_verified_relation(
                [], source="comparator", text="word overlap",
                relation="SUPPORT", method="structured_lexical_nomination",
            )

    def test_validated_agent_exchange_promotes_recorded_visual_witness(self):
        ledger = [{
            "id": "DW001", "source": "debate_visual_witness",
            "type": "entity_bound_observation",
            "text": "The plotted line falls left to right.",
            "relation": "NEUTRAL", "grounded": True,
            "decision_grade": False, "question_id": "Q1",
        }]
        promoted, metadata = add_cross_agent_verified_relation(
            ledger,
            {
                "question_id": "Q1",
                "claim_relation": "CONFLICT",
                "response_status": "VALID_DIRECTIONAL_ANSWER",
                "specific_evidence": True,
                "observed_state": "The plotted line falls left to right.",
                "witness_contract": {
                    "answer_status": "OBSERVED",
                    "observation": "The plotted line falls left to right.",
                    "direction_assigned": True,
                    "relation_candidate": "CONFLICT",
                },
            },
            {
                "stance": "ENDORSE", "_format_valid": True,
                "requirements_valid": True,
                "conflict_requirement": "the plotted line falls",
            },
            {"safe_for_automatic_directional_reasoning": True},
        )
        self.assertTrue(metadata["promoted"])
        self.assertEqual(promoted[-1]["id"], "AV001")
        self.assertEqual(promoted[-1]["relation"], "CONFLICT")
        self.assertEqual(promoted[-1]["derived_from_ids"], ["DW001"])

    def test_agent1_relation_cannot_promote_without_agent2_validation(self):
        ledger = [{
            "id": "DW001", "source": "debate_visual_witness",
            "type": "entity_bound_observation", "text": "A line falls.",
            "relation": "NEUTRAL", "grounded": True,
            "decision_grade": False, "question_id": "Q1",
        }]
        promoted, metadata = add_cross_agent_verified_relation(
            ledger,
            {
                "question_id": "Q1", "claim_relation": "CONFLICT",
                "response_status": "VALID_DIRECTIONAL_ANSWER",
                "specific_evidence": True, "observed_state": "A line falls.",
                "witness_contract": {
                    "answer_status": "OBSERVED",
                    "observation": "A line falls.",
                    "direction_assigned": True,
                    "relation_candidate": "CONFLICT",
                },
            },
            {
                "stance": "ENDORSE", "_format_valid": True,
                "requirements_valid": False,
            },
            {"safe_for_automatic_directional_reasoning": True},
        )
        self.assertFalse(metadata["promoted"])
        self.assertEqual(metadata["reason"], "claim_requirements_invalid")
        self.assertEqual(promoted, ledger)


class QuestionRoutingTests(unittest.TestCase):
    def test_mechanism_candidates_do_not_use_dataset_phenomenon(self):
        output = attach_reasoning_profile({
            "figurative_type": "literal",
            "polarity_reversal": "yes, praise reverses to criticism",
            "phenomenon": "metaphor",
        })
        self.assertEqual(
            output["reasoning_profile"]["primary_figurative_mechanism"],
            "SARCASM_POLARITY",
        )
        self.assertFalse(output["reasoning_profile"]["uses_gold_phenomenon"])

    def test_temporal_causal_route_asks_for_ordered_actions(self):
        plan = build_question_plan({
            "claim_contract": {
                "safe_for_directional_reasoning": True,
                "structural_reasoning_type": "TEMPORAL_CAUSAL_SEQUENCE",
            },
            "claim_relation": {"subject": "the character"},
        })
        self.assertEqual(plan.issue_type, "TEMPORAL_CAUSAL_SEQUENCE")
        self.assertIn("panel", plan.agent1_question.casefold())
        self.assertIn("action", plan.agent1_question.casefold())

    def test_sarcasm_route_separates_referent_and_polarity(self):
        plan = build_question_plan({
            "claim_contract": {
                "safe_for_directional_reasoning": True,
                "structural_reasoning_type": "AFFECTIVE_SCENE",
            },
            "claim_relation": {
                "subject": "the design",
                "figurative_mechanism": "sarcasm",
            },
        })
        self.assertEqual(plan.issue_type, "SARCASM_POLARITY")
        self.assertIn("polarity", plan.agent2_question.casefold())

    def test_metaphor_route_keeps_source_observation_literal(self):
        plan = build_question_plan({
            "claim_contract": {
                "safe_for_directional_reasoning": True,
                "structural_reasoning_type": "DIRECT_STATE",
            },
            "claim_relation": {
                "subject": "introverts",
                "figurative_mechanism": "metaphor",
            },
        })
        self.assertEqual(plan.issue_type, "METAPHOR_MAPPING")
        self.assertIn("source entity", plan.agent1_question.casefold())

    def test_outcome_claim_asks_about_actions_and_results(self):
        plan = build_question_plan({
            "claim_contract": {"safe_for_directional_reasoning": True},
            "claim_relation": {
                "subject": "products",
                "relation_family": "pace",
                "expected_visual_state": "products disappear quickly",
                "opposite_visual_state": "products last a long time",
            },
            "required_evidence_status": "GROUNDED_REVIEW_REQUIRED",
        })
        self.assertEqual(plan.issue_type, "COMPARISON_OR_OUTCOME")
        self.assertIn("action", plan.agent1_question)
        self.assertIn("outcome", plan.agent1_question)

    def test_normative_claim_does_not_request_a_symbol(self):
        plan = build_question_plan({
            "claim_contract": {
                "safe_for_directional_reasoning": True,
                "requires_normative_reasoning": True,
            },
            "claim_relation": {"subject": "the depicted behavior"},
        })
        self.assertEqual(plan.issue_type, "NORMATIVE_REASONING")
        self.assertNotIn("symbol", plan.agent1_question.casefold())

    def test_invalid_mediator_question_uses_neutral_typed_fallback(self):
        question = compile_visual_question(
            {
                "claim_contract": {"safe_for_directional_reasoning": True},
                "claim_relation": {
                    "subject": "the plotted line",
                    "expected_visual_state": "the line rises",
                    "opposite_visual_state": "the line falls",
                },
            },
            {"agent1_questions": ["Does the picture support the verdict?"]},
        )
        self.assertNotIn("support", question.casefold())
        self.assertNotIn("verdict", question.casefold())
        self.assertIn("plotted line", question.casefold())


class TribunalControllerTests(unittest.TestCase):
    def test_rejected_revision_reports_exact_failed_invariant(self):
        accepted, reason, audit = review_revision(
            {"label": "ENTAILS", "confidence": 0.6},
            {"label": "CONTRADICTS", "confidence": 0.7},
            [],
            visual_review={
                "recommendation": "CONTRADICTS",
                "specific_evidence": True,
                "reason": "A claimed conflict was proposed.",
            },
        )
        self.assertFalse(accepted)
        self.assertEqual(
            reason, "revision_did_not_cite_current_image_evidence"
        )
        self.assertEqual(
            audit["failed_invariant"],
            "revision_did_not_cite_current_image_evidence",
        )
        self.assertTrue(audit["acceptance_checks"])

    def test_three_source_semantic_resolution_can_change_label(self):
        ledger = [
            {
                "id": "DW001", "source": "debate_visual_witness",
                "type": "entity_bound_observation", "text": "The plotted line falls.",
                "relation": "NEUTRAL", "grounded": True,
                "decision_grade": False, "question_id": "Q1",
                "lifecycle_status": "ACTIVE",
            },
            {
                "id": "LC001", "source": "agent2",
                "type": "caption_proposition", "text": "The plotted line rises.",
                "relation": "NEUTRAL", "grounded": False,
                "decision_grade": False,
            },
        ]
        review = {
            "status": "RESOLVE", "relation": "CONFLICT",
            "provisional_verdict": "CONTRADICTS", "confidence": 0.84,
            "visual_observations": ["The plotted line falls."],
            "reason": "The observed direction conflicts with the caption condition.",
            "_format_valid": True, "_valid_evidence_ids": ["DW001"],
            "_invalid_evidence_ids": [],
        }
        decision, output_ledger, metadata = apply_tribunal_resolution(
            {"label": "ENTAILS", "confidence": 0.35},
            review,
            ledger,
            {
                "safe_for_directional_reasoning": True,
                "safe_for_automatic_directional_reasoning": False,
                "relation_pair_valid": True,
                "requires_background_knowledge": True,
            },
            agent2_requirements_valid=True,
            agent1_critique={
                "question_id": "Q1", "claim_relation": "UNRESOLVED"
            },
            agent2_critique={
                "_format_valid": True, "requirements_valid": True,
                "stance": "ENDORSE",
            },
        )
        self.assertTrue(metadata["accepted"])
        self.assertTrue(metadata["corroboration"]["promoted"])
        self.assertEqual(decision["label"], "CONTRADICTS")
        self.assertTrue(any(
            item.get("source") == "tribunal_relation_verifier"
            and item.get("decision_grade")
            for item in output_ledger
        ))

    def test_relation_first_contract_maps_direction_deterministically(self):
        review = parse_tribunal_review_response(json.dumps({
            "status": "RESOLVE", "relation": "CONFLICT",
            "confidence": 0.84, "evidence_ids": ["DW001"],
            "visual_observations": ["The line falls."],
            "issue": "Trend direction", "agent1_question": "",
            "agent2_question": "", "verification_request": "",
            "reason": "The observed direction opposes the caption condition.",
        }))
        self.assertTrue(review["_format_valid"])
        self.assertEqual(review["provisional_verdict"], "CONTRADICTS")

    def test_issue_text_can_fill_duplicate_reason_field(self):
        review = parse_tribunal_review_response(json.dumps({
            "status": "RESOLVE", "relation": "SUPPORT",
            "confidence": 0.82, "evidence_ids": ["DW001"],
            "visual_observations": ["The line rises."],
            "issue": "The visible rise matches the caption.",
            "agent1_question": "", "agent2_question": "",
            "verification_request": "",
        }))
        self.assertTrue(review["_format_valid"])
        self.assertEqual(review["reason"], review["issue"])
    def test_mediator_cannot_self_certify_neutral_observation(self):
        ledger = [{
            "id": "DW001", "source": "debate_visual_witness",
            "type": "entity_bound_observation",
            "text": "The plotted line appears to fall.",
            "relation": "NEUTRAL", "grounded": True,
            "decision_grade": False,
        }]
        review = {
            "status": "RESOLVE", "provisional_verdict": "CONTRADICTS",
            "confidence": 0.90,
            "visual_observations": ["The plotted line appears to fall."],
            "reason": "The visible endpoint appears lower.",
            "_format_valid": True, "_valid_evidence_ids": ["DW001"],
            "_invalid_evidence_ids": [],
        }
        decision, unchanged, metadata = apply_tribunal_resolution(
            {"label": "ENTAILS", "confidence": 0.8},
            review,
            ledger,
            {
                "safe_for_directional_reasoning": True,
                "relation_pair_valid": True,
            },
        )
        self.assertFalse(metadata["accepted"])
        self.assertEqual(
            metadata["reason"],
            "tribunal_citations_lack_independent_direction",
        )
        self.assertEqual(decision["label"], "ENTAILS")
        self.assertEqual(unchanged, ledger)

    def test_followup_question_cannot_expose_a_dataset_label(self):
        review = parse_tribunal_review_response(json.dumps({
            "status": "FOLLOW_UP",
            "provisional_verdict": "ABSTAIN",
            "confidence": 0.50,
            "evidence_ids": [],
            "visual_observations": [],
            "issue": "The visible state is unclear.",
            "agent1_question": "Does the image contradict the caption?",
            "agent2_question": "",
            "verification_request": "Inspect the subject.",
            "reason": "A neutral observation is still required.",
        }))
        self.assertFalse(review["_format_valid"])
        self.assertEqual(
            review["_format_error"], "tribunal_question_exposes_label"
        )

    def test_parser_accepts_exact_resolved_contract(self):
        review = parse_tribunal_review_response(json.dumps({
            "status": "RESOLVE",
            "provisional_verdict": "CONTRADICTS",
            "confidence": 0.84,
            "evidence_ids": ["dw001"],
            "visual_observations": ["The plotted line falls."],
            "issue": "The trend direction was disputed.",
            "agent1_question": "",
            "agent2_question": "",
            "verification_request": "",
            "reason": "The visible fall conflicts with the claimed rise.",
        }))
        self.assertTrue(review["_format_valid"])
        self.assertEqual(review["evidence_ids"], ["DW001"])

    def test_followup_is_bounded(self):
        review = {
            "status": "FOLLOW_UP", "provisional_verdict": "ABSTAIN",
            "confidence": 0.5, "agent1_questions": ["Does the line fall?"],
            "agent2_questions": [], "verification_requests": [],
            "issue": "Direction", "_format_valid": True,
            "_invalid_evidence_ids": [], "_valid_evidence_ids": [],
        }
        session = record_tribunal_round(new_tribunal_session({}), review, {})
        self.assertEqual(session["state"], "FOLLOW_UP_REQUIRED")
        self.assertTrue(followup_plan(review)["_usable"])
        session = record_tribunal_round(session, review, {})
        self.assertEqual(session["state"], "ABSTAINED")
        self.assertEqual(session["stop_reason"], "maximum_rounds_reached")

    def test_independent_resolution_uses_review_board(self):
        ledger = [
            {
                "id": "DW001", "source": "debate_visual_witness",
                "type": "entity_bound_observation",
                "text": "The plotted line falls from left to right.",
                "relation": "NEUTRAL", "grounded": True,
                "decision_grade": False, "evidence_level": "BINDING",
            },
            {
                "id": "AV001", "source": "cross_agent_relation_verifier",
                "type": "verified_relation",
                "text": "The observed fall conflicts with the claimed rise.",
                "relation": "CONFLICT", "grounded": True,
                "decision_grade": True,
                "evidence_level": "VERIFIED_RELATION",
                "verification_method": "cross_agent_structured_relation",
                "reliability": 0.78,
                "derived_from_ids": ["DW001"],
            },
        ]
        review = {
            "status": "RESOLVE", "provisional_verdict": "CONTRADICTS",
            "confidence": 0.84, "visual_observations": [
                "The plotted line falls from left to right."
            ],
            "reason": "The falling line conflicts with the claimed rise.",
            "_format_valid": True,
            "_valid_evidence_ids": ["DW001", "AV001"],
            "_invalid_evidence_ids": [],
        }
        decision, verified, metadata = apply_tribunal_resolution(
            {"label": "ENTAILS", "confidence": 0.4},
            review,
            ledger,
            {
                "safe_for_directional_reasoning": True,
                "relation_pair_valid": True,
            },
            agent2_requirements_valid=True,
        )
        self.assertTrue(metadata["accepted"])
        self.assertEqual(decision["label"], "CONTRADICTS")
        self.assertTrue(verified[-1]["decision_grade"])

    def test_resolution_cannot_overwrite_stronger_opposing_evidence(self):
        ledger = [
            {
                "id": "TV001", "source": "targeted_region_verifier",
                "type": "verified_relation", "text": "The line rises.",
                "relation": "SUPPORT", "grounded": True,
                "decision_grade": True, "evidence_level": "VERIFIED_RELATION",
                "verification_method": "deterministic_numeric_or_geometric_relation",
                "reliability": 1.0,
            },
            {
                "id": "DW001", "source": "debate_visual_witness",
                "type": "entity_bound_observation",
                "text": "The line appears to fall.", "relation": "NEUTRAL",
                "grounded": True, "decision_grade": False,
                "evidence_level": "BINDING",
            },
            {
                "id": "AV001", "source": "cross_agent_relation_verifier",
                "type": "verified_relation",
                "text": "The observed fall conflicts with the claimed rise.",
                "relation": "CONFLICT", "grounded": True,
                "decision_grade": True,
                "evidence_level": "VERIFIED_RELATION",
                "verification_method": "cross_agent_structured_relation",
                "reliability": 0.78,
                "derived_from_ids": ["DW001"],
            },
        ]
        review = {
            "status": "RESOLVE", "provisional_verdict": "CONTRADICTS",
            "confidence": 0.90,
            "visual_observations": ["The line appears to fall."],
            "reason": "The visible endpoint appears lower.",
            "_format_valid": True,
            "_valid_evidence_ids": ["DW001", "AV001"],
            "_invalid_evidence_ids": [],
        }
        decision, verified, metadata = apply_tribunal_resolution(
            {
                "label": "ENTAILS", "confidence": 0.8,
                "_model_cited_evidence_ids": ["TV001"],
            },
            review,
            ledger,
            {
                "safe_for_directional_reasoning": True,
                "relation_pair_valid": True,
            },
        )
        self.assertFalse(metadata["accepted"])
        self.assertEqual(decision["label"], "ENTAILS")
        self.assertEqual(
            metadata["reason"], "unresolved_opposing_decision_grade_evidence"
        )
        self.assertEqual(
            decision["_decision_trace"][-1]["stage"],
            "tribunal_resolution_preserved",
        )
        self.assertTrue(verified[-1]["decision_grade"])


class TribunalBatchIntegrationTests(unittest.TestCase):
    def test_tribunal_hearing_records_witnesses_without_calling_arbiter(self):
        class FakeVisualAgent:
            def __init__(self, _runtime):
                pass

            @staticmethod
            def critique(_image, prompt):
                self.assertIn("TRIBUNAL_VISUAL_WITNESS_ONLY", prompt)
                return {
                    "_format_valid": True,
                    "response_status": "VALID_OBSERVATION",
                    "observed_entity": "line", "observed_state": "line falls",
                    "image_region": "chart", "specific_evidence": True,
                    "recommendation": "ABSTAIN", "claim_relation": "UNRESOLVED",
                    "question_id": "Q1", "reason": "The line falls.",
                    "witness_contract": {
                        "question_id": "Q1", "answer_status": "OBSERVED",
                        "observation": "line falls", "region": "chart",
                        "direction_assigned": False,
                    },
                }

        class FakeAgent2:
            @staticmethod
            def critique(_caption, _prompt):
                return {
                    "stance": "ENDORSE", "_format_valid": True,
                    "requirements_valid": True,
                    "support_requirement": "line rises",
                    "conflict_requirement": "line falls",
                    "reason": "The caption asserts a rising line.",
                }

        class ForbiddenArbiter:
            @staticmethod
            def analyze(*_args, **_kwargs):
                raise AssertionError("tribunal hearing must not call the old arbiter")

        contract = {
            "safe_for_directional_reasoning": True,
            "safe_for_automatic_directional_reasoning": True,
        }
        case = {
            "key": "case", "caption": "The line rises.", "image": object(),
            "visual_output": {}, "language_output": {"claim_contract": contract},
            "comparison": {"claim_contract": contract, "claim_relation": {}},
            "decision": {"label": "ENTAILS", "confidence": 0.55},
            "evidence_ledger": [], "debate_level": 2,
            "tribunal_hearing": True, "force_visual_review": True,
            "mediation_plan": {
                "_usable": True,
                "agent1_questions": ["What direction does the line move?"],
            },
        }
        engine = DebateEngine()
        with (
            patch("engine.debate.Qwen3VLVisionModel", return_value=object()),
            patch("engine.debate.VisualGroundingAgent", FakeVisualAgent),
            patch("engine.debate.GPUManager.clear"),
        ):
            result = engine.run_debate_batch(
                [case],
                language_runtime={
                    "agent2": FakeAgent2(), "arbiter": ForbiddenArbiter(),
                },
            )["case"]
        self.assertEqual(result["label"], "ENTAILS")
        self.assertEqual(
            result["_debate"]["architecture"],
            "tribunal_targeted_hearing",
        )
        self.assertFalse(result["_debate"]["revision_accepted"])

    def test_one_review_round_resolves_and_records_state(self):
        class FakeMediator:
            def __init__(self, _runtime):
                pass

            def review(self, *_args, **_kwargs):
                return {
                    "status": "RESOLVE",
                    "provisional_verdict": "CONTRADICTS",
                    "confidence": 0.84,
                    "evidence_ids": ["DW001", "AV001"],
                    "visual_observations": ["The plotted line falls."],
                    "issue": "Trend direction",
                    "agent1_questions": [], "agent2_questions": [],
                    "verification_requests": [],
                    "reason": "The visible fall conflicts with the claimed rise.",
                    "_format_valid": True,
                    "_valid_evidence_ids": ["DW001", "AV001"],
                    "_invalid_evidence_ids": [],
                    "_generation_seconds": 0.01,
                }

        runner = StagewiseRunner.__new__(StagewiseRunner)
        sample = {"index": 0, "image": object(), "caption": "The line rises."}
        results = {0: {
            "visual_output": {}, "language_output": {"claim_contract": {
                "safe_for_directional_reasoning": True,
                "relation_pair_valid": True,
            }},
            "comparison": {},
            "evidence_ledger": [
                {
                    "id": "DW001", "source": "debate_visual_witness",
                    "type": "entity_bound_observation",
                    "text": "The plotted line falls.", "relation": "NEUTRAL",
                    "grounded": True, "decision_grade": False,
                    "evidence_level": "BINDING",
                },
                {
                    "id": "AV001", "source": "cross_agent_relation_verifier",
                    "type": "verified_relation",
                    "text": "The observed fall conflicts with the claimed rise.",
                    "relation": "CONFLICT", "grounded": True,
                    "decision_grade": True,
                    "evidence_level": "VERIFIED_RELATION",
                    "verification_method": "cross_agent_structured_relation",
                    "reliability": 0.78,
                    "derived_from_ids": ["DW001"],
                },
            ],
            "decision": {"label": "ENTAILS", "confidence": 0.4},
            "debate_details": {"agent2_requirements_valid": True},
            "timing": {"mediator_seconds": 0.0},
            "judge": {"mediation": {"status": "MEDIATE"}},
        }}
        with (
            patch("engine.batch_runner.QwenJudgeModel", return_value=object()),
            patch("engine.batch_runner.TribunalMediatorAgent", FakeMediator),
            patch("engine.batch_runner.GPUManager.clear"),
        ):
            followups, _ = runner._run_tribunal_review_round(
                [sample], results, 1
            )
        self.assertEqual(followups, [])
        self.assertEqual(results[0]["decision"]["label"], "CONTRADICTS")
        self.assertEqual(results[0]["judge"]["status"], "tribunal_revision_accepted")
        self.assertEqual(results[0]["judge"]["tribunal_session"]["state"], "RESOLVED")


class PublicRunnerIntegrationTests(unittest.TestCase):
    def test_public_orchestrator_delegates_to_canonical_stagewise_runner(self):
        from engine.orchestrator import Orchestrator

        class FakeRunner:
            def __init__(self, **settings):
                self.settings = settings
                self.debate = object()

            def run_samples(self, samples, callback):
                self.samples = samples
                callback(0, samples[0]["raw"], {"decision": {
                    "label": "ENTAILS"
                }}, 0.01)

        with patch("engine.orchestrator.StagewiseRunner", FakeRunner):
            orchestrator = Orchestrator(judge_mode="tribunal")
            output = orchestrator.run_sample({
                "image": object(), "caption": "A caption"
            })
        self.assertEqual(output["decision"]["label"], "ENTAILS")
        self.assertEqual(orchestrator.runner.settings["judge_mode"], "tribunal")
        self.assertEqual(
            orchestrator.runner.samples[0]["raw"]["id"], "api_sample"
        )


if __name__ == "__main__":
    unittest.main()
