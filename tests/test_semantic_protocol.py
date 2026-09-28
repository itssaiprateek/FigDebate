"""Exercise real proposal, verification and acceptance paths with scripted witnesses.

These tests establish contracts and provenance, not model semantic accuracy.
"""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

from agents.multimodal_judge import TribunalMediatorAgent
from engine import semantic_protocol as protocol
from engine.evidence_review_v5 import audit
from engine.semantic_bridge import build_semantic_bridge
from engine.tribunal import apply_tribunal_resolution
from engine.tribunal_repair import plan_repair
from engine.runtime_accounting import begin_accounting
from tests.test_evidence_review_v4 import Runtime
from tests.test_tribunal_v5 import v5_fixture
from tests.test_semantic_bridge import contract


class SemanticProtocolTests(unittest.TestCase):
    def setUp(self):
        begin_accounting()

    def run_case(self, relation="CONFLICT", objections=None, alignment="MATCH"):
        image, p, ledger, answers = v5_fixture()
        reading = dict(kind="LITERAL", referent="meeting participants", property="calmness")
        proposal = dict(interpreted_assertion=p["source_caption"],
                        decisive_reason=p["bridge_statement"], reading=reading,
                        evidence_ids=["VF1"], relation=relation)
        answers[2]["relation"] = answers[2]["condition_checks"][0]["relation"] = relation
        answers[2]["reading"] = deepcopy(reading)
        answers[3] = dict(alignment=alignment, objections=objections or [],
                          evidence_ids=["VF1"], reason="The same meeting and property are compared using the visible expressions.")
        runtime = Runtime([proposal] + answers)
        runtime.hardware_profile = SimpleNamespace(tribunal_protocol="evidence-review-5.0")
        runtime.tribunal_audit_mode = protocol.AUDIT_VERSION
        language = {"claim_contract": dict(contract(), source_caption=p["source_caption"],
                                            caption_proposition=p["source_caption"])}
        ledger.append(dict(id="LC1", source="agent2", type="caption_proposition",
                           text=p["source_caption"], grounded=False))
        review = TribunalMediatorAgent(runtime).review(image, p["source_caption"], {}, language,
                    {}, ledger, {}, current_decision={"label": "SECRET_INITIAL_LABEL"})
        bridge = build_semantic_bridge(review, ledger, language["claim_contract"], language)
        return runtime, review, bridge, ledger, language, p["source_caption"]

    def test_real_gate_both_directions_without_extra_call(self):
        for relation, expected, initial in [("SUPPORT","ENTAILS","CONTRADICTS"),("CONFLICT","CONTRADICTS","ENTAILS")]:
            rt, review, bridge, ledger, language, source = self.run_case(relation)
            self.assertEqual(len(rt.prompts),5)
            self.assertTrue(audit(bridge,ledger)["valid"])
            final, _, gate = apply_tribunal_resolution({"label":initial,"confidence":.3}, review, ledger,
                        language["claim_contract"], semantic_bridge_mode="corroborated", source_caption=source)
            self.assertTrue(gate["accepted"])
            self.assertEqual(final["label"], expected)
            for prompt in rt.prompts:
                self.assertNotIn("SECRET_INITIAL_LABEL",prompt)
            for prompt in rt.prompts[1:4]:
                payload=json.loads(prompt.rsplit("\n",1)[-1])
                self.assertNotIn("candidate",payload)
                self.assertNotIn("draft_argument",payload)

    def test_verified_explanation_does_not_inherit_initial_generation_failure(self):
        from engine.final_artifact import final_artifact
        _, review, _, ledger, language, source = self.run_case()
        for initial_label in ("ENTAILS", "CONTRADICTS"):
            initial = dict(label=initial_label, confidence=.3, explanation="Unfinished old assessment because",
                           _explanation_generation_failed=True, _binary_label_is_unverified=True)
            final, verified, gate = apply_tribunal_resolution(initial, review, ledger,
                    language["claim_contract"], semantic_bridge_mode="corroborated", source_caption=source)
            self.assertTrue(gate.get("accepted") or gate.get("confirmation_valid"))
            self.assertEqual(final["explanation"], review["reason"])
            self.assertFalse(final["_explanation_generation_failed"])
            self.assertFalse(final["_binary_label_is_unverified"])
            self.assertTrue(final_artifact(source, {"decision": final, "evidence_ledger": verified})
                            ["explanation_completeness"]["complete"])
            self.assertTrue(initial["_explanation_generation_failed"])

    def test_cli_preserves_legacy_and_explicit_baseline_controls(self):
        import sys
        from unittest.mock import patch
        from run_figdebate import parse_args
        for argv, expected in [([], "completion"), (["--reasoning-mode", "baseline"], "baseline"),
                               (["--execution-mode", "sequential"], "baseline")]:
            with patch.object(sys, "argv", ["run_figdebate.py"] + argv):
                self.assertEqual(parse_args().reasoning_mode, expected)

    def test_real_objection_blocks_matching_labels(self):
        objection=dict(type="UNSUPPORTED_INFERENCE",disputed_step="The meeting is calm.",
                       basis="The observed participants look alarmed.",evidence_ids=["VF1"])
        _,review,bridge,ledger,_,_=self.run_case(objections=[objection])
        self.assertFalse(audit(bridge,ledger)["valid"])
        self.assertTrue(review["_independent_verification"]["obligations"]["arguments"]["decision_errors"])

    def test_scope_mismatch_blocks_acceptance(self):
        objection=dict(type="SCOPE_ERROR",disputed_step="Claim refers to another meeting.",
                       basis="The evidence depicts the current meeting.",evidence_ids=["VF1"])
        _,_,bridge,ledger,_,_=self.run_case(objections=[objection],alignment="MISMATCH")
        self.assertFalse(audit(bridge,ledger)["valid"])

    def test_cannot_erase_or_mutate_projected_errors(self):
        _,_,bridge,ledger,_,_=self.run_case()
        call=bridge["independent_verification"]["obligations"]["arguments"]
        call["decision_errors"]=["An invented objection."]
        self.assertFalse(audit(bridge,ledger)["valid"])
        call["decision_errors"]=[]
        call["_focused_value"]["reason"]="Another explanation."
        self.assertFalse(audit(bridge,ledger)["valid"])

    def test_cannot_change_reading_without_invalidating_proof(self):
        _,_,bridge,ledger,_,_=self.run_case()
        bridge["reading"]["property"]="wealth"
        self.assertFalse(audit(bridge,ledger)["valid"])

    def test_removing_focused_requirement_cannot_downgrade_proof(self):
        _,_,bridge,ledger,_,_=self.run_case()
        bridge.pop("focused_audit_required")
        self.assertFalse(audit(bridge,ledger)["valid"])

    def test_missing_audit_or_truncated_response_never_passes(self):
        _,_,bridge,ledger,_,_=self.run_case()
        call=bridge["independent_verification"]["obligations"]["arguments"]
        call["_raw_output"]='{"alignment":"MATCH"'
        self.assertFalse(audit(bridge,ledger)["valid"])

    def test_ambiguous_reading_is_not_an_error_free_pass(self):
        objection=dict(type="AMBIGUOUS_READING",disputed_step="The expressions could concern a different event.",
                       basis="The event attachment is unclear.",evidence_ids=["VF1"])
        _,_,bridge,ledger,_,_=self.run_case(objections=[objection])
        self.assertFalse(audit(bridge,ledger)["valid"])

    def test_repeat_dispute_stops_even_if_question_is_rephrased(self):
        _,review,_,_,_,_=self.run_case()
        call=review["_independent_verification"]["obligations"]["arguments"]
        # Simulate a fully delivered but rejecting audit; this is a repair routing fixture.
        call["decision_errors"]=["The evidence does not establish the same participant."]
        history={}
        first=plan_repair(review,history)
        self.assertTrue(first)
        self.assertTrue(first["repair_context"]["dispute_id"])
        self.assertFalse(plan_repair(review,history))
        self.assertEqual(review["_follow_up_question_status"],"NO_NEW_AVAILABLE_CHECK")

    def test_complete_inconsistent_audit_gets_one_source_recheck_not_acceptance(self):
        _, review, _, _, _, _ = self.run_case()
        call = review["_independent_verification"]["obligations"]["arguments"]
        value = dict(alignment="MISMATCH", objections=[], evidence_ids=["VF1"],
                     reason="The assessed property differs between the candidate and verifier.")
        call.update(value, _raw_output=json.dumps(value), _format_valid=False,
                    _contract_error_kind="AUDIT_INCOMPLETE")
        protocol.project_audit(call)
        self.assertTrue(protocol.audit_wire_complete(call, {"VF1"}))
        self.assertFalse(protocol.audit_accepts(call, {"VF1"}))
        history = {}
        plan = plan_repair(review, history)
        self.assertEqual(plan["repair_reasons"], ["contradictory_audit"])
        self.assertFalse(plan_repair(review, history))
        call["_raw_output"] = '{"alignment":'
        self.assertFalse(plan_repair(review, {}))

    def test_reasoning_mode_cannot_reuse_an_old_initial_decision(self):
        from pathlib import Path
        from engine.cache_identity import stage_fingerprints
        root = Path(__file__).resolve().parents[1]
        config = dict(seed=42, pipeline_source_sha256="fixed", reasoning_mode="baseline")
        old = stage_fingerprints(root, config)
        for mode in ("completion", "grounded"):
            new = stage_fingerprints(root, dict(config, reasoning_mode=mode))
            self.assertNotEqual(old["initial_reasoning"], new["initial_reasoning"])
            self.assertEqual(old["visual_grounding"], new["visual_grounding"])

    def test_reading_modes_are_supported_by_official_runner(self):
        from engine.tribunal_repair import validate_options
        for mode in (protocol.ALIGNMENT_VERSION,protocol.AUDIT_VERSION):
            validate_options("bounded",mode,"tribunal","enabled","evidence-review-5.0","disabled")
            with self.assertRaises(ValueError):
                validate_options("bounded",mode,"tribunal","enabled","evidence-review-5.0","integrated")

    def test_reading_is_defined_in_prompt_and_precedes_judgment(self):
        from engine.simple_judge import prompt, schema
        text = prompt({"semantic_alignment_protocol": protocol.ALIGNMENT_VERSION})
        self.assertIn("kind is LITERAL", text)
        self.assertIn("UNCERTAIN requires relation UNRESOLVED", text)
        self.assertEqual(next(iter(schema({"VF1"}, aligned=True)["properties"])), "reading")
        from engine.evidence_review_v5 import DECISION
        self.assertEqual(next(iter(protocol.reading_schema(DECISION)["properties"])), "reading")

    def test_uncertainty_cannot_have_a_direction(self):
        self.assertTrue(protocol.reading_error(dict(reading=dict(kind="UNCERTAIN",referent="unknown",property="care"),
                                                    relation="SUPPORT")))


if __name__ == "__main__":
    unittest.main()
