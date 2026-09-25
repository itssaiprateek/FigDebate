"""General capability, source-identity and bounded-recovery regressions."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engine.source_spans import SourceSpans
from engine.review_routing import route_question
from engine.case_budget import case_budget, proposal_reserve, budget_estimates, observe_cost
from engine.evidence_verification import verify, audit
from tests.test_evidence_review_v4 import fixture, Runtime


class ReliabilityRevisionTests(unittest.TestCase):
    def setUp(self):
        from engine.runtime_accounting import begin_accounting
        begin_accounting()

    def test_routing_capabilities_across_wording(self):
        examples = [
            ("Does the text imply an impossible amount of work?", "COUNTER_INTERPRETATION", "tribunal"),
            ("Does the image support the caption's intended meaning?", "CAPTION_PREMISE", "tribunal"),
            ("Explain the idiom in the original caption.", "CAPTION_PREMISE", "language"),
            ("Read the exact visible text in each panel.", "COUNTER_INTERPRETATION", "visual"),
            ("Is there any readable text identifying the event?", "COUNTER_INTERPRETATION", "visual"),
            ("What is the final label?", "VISUAL_PREMISE", "blocked"),
            ("Does the character intend to insult the recipient?", "VISUAL_PREMISE", "tribunal")]
        for question, target, expected in examples:
            with self.subTest(question=question):
                self.assertEqual(route_question(question, target)[0], expected)

    def test_semantic_followup_reaches_packet_without_becoming_evidence(self):
        from engine.case_dossier import build_case_dossier, render_judge_dossier, compact_evidence_packet
        task = {"questions": ["Which interpretation preserves the depicted roles?"],
                "role": "tribunal_only", "is_new_visual_evidence": False}
        dossier = build_case_dossier("A caption", {}, {}, {}, {}, [], {"tribunal_semantic_questions": task})
        packet = compact_evidence_packet(render_judge_dossier(dossier))
        self.assertEqual(packet["tribunal_semantic_questions"], task)
        self.assertEqual(packet["evidence_ledger"], [])

    def test_spans_preserve_unicode_negation_and_whitespace(self):
        codec = SourceSpans("Nobody  won\n—not even Zoë.")
        self.assertEqual(codec.quote({"start": 0, "end": 3}), "Nobody  won\n—not")
        for span in ({"start": 0,"end": 99}, {"start": 2,"end": 1}, {"start": True,"end": 2}):
            with self.assertRaises(ValueError): codec.quote(span)

    def span_runtime(self, malformed=False):
        image, proposal, ledger, answers = fixture()
        answers[1]["bindings"][0].pop("caption_quote")
        answers[1]["bindings"][0]["caption_span"] = {"start": 1, "end": 2}
        answers[2]["condition_checks"][0].pop("caption_quote")
        answers[2]["condition_checks"][0]["caption_span"] = {"start": 3, "end": 4}
        if malformed:
            answers[2]["reason"] = "This is inconsistent with the"
            answers.insert(3, {"replacement": "Alarm conflicts with calm."})
        runtime = Runtime(answers)
        runtime.hardware_profile = SimpleNamespace(judge_source_spans=True)
        return image, proposal, ledger, runtime

    def test_bound_span_proof_accepts_and_changed_source_rejects(self):
        image, proposal, ledger, runtime = self.span_runtime()
        proof = verify(runtime, image, proposal, ledger)
        proposal["independent_verification"] = proof
        # Production audit is used through the same subject binding as live runs.
        from engine.evidence_verification import decision_valid
        self.assertTrue(decision_valid(proof["calls"][0], {"VF1"}, proposal["source_caption"]))
        self.assertTrue(audit(proposal, ledger)["valid"])
        self.assertEqual(proof["calls"][0]["condition_checks"][0]["caption_quote"], "calm.")
        self.assertFalse(decision_valid(proof["calls"][0], {"VF1"}, "The meeting is frantic."))
        changed = deepcopy(proposal)
        changed["source_caption"] = "The meeting is frantic."
        self.assertFalse(audit(changed, ledger)["valid"])

    def test_single_field_repair_keeps_relation_and_revalidates(self):
        image, proposal, ledger, runtime = self.span_runtime(malformed=True)
        proof = verify(runtime, image, proposal, ledger)
        decision = proof["calls"][0]
        self.assertTrue(decision["_format_valid"])
        self.assertTrue(decision["_field_repair_used"])
        self.assertEqual(decision["relation"], "CONFLICT")
        self.assertEqual(runtime.token_budgets[3], 160)
        self.assertEqual(len(runtime.prompts), 5)

    def test_budget_does_not_strand_fixed_reserve_or_extend_total(self):
        runtime = SimpleNamespace(hardware_profile=SimpleNamespace(judge_adaptive_budget=True,
                                    judge_verification_reserve_seconds=120))
        with patch("engine.case_budget.time.perf_counter", return_value=100):
            with case_budget(runtime, "case", 170):
                self.assertEqual(runtime._review_budget_deadline, 270)
                self.assertLess(proposal_reserve(runtime), 120)
                self.assertLess(budget_estimates(runtime)["followup"], 156)
                self.assertEqual(runtime._review_budget_deadline, 270)
        observe_cost(runtime, "proof", 60)
        self.assertEqual(budget_estimates(runtime)["proof"], 81)

    def test_witness_has_priority_over_guidance(self):
        from agents.multimodal_judge import TribunalMediatorAgent
        agent = TribunalMediatorAgent(SimpleNamespace())
        baseline = {"_format_valid": True, "relation": "UNRESOLVED", "status": "FOLLOW_UP",
                    "targeted_question": "Read the exact visible words on the sign.", "requested_follow_up": "VISUAL_PREMISE"}
        with patch.object(agent, "_review") as model:
            result = agent._precedent_followup(baseline, [{"id":"method"}], None, "source", {}, {}, {}, [], {}, 1, {}, {})
        model.assert_not_called()
        self.assertEqual(result["_precedent_feedback"]["reason"], "required_witness_before_optional_guidance")

    def test_visual_contrast_is_not_a_task_verdict(self):
        from agents.visual_adapter import exposes_answer_decision
        self.assertFalse(exposes_answer_decision("Contradiction: a tiny cup beside an enormous spoon."))
        for text in ("ENTAILS", "The caption contradicts the image.", "Final decision: support", "The answer is contradiction"):
            self.assertTrue(exposes_answer_decision(text), text)

    def test_repeated_ocr_is_invalid_not_silently_truncated(self):
        from engine.generation_recovery import repeated_tail
        from agents.visual_adapter import AtomicVisualQuestionController as C
        self.assertTrue(repeated_tail([1,2]*20))
        self.assertFalse(repeated_tail(list(range(40))))
        answer = C.validate_answer("logo "*40, "Read the text.", "ocr")
        self.assertFalse(answer[2]); self.assertEqual(answer[3], "repetitive_generation")

    def test_metrics_keep_earlier_harmful_proposal(self):
        from evaluation.tribunal_quality import summarize_tribunal
        row = {"ground_truth":"ENTAILS", "initial_prediction":"ENTAILS", "prediction":"ENTAILS",
               "judge_format_valid":False, "trace":{"judge":{"tribunal_reviews":[
                   {"_format_valid":True,"best_semantic_judgment":"CONTRADICTS"},
                   {"_format_valid":False,"_execution_status":"FAILED"}]}}}
        result = summarize_tribunal([row])
        self.assertEqual(result["all_hearing_and_guided_proposals"]["harmful"], 1)
        self.assertEqual(result["harmful_changes"], 0)

    def test_visual_dispatch_guard_never_calls_model_for_semantic_question(self):
        from engine.debate import DebateEngine
        agent = SimpleNamespace(critique=lambda *a: self.fail("semantic question reached visual model"))
        engine = DebateEngine()
        case = {"key":"case", "caption":"An expression", "image":object(), "visual_output":{},
                "decision":{}, "tribunal_hearing":True, "mediation_plan":{
                    "agent1_questions":["Does the text imply the person feels defeated?"]}}
        result = engine._run_visual_questions(agent, case)
        self.assertFalse(result["_format_valid"])
        self.assertEqual(result["response_status"], "NOT_DISPATCHED")

    def test_targeted_ocr_retry_does_not_double_repetition_budget(self):
        from agents.visual_adapter import VisualQuestion
        from agents.visual_grounding import VisualGroundingAgent
        from tests.test_qwen3vl_agent1 import FakeQwen3VLRuntime
        runtime = FakeQwen3VLRuntime(["logo " * 40, "SALE\nOPEN"])
        agent = VisualGroundingAgent(runtime)
        answer = agent._run_atomic_question(object(), VisualQuestion("crop", "ocr", "Read exact visible text.", 180))
        self.assertTrue(answer.valid)
        self.assertEqual([c[1] for c in runtime.calls], [180, 180])
        self.assertIn("spatial order", runtime.calls[1][0])

    def test_selective_ocr_preserves_locations_and_is_not_an_absence_claim(self):
        from agents.visual_grounding import VisualGroundingAgent
        agent = VisualGroundingAgent.__new__(VisualGroundingAgent)
        def generate(image, prompt, max_new_tokens, json_schema):
            self.assertIn("not an exhaustive", prompt)
            self.assertLessEqual(max_new_tokens, 240)
            return json.dumps({"status":"OBSERVED", "phrases":[
                {"text":"OPEN", "region":"top"}, {"text":"OPEN", "region":"bottom"}]}), .1
        agent.runtime = SimpleNamespace(generate=generate, _last_generation_diagnostics={})
        answer, _ = agent._recover_repeated_ocr(object(), 180)
        self.assertEqual(answer, "OPEN\nOPEN")
        self.assertEqual(len(agent._last_generation_diagnostics["ocr_phrase_locations"]), 2)
        self.assertEqual(agent._last_generation_diagnostics["ocr_coverage"], "selective_positive_observations")

    def test_selective_ocr_rejects_repeated_pair_and_partial_json(self):
        from agents.visual_grounding import VisualGroundingAgent
        from agents.visual_adapter import AtomicVisualQuestionController as C
        for raw in ('{', json.dumps({"status":"OBSERVED","phrases":[{"text":"OPEN","region":"top"}]*2})):
            agent = VisualGroundingAgent.__new__(VisualGroundingAgent)
            agent.runtime = SimpleNamespace(generate=lambda *a, **k:(raw,.1), _last_generation_diagnostics={})
            answer, _ = agent._recover_repeated_ocr(object(), 180)
            self.assertFalse(C.validate_answer(answer, "Read text", "ocr", agent._last_generation_diagnostics)[2])


if __name__ == "__main__": unittest.main()
