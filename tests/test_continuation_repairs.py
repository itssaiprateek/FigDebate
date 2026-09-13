import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

from engine.output_contracts import schema_for_contract, validate_shape
from engine.claim_semantics import audit_core, CHECKS
from engine.candidate_cases import candidate_dispute, candidate_hearing_plan
from engine.cache_identity import local_dependencies
from engine.stage_checkpoint import StageCheckpointStore
from engine.deliberation import deliberation_signature
from engine.case_dossier import fit_judge_packet
from engine.final_artifact import final_artifact


class ContinuationRepairsTests(unittest.TestCase):
    def test_text_and_list_limits_are_checked(self):
        self.assertFalse(validate_shape("long", {"type": "string", "maxLength": 3}))
        self.assertFalse(validate_shape(["a", "b"], {"type": "array", "maxItems": 1, "items": {"type": "string"}}))
        schema = schema_for_contract("tribunal_review")
        self.assertNotIn("maxLength", schema["properties"]["visual_premise"])
        self.assertEqual(schema["properties"]["evidence_ids"]["maxItems"], 4)

    def test_failed_audit_cannot_omit_repair_targets(self):
        output = {key: True for key in CHECKS}
        output.update(opposite_state_conflicts_with_full_claim=False, defective_fields=[], reason="bad state")
        class Agent:
            def _generate_section(self, *args, **kwargs):
                return output, "fixture", .1, {"schema_valid": True}
        audit = audit_core(Agent(), "Caption", {})
        self.assertIn("opposite_visual_state", audit["defective_fields"])
        self.assertFalse(audit["_format_valid"])

    def test_agreement_does_not_force_hearing(self):
        candidate = dict(relation="SUPPORT", decisive_question="none needed", _format_valid=True,
                         _execution_status="SUCCEEDED", claim_reading="Claim", decisive_observation="Fact", alternative="None")
        dispute = candidate_dispute({"_independent_candidate": candidate}, {"_independent_candidate": candidate})
        self.assertFalse(dispute["genuine_disagreement"])
        self.assertEqual(candidate_hearing_plan({"_usable": False}, dispute), {"_usable": False})

    def test_unresolved_candidate_routes_question_not_answer(self):
        dispute = {"unresolved": True, "decisive_questions": ["Which panel contains the number?"]}
        original = {"agent1_questions": ["Read visible text and its panel attachments."],
                    "agent2_questions": ["Explain the caption requirements."]}
        plan = candidate_hearing_plan(original, dispute)
        self.assertTrue(plan["_usable"])
        self.assertEqual(plan["agent1_questions"], original["agent1_questions"])
        self.assertEqual(plan["agent2_questions"], original["agent2_questions"])
        self.assertIn("Which panel", plan["candidate_question_routing"][0]["question"])
        self.assertEqual(plan["candidate_question_routing"][0]["destination"], "judge_candidate_context")
        self.assertNotIn("provisional_verdict", plan)

    def test_claim_clarification_is_new_information(self):
        old = {"evidence_ledger": [], "debate_details": {"agent2_critique": {"support_requirement": "growth"}}}
        new = deepcopy(old)
        new["debate_details"]["agent2_critique"]["support_requirement"] = "growth without difficulty"
        self.assertNotEqual(deliberation_signature(old), deliberation_signature(new))

    def test_static_cache_closure_includes_local_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Test fixtures only, not project edits.
            (root / "a.py").write_text("from helper import value")
            (root / "helper.py").write_text("value = 1")
            self.assertEqual(local_dependencies(root, ["a.py"]), ["a.py", "helper.py"])

    def test_shared_cache_never_reuses_downstream_or_incompatible_output(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as destination:
            sample = {"raw": {"id": "case"}, "caption": "original"}
            StageCheckpointStore(source, fingerprint="same").save("visual_grounding", sample, {"ok": 1})
            reader = StageCheckpointStore(destination, fingerprint="same", readonly_sources=[source])
            self.assertEqual(reader.load("visual_grounding", sample), {"ok": 1})
            self.assertIsNone(reader.load("tribunal_round_1", sample))
            sample["caption"] = "changed"
            self.assertIsNone(reader.load("visual_grounding", sample))
            self.assertFalse(reader.reuse_events[-1]["reused"])

    def test_tampered_checkpoint_payload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = {"raw": {"id": "case"}}
            store = StageCheckpointStore(tmp, enabled=True)
            store.save("stage", sample, {"value": 1})
            path = Path(store._path("stage", sample))
            record = json.loads(path.read_text())
            record["payload"]["value"] = 2
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                store.load("stage", sample)

    def test_requested_evidence_cannot_be_silently_dropped(self):
        packet = {"evidence_ledger": [{"id": "A", "text": "x" * 1000}, {"id": "B", "text": "y" * 1000}]}
        with self.assertRaisesRegex(ValueError, "Requested evidence"):
            fit_judge_packet(packet, json.dumps, max_tokens=20, protected_ids=["A", "B"])

    def test_evidence_removal_breaks_explanation_chain(self):
        item = {"id": "E", "text": "Verified relation", "relation": "SUPPORT", "grounded": True, "decision_grade": True}
        result = {"decision": {"label": "ENTAILS", "_model_cited_evidence_ids": ["E"]}, "evidence_ledger": [item]}
        self.assertTrue(final_artifact("Claim", result)["evidence_chain_complete"])
        result["evidence_ledger"] = []
        self.assertFalse(final_artifact("Claim", result)["evidence_chain_complete"])

    def test_counterfactual_relation_cannot_keep_same_explanation_support(self):
        item = {"id": "E", "text": "Verified relation", "relation": "CONFLICT", "grounded": True, "decision_grade": True}
        result = {"decision": {"label": "ENTAILS", "_model_cited_evidence_ids": ["E"]}, "evidence_ledger": [item]}
        self.assertFalse(final_artifact("Claim", result)["evidence_chain_complete"])


class InstalledTokenizerTests(unittest.TestCase):
    def test_current_transformers_adapter_accepts_complete_json(self):
        # CPU-only integration: catches the removed Transformers alias which
        # mocked model tests missed. No download and no GPU/model weights.
        from models.judge_model import default_judge_model_path
        location = Path(default_judge_model_path())
        if not (location / "tokenizer.json").is_file():
            self.skipTest("Local qualified tokenizer unavailable")
        import torch
        from transformers import AutoTokenizer
        from engine.output_contracts import prefix_constraint, complete_json_stopper, object_schema
        tokenizer = AutoTokenizer.from_pretrained(str(location), local_files_only=True)
        schema = object_schema({"ok": {"type": "boolean"}})
        allowed = prefix_constraint(tokenizer, schema)
        prefix = tokenizer.encode("Input:", add_special_tokens=False)
        first = allowed(0, torch.tensor(prefix))
        self.assertTrue(first)
        for token in tokenizer.encode('{"ok":true}', add_special_tokens=False):
            self.assertIn(token, allowed(0, torch.tensor(prefix)))
            prefix.append(token)
        stopper = complete_json_stopper(tokenizer, 0, schema)
        self.assertTrue(stopper(torch.tensor([tokenizer.encode('{"ok":true}', add_special_tokens=False)]), None))
