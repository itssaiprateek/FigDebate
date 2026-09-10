import json
import unittest

from engine.independent_review import verify_independently, audit_independent_record, parse_verification
from engine.independent_review import image_subject_hash
from PIL import Image


class IndependentReviewTests(unittest.TestCase):
    def test_actual_calls_are_label_blind_swapped_and_bound(self):
        calls = []
        class Runtime:
            _last_generation_diagnostics = {}
            def generate(self, _image, prompt, json_schema=None, **kwargs):
                calls.append(prompt)
                fields = json_schema["properties"]
                value = ({"verified": True} if "verified" in fields else
                    {"bridge_relation": "CONFLICT", "bridge_grounded": True,
                     "counter_relation": "UNRESOLVED", "counter_resolved": True} if "bridge_relation" in fields else
                    {"relation": "CONFLICT", "evidence_ids": ["VF1"]})
                return json.dumps(dict(value, reason="The line falls, not rises.")), 0.01
        proposal = {"source_caption": "The line rises.", "caption_premise": "The line rises.",
                    "visual_premise": "The line falls.", "visual_evidence_ids": ["VF1"],
                    "proposed_relation": "CONFLICT", "bridge_statement": "SECRET_PROPOSED_REASON",
                    "confidence": 0.999}
        ledger = [{"id": "VF1", "text": "The line falls."}]
        image = Image.new("RGB", (2, 2))
        proposal["image_sha256"] = image_subject_hash(image)
        proposal["independent_verification"] = verify_independently(Runtime(), image, proposal, ledger)
        self.assertEqual(len(calls), 5)
        self.assertNotEqual(calls[2], calls[3])
        self.assertTrue(all("SECRET_PROPOSED_REASON" not in call and "0.999" not in call for call in calls[:4]))
        self.assertTrue(audit_independent_record(proposal, ledger)["valid"])
        proposal["visual_premise"] = "The line rises."
        self.assertFalse(audit_independent_record(proposal, ledger)["valid"])

    def test_strings_are_not_valid_boolean_proof(self):
        value = {"relation": "SUPPORT", "visual_premise_supported": "true",
                 "caption_premise_preserved": True, "entity_scope_consistent": True,
                 "evidence_ids": ["VF1"], "reason": "Reason"}
        self.assertFalse(parse_verification(json.dumps(value))["_format_valid"])

    def test_no_independent_record_means_no_corroboration(self):
        self.assertFalse(audit_independent_record({"confidence": 1.0}, [])["valid"])
