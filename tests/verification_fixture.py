"""Explicit mocked verifier outcomes; never imported by production code."""
import json
from engine.independent_review import verify_independently
from engine.independent_review import image_subject_hash
from PIL import Image
from engine.semantic_bridge import build_semantic_bridge


def with_independent_fixture(review, ledger, contract):
    review = dict(review)
    image = Image.new("RGB", (2, 2), "white")
    review["_case_image_sha256"] = image_subject_hash(image)
    proposal = build_semantic_bridge(review, ledger, contract)
    class Runtime:
        _last_generation_diagnostics = {}
        def generate(self, _image, _prompt, json_schema=None, **kwargs):
            fields = (json_schema or {}).get("properties", {})
            if "bridge_relation" in fields:
                value = {"bridge_relation": proposal["proposed_relation"], "bridge_grounded": True,
                         "counter_relation": "UNRESOLVED", "counter_resolved": True}
            elif "verified" in fields:
                value = {"verified": True}
            else:
                value = {"relation": proposal["proposed_relation"], "evidence_ids": proposal["visual_evidence_ids"]}
            return json.dumps(dict(value, reason="Explicit synthetic verification fixture, not model quality.")), 0.01
    review["_independent_verification"] = verify_independently(Runtime(), image, proposal, ledger)
    return review
