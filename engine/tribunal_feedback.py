"""Frozen, label-free reasoning precedents for the tribunal, never its verifier.

Synthetic methodological examples are distinguished from reviewed training cases.
No online updates, answer labels, lexical similarity, or sample-ID routing.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


FEATURES = {"figurative", "comparison", "speaker_scope", "polarity"}


def structural_features(language):
    language = language or {}
    contract = language.get("claim_contract") or {}
    relation = language.get("claim_relation") or {}
    kind = str(language.get("figurative_type", "")).lower()
    features = set()
    if kind in {"humor", "metaphor", "sarcasm"}:
        features.add("figurative")
    structure = str(contract.get("structural_reasoning_type", "")).upper()
    direction = str(contract.get("comparison_direction") or "").strip().casefold()
    if (relation.get("relation_family") in {"comparison", "comparative"} or structure == "COMPARATIVE_LAYOUT"
            or direction not in {"", "unknown", "unresolved", "none", "not_applicable", "n/a"}):
        features.add("comparison")
    # A time interval or a metaphor's source domain does not identify a speaker.
    # Require explicit attribution structure rather than any nonempty scope field.
    if structure == "QUOTED_STATEMENT_AND_REACTION" or contract.get("requires_speaker_attribution") is True:
        features.add("speaker_scope")
    if language.get("polarity_reversal") is True or kind == "sarcasm":
        features.add("polarity")
    return kind, features


class TribunalPrecedents:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        document = json.loads(raw)
        if set(document) != {"schema_version", "entries"} or document["schema_version"] != "tribunal-precedents-1":
            raise ValueError("Invalid tribunal precedent library")
        entries = document["entries"]
        if not isinstance(entries, list) or len(entries) > 100:
            raise ValueError("Expected at most 100 frozen precedents")
        seen = set()
        fields = {"id", "origin", "review_note", "types", "features", "principle", "applies_when", "exclude_when", "example", "counterexample", "source"}
        for e in entries:
            if not isinstance(e, dict) or set(e) != fields:
                raise ValueError("Unexpected precedent fields; labels and arbitrary metadata are forbidden")
            if not isinstance(e["id"], str) or not e["id"] or e["id"] in seen:
                raise ValueError("Precedent IDs must be nonempty and unique")
            seen.add(e["id"])
            for key in ("review_note", "principle", "applies_when", "exclude_when", "example", "counterexample"):
                if not isinstance(e[key], str) or not e[key].strip() or len(e[key]) > 360:
                    raise ValueError("Precedent text must be nonempty and bounded")
            if not isinstance(e["features"], list) or not e["features"] or not set(e["features"]) <= FEATURES:
                raise ValueError("Unknown structural features")
            if not isinstance(e["types"], list) or not e["types"] or not set(e["types"]) <= {"humor", "metaphor", "sarcasm"}:
                raise ValueError("Unknown figurative types")
            if e["origin"] == "synthetic_methodological":
                if e["source"] != {}:
                    raise ValueError("Synthetic precedents cannot claim dataset provenance")
            elif e["origin"] == "reviewed_training_case":
                source = e["source"]
                if set(source) != {"split", "image_sha256", "caption_sha256", "group_id"} or source["split"] != "train":
                    raise ValueError("Empirical precedents require training provenance and a template group")
                if any(not isinstance(source[k], str) or not source[k] for k in source):
                    raise ValueError("Missing precedent provenance")
                for key in ("image_sha256", "caption_sha256"):
                    if len(source[key]) != 64 or any(c not in "0123456789abcdef" for c in source[key]):
                        raise ValueError("Invalid source digest")
            else:
                raise ValueError("Unreviewed or evaluation-derived memory is forbidden")
        self._entries = deepcopy(entries)

    def assert_disjoint(self, identities):
        """Fail before inference on overlapping images, captions or template groups."""
        identities = list(identities)
        empirical = [e["source"] for e in self._entries if e["origin"] == "reviewed_training_case"]
        for source in empirical:
            for current in identities:
                if not all(current.get(k) for k in ("image_sha256", "caption_sha256", "group_id")):
                    raise ValueError("Empirical precedent evaluation requires image/caption hashes and template group IDs")
                if any(current[k] == source[k] for k in ("image_sha256", "caption_sha256", "group_id")):
                    raise ValueError("Precedent/evaluation overlap")

    def retrieve(self, language, max_items=1):
        kind, features = structural_features(language)
        matches = [e for e in self._entries if kind in e["types"] and set(e["features"]) <= features]
        matches.sort(key=lambda e: (-len(e["features"]), e["id"]))
        return [{k: deepcopy(e[k]) for k in ("id", "origin", "principle", "applies_when", "exclude_when", "example", "counterexample")}
                for e in matches[:min(2, max(0, max_items))]]
