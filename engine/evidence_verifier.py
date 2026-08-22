"""Atomic, abstaining relation verification for grounded evidence entries."""

from copy import deepcopy
import time

from models.nli_model import NliVerifier


class AtomicEvidenceVerifier:
    """Nominate possible evidence directions for audit, without promoting them.

    Generic textual NLI is not equivalent to figurative image-caption
    entailment. Its scores are therefore diagnostic candidates until an
    independent grounded mechanism corroborates the direction.
    """

    MIN_RELATION_PROBABILITY = 0.70
    MIN_RELATION_MARGIN = 0.15
    MAX_ATOMIC_ITEMS = 8
    RUNTIME_PROMOTION_ENABLED = False

    def __init__(self, nli_verifier=None):
        self.nli = nli_verifier or NliVerifier()

    @staticmethod
    def claim_text(language_output):
        figurative_type = str(
            language_output.get("figurative_type", "")
        ).strip().lower()
        if figurative_type in {"sarcasm", "metaphor", "humor"}:
            intended = str(language_output.get("intended_meaning", "")).strip()
            if intended and intended.lower() not in {"none", "unknown"}:
                return intended
        return str(
            language_output.get("caption_proposition")
            or language_output.get("intended_meaning")
            or language_output.get("surface_meaning")
            or ""
        ).strip()

    @classmethod
    def _resolve_relation(cls, probabilities):
        entailment = float(probabilities.get("entailment", 0.0))
        contradiction = float(probabilities.get("contradiction", 0.0))
        neutral = float(probabilities.get("neutral", 0.0))
        if entailment >= contradiction:
            relation, selected, alternative = "SUPPORT", entailment, contradiction
        else:
            relation, selected, alternative = "CONFLICT", contradiction, entailment
        runner_up = max(alternative, neutral)
        accepted = (
            selected >= cls.MIN_RELATION_PROBABILITY
            and selected - runner_up >= cls.MIN_RELATION_MARGIN
        )
        return relation if accepted else "NEUTRAL", selected, selected - runner_up

    def verify(self, ledger, language_output, comparison=None):
        started = time.time()
        output = deepcopy(ledger or [])
        claim = self.claim_text(language_output)
        comparison = comparison or {}
        nominations = {
            " ".join(str(item.get("text", "")).casefold().split()): item
            for item in comparison.get("structured_relation_candidates", []) or []
            if str(item.get("text", "")).strip()
        }
        unresolved_binding = bool(
            comparison.get("relation_binding_required", False)
            and not comparison.get("relation_binding_observed", False)
        )
        candidates = []
        for index, item in enumerate(output):
            if not item.get("grounded", False):
                continue
            if item.get("source") != "agent1":
                continue
            if item.get("type") not in {
                "visual_fact", "visual_relation", "visible_text"
            }:
                continue
            if unresolved_binding and item.get("type") == "visible_text":
                item["verification"] = {
                    "status": "SKIPPED_UNRESOLVED_REGION_BINDING",
                    "decision_grade": False,
                }
                continue
            candidates.append((index, item))
            if len(candidates) >= self.MAX_ATOMIC_ITEMS:
                break

        if not claim or not candidates:
            return output, {
                "claim": claim,
                "candidate_count": len(candidates),
                "verified_count": 0,
                "support_count": 0,
                "conflict_count": 0,
                "neutral_count": len(candidates),
                "verification_policy": "generic_nli_diagnostic_only",
                "runtime_promotion_enabled": self.RUNTIME_PROMOTION_ENABLED,
                "promotion_reason": (
                    "Generic text NLI requires independent current-image corroboration."
                ),
                "seconds": round(time.time() - started, 4),
                "model": self.nli.MODEL_ID,
                "revision": self.nli.REVISION,
            }

        probabilities = self.nli.predict_batch(
            [(item["text"], claim) for _, item in candidates]
        )
        for (index, item), scores in zip(candidates, probabilities):
            candidate_relation, selected, margin = self._resolve_relation(scores)
            nomination = nominations.get(
                " ".join(str(item.get("text", "")).casefold().split())
            )
            nominated_relation = (
                nomination.get("proposed_relation") if nomination else None
            )
            diagnostic_agreement = bool(
                nominated_relation in {"SUPPORT", "CONFLICT"}
                and candidate_relation == nominated_relation
            )
            corroborated = bool(
                self.RUNTIME_PROMOTION_ENABLED and diagnostic_agreement
            )
            # Both the lexical nomination and generic NLI operate on generated
            # text, not image pixels. Their agreement is useful for routing but
            # is not independent multimodal corroboration.
            output[index]["relation"] = (
                nominated_relation if corroborated else "NEUTRAL"
            )
            output[index]["verification"] = {
                "status": (
                    "CORROBORATED_STRUCTURED_RELATION"
                    if corroborated
                    else "STRUCTURED_NLI_AGREEMENT_DIAGNOSTIC_ONLY"
                    if diagnostic_agreement
                    else "UNVALIDATED_NLI_CANDIDATE"
                    if candidate_relation != "NEUTRAL"
                    else "ABSTAINED"
                ),
                "decision_grade": corroborated,
                "diagnostic_agreement": diagnostic_agreement,
                "candidate_relation": candidate_relation,
                "nominated_relation": nominated_relation,
                "relation_family": (
                    nomination.get("relation_family") if nomination else None
                ),
                "matched_cues": nomination.get("matched_cues", []) if nomination else [],
                "claim": claim,
                "entailment": round(float(scores.get("entailment", 0.0)), 6),
                "contradiction": round(float(scores.get("contradiction", 0.0)), 6),
                "neutral": round(float(scores.get("neutral", 0.0)), 6),
                "selected_probability": round(selected, 6),
                "margin": round(margin, 6),
                "minimum_probability": self.MIN_RELATION_PROBABILITY,
                "minimum_margin": self.MIN_RELATION_MARGIN,
                "model": self.nli.MODEL_ID,
                "revision": self.nli.REVISION,
            }

        verified_items = [output[index] for index, _ in candidates]
        candidate_support_count = sum(
            item.get("verification", {}).get("candidate_relation") == "SUPPORT"
            for item in verified_items
        )
        candidate_conflict_count = sum(
            item.get("verification", {}).get("candidate_relation") == "CONFLICT"
            for item in verified_items
        )
        promoted_support_count = sum(
            item.get("verification", {}).get("decision_grade", False)
            and item.get("relation") == "SUPPORT"
            for item in verified_items
        )
        promoted_conflict_count = sum(
            item.get("verification", {}).get("decision_grade", False)
            and item.get("relation") == "CONFLICT"
            for item in verified_items
        )
        promoted_count = promoted_support_count + promoted_conflict_count
        return output, {
            "claim": claim,
            "candidate_count": len(candidates),
            "candidate_direction_count": (
                candidate_support_count + candidate_conflict_count
            ),
            "candidate_support_count": candidate_support_count,
            "candidate_conflict_count": candidate_conflict_count,
            "verified_count": promoted_count,
            "support_count": promoted_support_count,
            "conflict_count": promoted_conflict_count,
            "neutral_count": len(candidates) - promoted_count,
            "verification_policy": "generic_nli_diagnostic_only",
            "runtime_promotion_enabled": self.RUNTIME_PROMOTION_ENABLED,
            "promotion_reason": (
                "Generic text NLI requires independent current-image corroboration."
            ),
            "seconds": round(time.time() - started, 4),
            "model": self.nli.MODEL_ID,
            "revision": self.nli.REVISION,
        }


def merge_verified_evidence(comparison, ledger, verification_summary):
    """Attach verifier diagnostics without promoting uncorroborated NLI output."""
    output = deepcopy(comparison or {})
    output["_pre_verification_status"] = output.get("required_evidence_status")
    output["_pre_verification_recommendation"] = output.get("recommendation")
    output["atomic_evidence_verification"] = verification_summary
    output["grounded_evidence_catalog"] = [
        {
            "id": item["id"],
            "source": item.get("source"),
            "type": item.get("type"),
            "text": item.get("text", ""),
            "decision_grade": bool(
                item.get("decision_grade", False)
                or item.get("verification", {}).get("decision_grade", False)
            ),
            "verification_method": item.get("verification_method"),
        }
        for item in (ledger or [])
        if item.get("grounded", False)
        and item.get("source") in {
            "agent1", "comparator", "targeted_region_verifier"
        }
    ]
    support = list(output.get("supporting_evidence", []) or [])
    conflict = list(output.get("contradicting_evidence", []) or [])
    for item in ledger or []:
        verification = item.get("verification", {}) or {}
        if not verification.get("decision_grade", False):
            continue
        text = f"[VISUAL][{item['id']}] {item['text']}"
        if item.get("relation") == "SUPPORT" and text not in support:
            support.append(text)
        elif item.get("relation") == "CONFLICT" and text not in conflict:
            conflict.append(text)
    output["supporting_evidence"] = support
    output["contradicting_evidence"] = conflict
    output["supporting_points"] = support
    output["conflicting_points"] = conflict
    if support and conflict:
        output["required_evidence_status"] = "MIXED_VERIFIED_EVIDENCE"
        output["recommendation"] = "REVIEW_MIXED_EVIDENCE"
    elif support:
        output["required_evidence_status"] = "SUPPORTED"
        output["recommendation"] = "LEAN_ENTAILS"
    elif conflict:
        output["required_evidence_status"] = "CONFLICTING"
        output["recommendation"] = "LEAN_CONTRADICTS"
    return output
