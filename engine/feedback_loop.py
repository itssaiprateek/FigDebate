"""Auditable, verified prompt feedback for FigDebate.

This module deliberately does not infer an error from the system's own output.
It either collects a gold-verified error candidate or applies a human-reviewed
prompt example supplied before a run. That prevents self-reinforcing feedback.
"""

import json
import os
import re
from datetime import datetime

from engine.evidence_ledger import RELATION_FOR_LABEL, evidence_ids
from engine.review_board import review_revision


class FeedbackLoop:
    VALID_AGENTS = {"agent1", "agent2", "arbiter"}
    CALIBRATED_RULES = {
        "missed_grounded_support": (
            "When two or more concrete visual anchors match a caption claim, "
            "treat that as positive entailment evidence instead of rejecting it "
            "because a broader interpretation is uncertain."
        ),
        "missed_grounded_conflict": (
            "When a concrete visual fact directly opposes an explicit caption "
            "direction, treat it as contradiction evidence rather than relying "
            "only on general scene similarity."
        ),
        "unsupported_contradiction": (
            "Never use absent support as a contradiction. A CONTRADICTS decision "
            "needs an observed visual conflict with the caption meaning."
        ),
        "unsupported_entailment": (
            "Never use broad topic similarity as entailment. An ENTAILS decision "
            "needs an observed visual fact that supports the caption meaning."
        ),
        "caption_analysis_unresolved": (
            "When the caption itself has no clear figurative device, mark it "
            "literal instead of guessing a phenomenon that may exist only in the image."
        ),
    }
    RULE_CONDITIONS = {
        "missed_grounded_support": {
            "comparison_status": "SUPPORTED",
            "requires_support": True,
        },
        "missed_grounded_conflict": {
            "comparison_status": "CONFLICTING",
            "requires_conflict": True,
        },
        "unsupported_contradiction": {
            "decision_label": "CONTRADICTS",
            "requires_support": True,
            "requires_no_conflict": True,
        },
        "unsupported_entailment": {
            "decision_label": "ENTAILS",
            "requires_conflict": True,
            "requires_no_support": True,
        },
        "caption_analysis_unresolved": {
            "figurative_type": "unknown",
        },
    }

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "has", "have", "he", "her", "his", "i", "in", "is", "it",
        "its", "of", "on", "or", "she", "that", "the", "their", "them",
        "they", "this", "to", "was", "were", "with", "you", "your",
    }
    MIN_CASE_SIMILARITY = 0.72
    MEMORY_SCHEMA_VERSION = 2

    def __init__(self, max_examples=100, log_file="step4_feedback_log.json"):
        self.max_examples = max_examples
        self.log_file = log_file
        self.agent1_memory = []
        self.agent2_memory = []
        self.arbiter_memory = []
        print("[FeedbackLoop] Ready.")

    def _memory_for(self, agent):
        if agent not in self.VALID_AGENTS:
            raise ValueError(f"Unknown feedback agent: {agent}")
        if agent == "agent1":
            return self.agent1_memory
        if agent == "agent2":
            return self.agent2_memory
        return self.arbiter_memory

    def save_log(self, event_type, metadata, details):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "metadata": metadata or {},
            "details": details or {},
        }
        data = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError):
                data = []
        data.append(entry)
        with open(self.log_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    @classmethod
    def _infer_legacy_failure_type(cls, failure_type, example):
        if failure_type in cls.RULE_CONDITIONS:
            return failure_type
        text = str(example or "").lower()
        for known, advice in cls.CALIBRATED_RULES.items():
            if text == advice.lower():
                return known
        return failure_type

    def add_verified_example(
        self,
        agent,
        example,
        failure_type,
        metadata=None,
        conditions=None,
    ):
        """Apply one externally reviewed correction to a selected agent prompt."""
        if not isinstance(example, str) or not example.strip():
            raise ValueError("A verified feedback example must be non-empty text.")
        failure_type = self._infer_legacy_failure_type(failure_type, example)
        item = {
            "agent": agent,
            "example": example.strip(),
            "failure_type": failure_type,
            "conditions": conditions or self.RULE_CONDITIONS.get(failure_type, {}),
            "metadata": metadata or {},
        }
        memory = self._memory_for(agent)
        if any(
            stored.get("example") == item["example"]
            and stored.get("failure_type") == item["failure_type"]
            for stored in memory
        ):
            return False
        memory.append(item)
        if len(memory) > self.max_examples:
            memory.pop(0)
        self.save_log(
            "verified_prompt_update",
            metadata,
            {
                "agent": agent,
                "failure_type": failure_type,
                "memory_size": len(memory),
                "example": example.strip(),
                "conditions": item["conditions"],
            },
        )
        return True

    @classmethod
    def _tokens(cls, value):
        return {
            token for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
            if len(token) > 2 and token not in cls.STOPWORDS
        }

    @classmethod
    def build_case_signature(cls, context):
        context = context or {}
        language = context.get("language_output", {}) or {}
        visual = context.get("visual_output", {}) or {}
        comparison = context.get("comparison", {}) or {}
        ledger = context.get("evidence_ledger", []) or []
        claim_relation = (
            language.get("claim_relation")
            or comparison.get("claim_relation")
            or {}
        )
        text_parts = [
            language.get("caption_proposition", ""),
            language.get("intended_meaning", ""),
        ]
        relations = sorted({
            item.get("relation") for item in ledger
            if item.get("relation") in {"SUPPORT", "CONFLICT"}
            and item.get("grounded", False)
            and (
                item.get("decision_grade", False)
                or item.get("verification", {}).get(
                    "decision_grade", False
                )
            )
        })
        claim_contract = language.get("claim_contract", {}) or {}
        return {
            "tokens": sorted(cls._tokens(" ".join(map(str, text_parts)))),
            "figurative_type": str(language.get("figurative_type", "")).lower(),
            "evidence_status": comparison.get("required_evidence_status"),
            "evidence_relations": relations,
            "relation_family": claim_relation.get("relation_family", "unresolved"),
            "claim_polarity": claim_relation.get("polarity", "unresolved"),
            "claim_contract_valid": bool(
                claim_contract.get("safe_for_directional_reasoning", False)
            ),
        }

    @staticmethod
    def _diagnostic_for_failure(failure_type):
        return {
            "missed_grounded_support": (
                "Which current-image facts positively instantiate the caption relation?"
            ),
            "missed_grounded_conflict": (
                "Which observed current-image state is the explicit opposite of the caption relation?"
            ),
            "unsupported_contradiction": (
                "Does an observed opposite relation exist, or is support merely absent?"
            ),
            "unsupported_entailment": (
                "Does a current-image fact support the intended relation, or is the overlap only thematic?"
            ),
            "caption_analysis_unresolved": (
                "Is the caption itself figurative, or is the figurative mechanism only visual?"
            ),
        }.get(failure_type, "What current-sample evidence would prevent repeating this reasoning error?")

    def add_verified_case(
        self,
        context,
        ground_truth,
        failure_type,
        advice,
        metadata=None,
    ):
        if ground_truth not in {"ENTAILS", "CONTRADICTS"}:
            return False
        metadata = metadata or {}
        memory_id = str(
            metadata.get("sample_id")
            or metadata.get("id")
            or f"case_{len(self.arbiter_memory) + 1}"
        )
        signature = self.build_case_signature(context)
        if (
            signature.get("relation_family") == "unresolved"
            or not signature.get("claim_contract_valid", False)
        ):
            return False
        pattern_key = "|".join(map(str, (
            failure_type,
            signature.get("relation_family"),
            signature.get("claim_polarity"),
            signature.get("evidence_status"),
            signature.get("figurative_type"),
            " ".join(signature.get("tokens", [])[:8]),
        )))
        existing = next(
            (item for item in self.arbiter_memory if item.get("pattern_key") == pattern_key),
            None,
        )
        if existing:
            source_ids = existing.setdefault("source_sample_ids", [])
            if memory_id not in source_ids:
                source_ids.append(memory_id)
            return False
        item = {
            "agent": "arbiter",
            "schema_version": self.MEMORY_SCHEMA_VERSION,
            "memory_type": "procedural_case",
            "memory_id": memory_id,
            "failure_type": failure_type,
            "failure_mechanism": failure_type,
            "repair_action": advice.strip(),
            "example": advice.strip(),
            "signature": signature,
            "pattern_key": pattern_key,
            "diagnostic_question": self._diagnostic_for_failure(failure_type),
            "application_conditions": {
                "relation_family": signature.get("relation_family"),
                "evidence_status": signature.get("evidence_status"),
                "figurative_type": signature.get("figurative_type"),
                "claim_contract_valid": True,
            },
            "exclusion_conditions": [
                "Do not copy source-sample visual facts.",
                "Do not change a label without current-image evidence.",
                "Missing support alone is not conflict.",
            ],
            "source_sample_ids": [memory_id],
            "reliability": {
                "applications": 0,
                "corrections": 0,
                "harms": 0,
                "beta_mean": 0.5,
            },
            "metadata": metadata,
        }
        self.arbiter_memory.append(item)
        if len(self.arbiter_memory) > self.max_examples:
            self.arbiter_memory.pop(0)
        self.save_log(
            "verified_case_memory",
            metadata,
            {
                "memory_id": memory_id,
                "failure_type": failure_type,
                "memory_type": "procedural_case",
                "signature": item["signature"],
            },
        )
        return True

    def load_verified_examples(self, path):
        """Load immutable procedural memory, migrating legacy cases safely."""
        with open(path, "r", encoding="utf-8") as handle:
            examples = json.load(handle)
        if not isinstance(examples, list):
            raise ValueError("Verified feedback file must contain a JSON list.")
        for item in examples:
            if not isinstance(item, dict):
                raise ValueError("Each verified feedback item must be an object.")
            if item.get("memory_type") in {
                "verified_case", "procedural_case"
            }:
                required = {"memory_id", "example", "signature"}
                missing = required - set(item)
                if missing:
                    raise ValueError(
                        f"Verified case memory is missing: {sorted(missing)}"
                    )
                loaded = {
                    key: value for key, value in item.items()
                    if key not in {
                        "verified_relation", "source_evidence_ids"
                    }
                }
                loaded.update({
                    "agent": "arbiter",
                    "schema_version": self.MEMORY_SCHEMA_VERSION,
                    "memory_type": "procedural_case",
                    "failure_mechanism": loaded.get(
                        "failure_mechanism",
                        loaded.get("failure_type", "reviewed_error"),
                    ),
                    "repair_action": loaded.get(
                        "repair_action", loaded.get("example", "")
                    ),
                    "diagnostic_question": loaded.get(
                        "diagnostic_question",
                        self._diagnostic_for_failure(
                            loaded.get("failure_type", "reviewed_error")
                        ),
                    ),
                    "reliability": loaded.get("reliability") or {
                        "applications": 0,
                        "corrections": 0,
                        "harms": 0,
                        "beta_mean": 0.5,
                    },
                })
                loaded["signature"] = {
                    key: value for key, value in loaded["signature"].items()
                    if key != "initial_label"
                }
                if not any(
                    stored.get("memory_id") == loaded["memory_id"]
                    for stored in self.arbiter_memory
                ):
                    self.arbiter_memory.append(loaded)
                continue
            self.add_verified_example(
                item.get("agent"),
                item.get("example"),
                item.get("failure_type", "reviewed_error"),
                item.get("metadata", {}),
                item.get("conditions"),
            )

    @staticmethod
    def _rule_applies(item, context):
        context = context or {}
        comparison = context.get("comparison", {}) or {}
        decision = context.get("decision", {}) or {}
        language = context.get("language_output", {}) or {}
        conditions = item.get("conditions", {}) or {}
        if not conditions:
            return False
        if conditions.get("comparison_status") and comparison.get(
            "required_evidence_status"
        ) != conditions["comparison_status"]:
            return False
        if conditions.get("decision_label") and decision.get("label") != conditions[
            "decision_label"
        ]:
            return False
        if conditions.get("figurative_type") and str(
            language.get("figurative_type", "")
        ).lower() != conditions["figurative_type"]:
            return False
        support = bool(comparison.get("supporting_evidence"))
        conflict = bool(comparison.get("contradicting_evidence"))
        if conditions.get("requires_support") and not support:
            return False
        if conditions.get("requires_conflict") and not conflict:
            return False
        if conditions.get("requires_no_support") and support:
            return False
        if conditions.get("requires_no_conflict") and conflict:
            return False
        return True

    @classmethod
    def _case_similarity(cls, item, context):
        current = cls.build_case_signature(context)
        stored = item.get("signature", {}) or {}
        current_family = current.get("relation_family", "unresolved")
        stored_family = stored.get("relation_family", "unresolved")
        if (
            current_family == "unresolved"
            or stored_family == "unresolved"
            or current_family != stored_family
            or not current.get("claim_contract_valid", False)
        ):
            return 0.0
        current_type = current.get("figurative_type")
        stored_type = stored.get("figurative_type")
        if (
            current_type and stored_type
            and current_type not in {"unknown", "literal"}
            and stored_type not in {"unknown", "literal"}
            and current_type != stored_type
        ):
            return 0.0
        current_tokens = set(current.get("tokens", []))
        stored_tokens = set(stored.get("tokens", []))
        union = current_tokens | stored_tokens
        token_score = len(current_tokens & stored_tokens) / len(union) if union else 0.0
        if token_score < 0.15:
            return 0.0
        score = 0.35 * token_score
        if current_type == stored_type:
            score += 0.15
        if current.get("evidence_status") == stored.get("evidence_status"):
            score += 0.15
        score += 0.25
        current_polarity = current.get("claim_polarity", "unresolved")
        stored_polarity = stored.get("claim_polarity", "unresolved")
        if current_polarity != "unresolved" and current_polarity == stored_polarity:
            score += 0.10
        return round(score, 6)

    def matching_rules(self, agent, context, max_rules=2):
        matches = []
        for item in self._memory_for(agent):
            if item.get("memory_type") in {
                "verified_case", "procedural_case"
            }:
                reliability = item.get("reliability", {}) or {}
                applications = int(reliability.get("applications", 0) or 0)
                beta_mean = float(reliability.get("beta_mean", 0.5) or 0.5)
                if applications >= 3 and beta_mean < 0.5:
                    continue
                score = self._case_similarity(item, context)
                if score >= self.MIN_CASE_SIMILARITY:
                    matched = dict(item)
                    matched["_match_score"] = score
                    matches.append(matched)
            elif self._rule_applies(item, context):
                matched = dict(item)
                matched["_match_score"] = 1.0
                matches.append(matched)
        return sorted(
            matches,
            key=lambda item: (-item.get("_match_score", 0.0), item.get("memory_id", "")),
        )[:max_rules]

    def build_prompt(self, agent, instruction, context=None):
        memory = self.matching_rules(agent, context)
        if not memory:
            return None
        examples = "\n\n".join(
            (
                f"Verified reasoning pattern {item.get('pattern_key', item.get('memory_id'))} "
                f"(similarity {item.get('_match_score', 0.0):.3f}):\n"
                f"Principle: {item['example']}\n"
                f"Current-case diagnostic: {item.get('diagnostic_question', self._diagnostic_for_failure(item.get('failure_type')))}\n"
                "Use only current-image evidence when answering the diagnostic."
                if item.get("memory_type") in {
                    "verified_case", "procedural_case"
                }
                else f"Verified rule {item['failure_type']}:\n{item['example']}"
            )
            for item in memory
        )
        return (
            f"{instruction}\n\n"
            "Use the reviewed examples only as error-avoidance guidance. "
            "Do not copy their facts into the current sample.\n\n"
            f"{examples}"
        )

    def matching_rule_ids(self, agent, context):
        return [
            item.get("memory_id") or item["failure_type"]
            for item in self.matching_rules(agent, context)
        ]

    def matching_rule_scores(self, agent, context):
        return {
            item.get("memory_id") or item["failure_type"]: item.get("_match_score", 0.0)
            for item in self.matching_rules(agent, context)
        }

    @staticmethod
    def _relation_strength(ledger, relation):
        strengths = []
        for item in ledger or []:
            if item.get("relation") != relation or not item.get("grounded", False):
                continue
            if (
                item.get("source") in {
                    "comparator", "targeted_region_verifier"
                }
                and (
                    item.get("decision_grade", False)
                    or item.get("verification", {}).get(
                        "decision_grade", False
                    )
                )
            ):
                strengths.append(1.0)
        return max(strengths, default=0.0)

    @classmethod
    def accept_feedback_revision(cls, original, candidate, ledger, matches):
        # Procedural memory may request a review, but it never supplies a gold
        # direction. Any candidate is judged only on current-image evidence.
        accepted, reason, _ = review_revision(
            original, candidate, ledger, visual_review=None
        )
        return accepted, reason

    def record_reliability_outcome(
        self, memory_ids, original_label, final_label, ground_truth
    ):
        """Update reliability after a held-out batch, never during inference."""
        updates = []
        for item in self.arbiter_memory:
            if item.get("memory_id") not in set(memory_ids or []):
                continue
            reliability = item.setdefault("reliability", {})
            reliability["applications"] = int(
                reliability.get("applications", 0) or 0
            ) + 1
            if original_label != final_label:
                if original_label != ground_truth and final_label == ground_truth:
                    reliability["corrections"] = int(
                        reliability.get("corrections", 0) or 0
                    ) + 1
                elif original_label == ground_truth and final_label != ground_truth:
                    reliability["harms"] = int(
                        reliability.get("harms", 0) or 0
                    ) + 1
            corrections = int(reliability.get("corrections", 0) or 0)
            harms = int(reliability.get("harms", 0) or 0)
            reliability["beta_mean"] = round(
                (corrections + 1) / (corrections + harms + 2), 6
            )
            updates.append({
                "memory_id": item.get("memory_id"),
                **reliability,
            })
        return updates

    def calibration_rule(self, language_output, comparison, decision, ground_truth, phenomenon=None):
        """Return a pre-approved general rule for a known development error.

        The rule depends only on the gold label and existing grounded signals;
        it never lets the system invent a sample-specific explanation.
        """
        prediction = decision.get("label")
        if prediction == ground_truth:
            return None

        status = comparison.get("required_evidence_status")
        if prediction == "CONTRADICTS" and status == "SUPPORTED":
            failure_type = "missed_grounded_support"
            return "arbiter", failure_type, self.CALIBRATED_RULES[failure_type]
        if prediction == "ENTAILS" and status == "CONFLICTING":
            failure_type = "missed_grounded_conflict"
            return "arbiter", failure_type, self.CALIBRATED_RULES[failure_type]
        if prediction == "CONTRADICTS" and not comparison.get("contradicting_evidence"):
            failure_type = "unsupported_contradiction"
            return "arbiter", failure_type, self.CALIBRATED_RULES[failure_type]
        if prediction == "ENTAILS" and not comparison.get("supporting_evidence"):
            failure_type = "unsupported_entailment"
            return "arbiter", failure_type, self.CALIBRATED_RULES[failure_type]
        if str(language_output.get("figurative_type", "")).lower() == "unknown":
            failure_type = "caption_analysis_unresolved"
            return "agent2", failure_type, self.CALIBRATED_RULES[failure_type]
        return None

    def export_examples(self):
        """Serialize immutable calibrated guidance for a later held-out run."""
        examples = []
        for agent in sorted(self.VALID_AGENTS):
            for item in self._memory_for(agent):
                examples.append({
                    **item,
                    "metadata": {
                        **item.get("metadata", {}),
                        "source": "development_calibration",
                    },
                })
        return examples

    @staticmethod
    def classify_verified_error(
        language_output,
        decision,
        ground_truth,
        phenomenon=None,
        visual_output=None,
        comparison=None,
    ):
        """Conservatively categorize a known wrong final label for review."""
        if ground_truth not in {"ENTAILS", "CONTRADICTS"}:
            raise ValueError("ground_truth must be ENTAILS or CONTRADICTS.")
        if decision.get("label") == ground_truth:
            return None

        visual_output = visual_output or {}
        comparison = comparison or {}
        status = comparison.get("required_evidence_status")
        if status == "INSUFFICIENT_VISUAL_EVIDENCE" or not visual_output.get(
            "schema_complete", True
        ):
            return "visual_grounding_candidate"
        if status in {
            "SEMANTIC_REVIEW_REQUIRED",
            "GROUNDED_REVIEW_REQUIRED",
        }:
            return "cross_modal_reasoning_candidate"

        predicted_type = str(language_output.get("figurative_type", "")).lower()
        if predicted_type == "unknown":
            return "caption_analysis_candidate"
        return "arbiter_decision_candidate"

    def record_verified_error(
        self,
        visual_output,
        language_output,
        comparison,
        decision,
        ground_truth,
        phenomenon=None,
        metadata=None,
        evidence_ledger=None,
    ):
        """Log a gold-verified candidate without changing any prompt memory."""
        failure_type = self.classify_verified_error(
            language_output,
            decision,
            ground_truth,
            phenomenon,
            visual_output=visual_output,
            comparison=comparison,
        )
        event = {
            "failure_type": failure_type,
            "candidate_recorded": failure_type is not None,
            "agent1_memory_size": len(self.agent1_memory),
            "agent2_memory_size": len(self.agent2_memory),
        }
        if failure_type is not None:
            self.save_log(
                "verified_error_candidate",
                metadata,
                {
                    **event,
                    "ground_truth": ground_truth,
                    "prediction": decision.get("label"),
                    "figurative_type": language_output.get("figurative_type"),
                    "comparison_status": comparison.get("required_evidence_status"),
                    "visual_summary": visual_output.get("visual_description", ""),
                    "intended_meaning": language_output.get("intended_meaning", ""),
                    "evidence_ids": evidence_ids(evidence_ledger or []),
                },
            )
        return event

    # Backward-compatible facade. Without a verified label it intentionally
    # returns no feedback candidate and never alters prompt memory.
    def generate_feedback(
        self,
        visual_output,
        language_output,
        comparison,
        decision,
        ground_truth=None,
        phenomenon=None,
        metadata=None,
        apply_calibration=False,
        evidence_ledger=None,
    ):
        if ground_truth is None:
            return {
                "failure_type": None,
                "candidate_recorded": False,
                "agent1_memory_size": len(self.agent1_memory),
                "agent2_memory_size": len(self.agent2_memory),
                "arbiter_memory_size": len(self.arbiter_memory),
                "update_applied": False,
            }
        event = self.record_verified_error(
            visual_output,
            language_output,
            comparison,
            decision,
            ground_truth,
            phenomenon,
            metadata,
            evidence_ledger,
        )
        event["arbiter_memory_size"] = len(self.arbiter_memory)
        event["update_applied"] = False
        event["target_agent"] = None

        if not apply_calibration:
            return event
        if event.get("failure_type") is None:
            # Calibration is supervised by the external gold label. Correct
            # predictions are audit events, not errors to memorize.
            return event

        relation = RELATION_FOR_LABEL.get(ground_truth)
        case_failure_type = (
            "missed_grounded_support" if relation == "SUPPORT"
            else "missed_grounded_conflict"
        )
        case_context = {
            "visual_output": visual_output,
            "language_output": language_output,
            "comparison": comparison,
            "decision": decision,
            "evidence_ledger": evidence_ledger or [],
        }
        case_applied = self.add_verified_case(
            case_context,
            ground_truth,
            case_failure_type,
            self.CALIBRATED_RULES[case_failure_type],
            metadata,
        )
        if case_applied:
            event.update({
                "update_applied": True,
                "target_agent": "arbiter",
                "failure_type": case_failure_type,
                "agent1_memory_size": len(self.agent1_memory),
                "agent2_memory_size": len(self.agent2_memory),
                "arbiter_memory_size": len(self.arbiter_memory),
            })
            return event
        event["failure_type"] = "unverified_memory_candidate"
        return event

    def generate_feedback_batch(self, history):
        return [
            self.generate_feedback(
                sample["visual_output"],
                sample["language_output"],
                sample["comparison"],
                sample["decision"],
                sample.get("ground_truth"),
                sample.get("phenomenon"),
                sample.get("metadata"),
                evidence_ledger=sample.get("evidence_ledger"),
            )
            for sample in history
        ]
