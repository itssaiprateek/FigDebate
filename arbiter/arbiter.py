import time
import math
import re
try:
    import torch
except ImportError:
    torch = None

from utils.arbiter_parser import parse_arbiter_response
from utils.decision_scoring import (
    evidence_adjusted_confidence,
    position_balanced_relation_scores,
)
from engine.region_verifier import verify_region_pairs


class Arbiter:
    
    DEFAULT_PROMPT = """
You are the final decision-maker for a binary figurative image-caption task.
Use only the supplied evidence. ENTAILS needs direct visual support for the
caption's figurative meaning. CONTRADICTS needs direct visual conflict. Missing
evidence alone is not conflict. Never invent facts or return a third label.

Return exactly these short sections and nothing else:
Final Decision: ENTAILS or CONTRADICTS
Confidence: one decimal from 0 to 1; use <=0.35 for a close binary choice
Visual Support: one [VISUAL] fact, or None
Contradictions: one [VISUAL] fact, or None
Reasoning: one sentence citing [VISUAL], [CAPTION], or [COMPARATOR].
"""
    CONFIDENCE_THRESHOLD = 0.60
    FALLBACK_CONFIDENCE_CAP = 0.49


    def __init__(self, mistral_model, tokenizer, nli_verifier=None):
        if torch is None:
            raise RuntimeError(
                "The Arbiter requires PyTorch. Run check_environment.py."
            )

        self.model = mistral_model
        self.tokenizer = tokenizer
        self._nli_verifier = nli_verifier

        print("[Arbiter] Ready.")

        print("\n========== DEVICE MAP ==========")

        if hasattr(self.model, "hf_device_map"):

            for module, device in self.model.hf_device_map.items():
                print(f"{module:<40} {device}")

        else:
            print("Single device model.")

        print("================================\n")

    # ------------------------------------------------------------
    # Hard fallback so "label" is NEVER anything but ENTAILS or
    # CONTRADICTS, even if the model ignores instructions or the
    # output gets cut off. This is a safety net, not the primary path
    # -- if it fires often, that's a signal the prompt/token budget
    # still needs work, so it's returned alongside the flag below.
    # ------------------------------------------------------------

    @staticmethod
    def _normalize_label(raw_label, raw_response_text: str):
        """
        Accept only a label parsed from the dedicated Final Decision section.

        A malformed response must remain invalid here. Choosing a label from
        arbitrary words in the response would turn reasoning text into a
        fabricated prediction.
        """
        if raw_label in ("ENTAILS", "CONTRADICTS"):
            return raw_label, False

        return None, True

    @staticmethod
    def _summarize_visual(visual_grounding) -> str:
        if not isinstance(visual_grounding, dict):
            return str(visual_grounding)

        parts = []

        visible_text = visual_grounding.get("visible_text", []) or []
        if visible_text:
            parts.append(
                "Visible text:\n" + "; ".join(str(item) for item in visible_text[:5])
            )

        visual_facts = visual_grounding.get("visual_facts", []) or []
        if visual_facts:
            parts.append(
                "Visual facts:\n" + "; ".join(str(item) for item in visual_facts[:5])
            )

        visual_relations = visual_grounding.get("visual_relations", []) or []
        if visual_relations:
            parts.append(
                "Visual relations:\n"
                + "; ".join(str(item) for item in visual_relations[:5])
            )

        description = visual_grounding.get("visual_description", "")
        if description:
            parts.append(f"Description:\n{str(description)[:700]}")

        objects = visual_grounding.get("objects", []) or []
        if objects:
            if isinstance(objects, (list, tuple)):
                objects_text = ", ".join(str(item) for item in objects if str(item).strip())
            else:
                objects_text = str(objects)
            parts.append(f"Objects:\n{objects_text}")

        scene_type = visual_grounding.get("scene_type", "")
        if scene_type:
            parts.append(f"Scene:\n{scene_type}")

        symbolic_tone = visual_grounding.get("symbolic_tone", "")
        if symbolic_tone:
            parts.append(
                f"Tentative symbolic interpretation (not direct evidence):\n"
                f"{symbolic_tone}"
            )

        uncertain = visual_grounding.get("uncertain_observations", []) or []
        if uncertain:
            uncertain_text = "; ".join(str(item) for item in uncertain if str(item).strip())
            parts.append(f"Uncertain observations:\n{uncertain_text}")

        return "\n\n".join(parts) if parts else "No visual summary available."

    @staticmethod
    def _summarize_language(language_understanding) -> str:
        if not isinstance(language_understanding, dict):
            return str(language_understanding)

        parts = []

        surface = language_understanding.get("surface_meaning", "")
        if surface:
            parts.append(f"Surface meaning:\n{str(surface)[:350]}")

        figurative = language_understanding.get("figurative_type", "")
        if figurative:
            parts.append(f"Figurative type:\n{figurative}")

        intended = language_understanding.get("intended_meaning", "")
        if intended:
            parts.append(f"Intended meaning:\n{str(intended)[:350]}")

        proposition = language_understanding.get("caption_proposition", "")
        if proposition:
            parts.append(f"Caption proposition:\n{str(proposition)[:350]}")

        background = language_understanding.get("background_knowledge", "")
        if background:
            parts.append(f"Background knowledge:\n{str(background)[:250]}")

        non_literal = language_understanding.get("non_literal_expressions", []) or []
        if non_literal:
            non_literal_text = "; ".join(str(item) for item in non_literal if str(item).strip())
            parts.append(f"Non-literal expressions:\n{non_literal_text}")

        explicit_claims = language_understanding.get(
            "explicit_claims",
            [],
        ) or []
        if explicit_claims:
            explicit_text = "; ".join(
                str(item) for item in explicit_claims if str(item).strip()
            )
            parts.append(f"Caption-explicit claims:\n{explicit_text}")

        implicit_claims = language_understanding.get(
            "implicit_claims",
            [],
        ) or []
        if implicit_claims:
            implicit_text = "; ".join(
                str(item) for item in implicit_claims if str(item).strip()
            )
            parts.append(
                f"Caption implications (weaker evidence):\n{implicit_text}"
            )

        return "\n\n".join(parts) if parts else "No language summary available."

    @staticmethod
    def _summarize_comparison(comparison) -> str:
        if not isinstance(comparison, dict):
            return str(comparison)

        parts = []

        alignment = comparison.get("alignment_score")
        if alignment is not None:
            parts.append(f"Alignment score:\n{alignment}")

        supporting = (
            comparison.get("supporting_evidence")
            or comparison.get("supporting_points", [])
            or []
        )
        if supporting:
            if isinstance(supporting, (list, tuple)):
                supporting_text = "; ".join(str(item) for item in supporting if str(item).strip())
            else:
                supporting_text = str(supporting)
            parts.append(f"Supporting evidence:\n{supporting_text}")

        contradicting = (
            comparison.get("contradicting_evidence")
            or comparison.get("conflicting_points", [])
            or []
        )
        if contradicting:
            if isinstance(contradicting, (list, tuple)):
                conflicting_text = "; ".join(str(item) for item in contradicting if str(item).strip())
            else:
                conflicting_text = str(contradicting)
            parts.append(f"Contradicting evidence:\n{conflicting_text}")

        grounded_anchors = comparison.get("grounded_anchor_evidence", []) or []
        if grounded_anchors:
            if isinstance(grounded_anchors, (list, tuple)):
                anchors_text = "; ".join(
                    str(item) for item in grounded_anchors if str(item).strip()
                )
            else:
                anchors_text = str(grounded_anchors)
            parts.append(
                "Grounded anchors requiring polarity review:\n"
                f"{anchors_text}"
            )

        catalog = comparison.get("grounded_evidence_catalog", []) or []
        if catalog:
            catalog_lines = [
                f"[{item.get('id')}] {item.get('text')}"
                for item in catalog
                if item.get("id") and str(item.get("text", "")).strip()
            ]
            if catalog_lines:
                parts.append(
                    "Current-image evidence catalog:\n"
                    + "\n".join(catalog_lines[:12])
                )

        missing = (
            comparison.get("missing_evidence")
            or comparison.get("missing_visual_concepts", [])
            or []
        )
        if missing:
            if isinstance(missing, (list, tuple)):
                missing_text = "; ".join(str(item) for item in missing if str(item).strip())
            else:
                missing_text = str(missing)
            parts.append(f"Missing evidence:\n{missing_text}")

        unsupported = comparison.get("unsupported_inferences", []) or []
        if unsupported:
            if isinstance(unsupported, (list, tuple)):
                unsupported_text = "; ".join(str(item) for item in unsupported if str(item).strip())
            else:
                unsupported_text = str(unsupported)
            parts.append(f"Unsupported inferences:\n{unsupported_text}")

        neutral = comparison.get("neutral_notes", []) or []
        if neutral:
            if isinstance(neutral, (list, tuple)):
                neutral_text = "; ".join(str(item) for item in neutral if str(item).strip())
            else:
                neutral_text = str(neutral)
            parts.append(f"Neutral notes:\n{neutral_text}")

        review_questions = comparison.get("review_questions", []) or []
        if review_questions:
            parts.append(
                "Cross-modal checks:\n"
                + "; ".join(str(item) for item in review_questions[:3])
            )

        mediation_questions = comparison.get("mediation_questions", []) or []
        if mediation_questions:
            parts.append(
                "Independent mediator checks (advisory; verify from debate evidence):\n"
                + "; ".join(str(item) for item in mediation_questions[:6])
            )

        feedback_warning = comparison.get("feedback_warning", {}) or {}
        diagnostic_questions = feedback_warning.get(
            "diagnostic_questions", []
        ) or []
        if diagnostic_questions:
            parts.append(
                "Procedural memory checks (advisory; never label evidence):\n"
                + "; ".join(
                    str(item) for item in diagnostic_questions[:3]
                )
            )

        recommendation = comparison.get("recommendation")
        if recommendation:
            parts.append(f"Comparator recommendation:\n{recommendation}")

        required_evidence_status = comparison.get("required_evidence_status")
        if required_evidence_status:
            parts.append(
                f"Required evidence status:\n{required_evidence_status}"
            )

        if parts:
            parts.append(
                "Binary decision rule: direct evidence can support either label. "
                "Missing evidence is uncertainty, not contradiction by itself."
            )

        return "\n\n".join(parts) if parts else "No comparison summary available."

    @staticmethod
    def _to_spec_schema(parsed: dict, raw_response: str) -> dict:

        label, label_was_forced = Arbiter._normalize_label(
            parsed.get("label"), raw_response
        )

        confidence = parsed.get("confidence")
        confidence_is_valid = (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and math.isfinite(confidence)
            and 0.0 <= confidence <= 1.0
        )
        if not confidence_is_valid:
            confidence = None

        final_decision_valid = not label_was_forced and confidence_is_valid

        explanation_parts = [
            parsed.get("evidence_summary", ""),
            parsed.get("reasoning", ""),
        ]
        explanation = " ".join(
            p.strip() for p in explanation_parts if p and p.strip()
        )
        if not explanation:
            explanation = raw_response.strip()

        debate_needed = (
            not final_decision_valid
            or confidence < Arbiter.CONFIDENCE_THRESHOLD
        )

        return {
            "label": label,
            "explanation": explanation,
            "visual_support": parsed.get("visual_support", []),
            "contradictions": parsed.get("contradictions", []),
            "missing_evidence": parsed.get("missing_evidence", []),
            "confidence": confidence,
            "debate_needed": debate_needed,
            "decision_method": "primary",
            "evidence_sources": {
                "visual_support": "visual_grounding",
                "contradictions": "visual_grounding_or_comparator",
                "missing_evidence": "missing_or_uncertain_evidence",
            },
            # audit flag -- count how often this fires; frequent firing
            # means the prompt/token budget still needs tuning, not that
            # the fallback logic itself is wrong
            "_label_was_forced": label_was_forced,
            "_confidence_was_invalid": not confidence_is_valid,
            "_final_decision_valid": final_decision_valid,
            "_format_retry_used": False,
            "_binary_resolution_used": False,
            # kept for debugging / feedback_loop.py failure classification,
            # NOT part of the orchestrator contract
            "_internal": parsed,
        }

    @staticmethod
    def _build_binary_resolution_prompt(
        caption,
        visual_summary,
        language_summary,
        comparison_summary,
        debate_section,
    ):
        """Build the fixed context used to score the two allowed labels."""
        return f"""<s>[INST]
This is a binary figurative image-caption entailment task. Select the more
supported label from the two choices below. There is no neutral label.

Choose ENTAILS when visual evidence supports the caption's figurative meaning.
Choose CONTRADICTS when visual evidence conflicts with that meaning. Missing
evidence alone is not a contradiction; use a low-confidence best binary choice
when the evidence is close. Do not invent facts.

Caption:
{caption}

Visual evidence:
{visual_summary}

Caption analysis:
{language_summary}

Comparator evidence:
{comparison_summary}

{debate_section}

Return exactly one word: ENTAILS or CONTRADICTS.
[/INST]
"""

    @staticmethod
    def _build_relation_assessment_prompt(
        caption,
        visual_summary,
        language_summary,
        comparison_summary,
        debate_section,
    ):
        return f"""<s>[INST]
Compare the observed image evidence with the caption's intended meaning. The
figurative mechanism may occur in the image, the caption, or both. Use only the
supplied evidence and do not treat absent evidence as a visual conflict.
Do not assign visible text to a side, panel, object, person, or color unless
the supplied visual evidence explicitly makes that association.

Caption:
{caption}

Observed image evidence:
{visual_summary}

Caption analysis:
{language_summary}

Comparator notes:
{comparison_summary}

{debate_section}

Return exactly four concise lines. Do not output an entailment label.
Visual Evidence: one specific supplied fact, or insufficient observed evidence
Evidence IDs: one or more supplied IDs such as VF001, or NONE
Caption Meaning: the proposition that must be checked
Relation Analysis: explain whether the fact supports, conflicts with, or is
insufficient for that proposition, including figurative meaning when relevant
[/INST]
"""

    @staticmethod
    def _extract_evidence_ids(assessment):
        return list(dict.fromkeys(re.findall(
            r"\b(?:VF|VR|VT|VS|CS|CC|CA|TV|DV)\d{3}\b",
            str(assessment or "").upper(),
        )))

    @classmethod
    def _resolve_evidence_ids(cls, assessment, comparison):
        catalog = (comparison or {}).get("grounded_evidence_catalog", []) or []
        catalog_ids = {item.get("id") for item in catalog if item.get("id")}
        explicit = [
            item_id for item_id in cls._extract_evidence_ids(assessment)
            if item_id in catalog_ids
        ]
        if explicit:
            return explicit
        match = re.search(
            r"(?im)^\s*Visual Evidence\s*:\s*(.+?)\s*$",
            str(assessment or ""),
        )
        if not match:
            return []
        quote_tokens = set(re.findall(r"[a-z0-9]+", match.group(1).lower()))
        if len(quote_tokens) < 3:
            return []
        candidates = []
        for item in catalog:
            item_tokens = set(re.findall(
                r"[a-z0-9]+", str(item.get("text", "")).lower()
            ))
            overlap = len(quote_tokens & item_tokens)
            containment = overlap / min(len(quote_tokens), len(item_tokens)) \
                if item_tokens else 0.0
            if overlap >= 3 and containment >= 0.75:
                candidates.append((containment, overlap, item.get("id")))
        if not candidates:
            return []
        candidates.sort(key=lambda value: (-value[0], -value[1], value[2]))
        return [candidates[0][2]]

    @staticmethod
    def _relation_status(label, assessment, comparison, cited_evidence_ids):
        """Return the internal three-way evidence relation and deficiencies."""
        expected = {"ENTAILS": "SUPPORT", "CONTRADICTS": "CONFLICT"}.get(label)
        catalog = {
            item.get("id"): item
            for item in ((comparison or {}).get("grounded_evidence_catalog", []) or [])
            if item.get("id")
            and str(item.get("lifecycle_status", "ACTIVE")).upper()
            in {"ACTIVE", "RECONFIRMED"}
        }
        cited_directional = [
            catalog[item_id]
            for item_id in (cited_evidence_ids or [])
            if item_id in catalog
            and catalog[item_id].get("decision_grade", False)
            and catalog[item_id].get("relation") == expected
        ]
        support = (comparison or {}).get("supporting_evidence", []) or []
        conflict = (comparison or {}).get("contradicting_evidence", []) or []
        status = (comparison or {}).get("required_evidence_status")
        if cited_directional:
            relation_status = expected
        elif status == "SUPPORTED" and support and not conflict and expected == "SUPPORT":
            relation_status = "SUPPORT"
        elif status == "CONFLICTING" and conflict and not support and expected == "CONFLICT":
            relation_status = "CONFLICT"
        else:
            relation_status = "INSUFFICIENT"

        deficiencies = []
        if relation_status == "INSUFFICIENT":
            if not (comparison or {}).get("visual_schema_complete", True):
                deficiencies.append("MISSING_VISUAL_OBSERVATION")
            if (
                (comparison or {}).get("relation_binding_required", False)
                and not (comparison or {}).get("relation_binding_observed", False)
            ):
                deficiencies.append("UNRESOLVED_TEXT_BINDING")
            if status == "MIXED_VERIFIED_EVIDENCE" or (support and conflict):
                deficiencies.append("CONFLICTING_ACTIVE_EVIDENCE")
            if (
                (comparison or {}).get("has_symbolic_evidence", False)
                and not support and not conflict
            ):
                deficiencies.append("UNRESOLVED_SYMBOL_ATTACHMENT")
            if not deficiencies:
                deficiencies.append("INSUFFICIENT_DIRECTIONAL_EVIDENCE")
        return relation_status, deficiencies

    @staticmethod
    def _unopposed_verified_relation(comparison):
        """Return a unique decision-grade direction already in the ledger."""
        by_relation = {"SUPPORT": [], "CONFLICT": []}
        for item in (comparison or {}).get("grounded_evidence_catalog", []) or []:
            relation = item.get("relation")
            if (
                relation in by_relation
                and item.get("decision_grade", False)
                and str(item.get("lifecycle_status", "ACTIVE")).upper()
                in {"ACTIVE", "RECONFIRMED"}
                and item.get("id")
            ):
                by_relation[relation].append(item["id"])
        if by_relation["SUPPORT"] and not by_relation["CONFLICT"]:
            return "ENTAILS", by_relation["SUPPORT"]
        if by_relation["CONFLICT"] and not by_relation["SUPPORT"]:
            return "CONTRADICTS", by_relation["CONFLICT"]
        return None, []

    @staticmethod
    def _build_citation_retry_prompt(caption, comparison_summary):
        return f"""<s>[INST]
Select one observation from the supplied current-image evidence catalog. Copy
that observation without interpretation. Do not add a relation or a label.

Caption:
{caption}

Evidence catalog and comparator notes:
{comparison_summary}

Return exactly two lines:
Visual Evidence: exact copied catalog observation, or NONE
Evidence IDs: its exact catalog ID, or NONE
[/INST]
"""

    @staticmethod
    def _build_relation_choice_prompt(
        caption,
        visual_summary,
        language_summary,
        comparison_summary,
        assessment,
        support_first=True,
    ):
        support = "The observed image evidence supports the caption's intended meaning."
        conflict = "The observed image evidence conflicts with the caption's intended meaning."
        option_a, option_b = (support, conflict) if support_first else (conflict, support)
        return f"""<s>[INST]
Choose the better-supported semantic relation. Missing evidence alone does not
establish conflict. Consider literal text, spatial layout, incongruity, and the
caption's figurative meaning. Do not invent visual facts.
Do not bind OCR phrases to image regions or objects unless Agent 1 explicitly
reported that binding. An unbound phrase cannot prove either relation.
When Agent 1 supplies a targeted region-bound reinspection, treat it as the
current visual evidence and prefer it over an older unbound OCR summary. In a
comparison graphic, compare each object's outcome text with the corresponding
caption claim; an attempted action and its displayed outcome are not the same.

Caption:
{caption}

Observed image evidence:
{visual_summary}

Caption analysis:
{language_summary}

Comparator notes:
{comparison_summary}

Evidence assessment:
{assessment}

Choice A: {option_a}
Choice B: {option_b}

Answer with exactly one letter: A or B.
[/INST]
"""

    def _verify_targeted_region_relation(
        self, caption, region_pairs, claim_relation=None
    ):
        """Verify region pairs without using generic text NLI as visual proof."""
        started = time.time()
        verification = verify_region_pairs(region_pairs, claim_relation or {})
        if not verification.get("resolved", False):
            raise ValueError(
                "Targeted region relation abstained: "
                + verification.get("reason", "unresolved")
            )
        label = verification["label"]
        raw_confidence = float(verification["confidence"])
        scores = {
            "ENTAILS": round(
                raw_confidence if label == "ENTAILS" else 1.0 - raw_confidence,
                6,
            ),
            "CONTRADICTS": round(
                raw_confidence if label == "CONTRADICTS" else 1.0 - raw_confidence,
                6,
            ),
            "NEUTRAL": 0.0,
            "verification_method": verification["method"],
        }
        response = (
            "Deterministic region-pair verification: "
            f"{verification['reason']}; pairs={verification.get('pair_results', [])}"
        )
        return (
            label,
            raw_confidence,
            scores,
            response,
            time.time() - started,
            verification,
        )

    @staticmethod
    def _build_format_retry_prompt(
        caption,
        visual_summary,
        language_summary,
        comparison_summary,
        debate_section,
    ):
        """Build a small retry that repairs format without redoing reasoning."""
        return f"""<s>[INST]
Your previous decision did not use one of the two labels required by this
binary figurative image-caption entailment task.

Using the evidence below, choose the more plausible label. Missing evidence
alone is not a contradiction. Do not answer NEUTRAL, UNCERTAIN, or anything
other than the required format.

Caption:
{caption}

Visual evidence:
{visual_summary}

Caption analysis:
{language_summary}

Comparator evidence:
{comparison_summary}

{debate_section}

Return exactly:
Final Decision:
ENTAILS or CONTRADICTS
Confidence:
0.00
[/INST]
"""

    def _generate_response(self, prompt, max_new_tokens):
        """Generate one response and return its text with generation time."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
        ).to(self.model.device)

        start = time.time()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[:, inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ).strip()
        return response, time.time() - start

    def _completion_log_scores(self, prompt, candidates):
        """Return length-normalized log likelihoods for candidate completions."""
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"][0].tolist()
        scores = {}

        with torch.inference_mode():
            for label in candidates:
                encoded = self.tokenizer(
                    prompt + label,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(self.model.device)
                full_ids = encoded["input_ids"][0].tolist()
                prefix_length = len(prompt_ids)

                if full_ids[:prefix_length] != prompt_ids:
                    raise ValueError("Tokenizer changed the binary-resolution prompt prefix.")

                label_ids = full_ids[prefix_length:]
                if not label_ids:
                    raise ValueError("Binary-resolution label tokenization is empty.")

                logits = self.model(**encoded).logits[0]
                log_probs = torch.log_softmax(logits, dim=-1)
                score = sum(
                    log_probs[prefix_length + offset - 1, token_id].item()
                    for offset, token_id in enumerate(label_ids)
                ) / len(label_ids)
                scores[label] = score

        return scores

    def _label_log_scores(self, prompt):
        return self._completion_log_scores(prompt, ("ENTAILS", "CONTRADICTS"))

    def _score_semantic_relations(
        self,
        caption,
        visual_summary,
        language_summary,
        comparison_summary,
        assessment,
    ):
        """Score support/conflict twice with reversed option positions."""
        forward_prompt = self._build_relation_choice_prompt(
            caption,
            visual_summary,
            language_summary,
            comparison_summary,
            assessment,
            support_first=True,
        )
        reversed_prompt = self._build_relation_choice_prompt(
            caption,
            visual_summary,
            language_summary,
            comparison_summary,
            assessment,
            support_first=False,
        )
        forward_scores = self._completion_log_scores(forward_prompt, ("A", "B"))
        reversed_scores = self._completion_log_scores(reversed_prompt, ("A", "B"))
        balanced = position_balanced_relation_scores(forward_scores, reversed_scores)
        label = (
            "ENTAILS"
            if balanced["ENTAILS"] >= balanced["CONTRADICTS"]
            else "CONTRADICTS"
        )
        confidence = max(balanced["ENTAILS"], balanced["CONTRADICTS"])
        scores = {
            "ENTAILS": round(balanced["ENTAILS"], 6),
            "CONTRADICTS": round(balanced["CONTRADICTS"], 6),
            "support_log_score": round(balanced["support_log_score"], 6),
            "conflict_log_score": round(balanced["conflict_log_score"], 6),
            "forward_A": round(float(forward_scores["A"]), 6),
            "forward_B": round(float(forward_scores["B"]), 6),
            "reversed_A": round(float(reversed_scores["A"]), 6),
            "reversed_B": round(float(reversed_scores["B"]), 6),
        }
        return label, confidence, scores

    def _score_binary_labels(self, prompt):
        """Score labels with a neutral-context correction for label-token priors."""
        scores = self._label_log_scores(prompt)
        neutral_prompt = """<s>[INST]
Return exactly one word: ENTAILS or CONTRADICTS.
[/INST]
"""
        neutral_scores = self._label_log_scores(neutral_prompt)
        calibrated = {
            label: scores[label] - neutral_scores[label]
            for label in ("ENTAILS", "CONTRADICTS")
        }
        score_tensor = torch.tensor([
            calibrated["ENTAILS"],
            calibrated["CONTRADICTS"],
        ])
        probabilities = torch.softmax(score_tensor, dim=0).tolist()
        label = "ENTAILS" if probabilities[0] >= probabilities[1] else "CONTRADICTS"
        return label, float(max(probabilities)), {
            "ENTAILS": round(float(probabilities[0]), 6),
            "CONTRADICTS": round(float(probabilities[1]), 6),
            "raw_entails": round(float(scores["ENTAILS"]), 6),
            "raw_contradicts": round(float(scores["CONTRADICTS"]), 6),
        }

    # ======================================================
    # Main Reasoning Function
    # ======================================================

    def analyze(
        self,
        caption,
        visual_grounding,
        language_understanding,
        comparison,
        agent1_critique=None,
        agent2_critique=None,
        previous_decision=None,
        feedback=None,
    ):

        debate_section = ""

        if agent1_critique is not None and agent2_critique is not None:

            advocates = agent1_critique.get("advocates", {}) or {}
            entails_case = advocates.get("entails", []) or []
            contradicts_case = advocates.get("contradicts", []) or []
            neutral_case = advocates.get("neutral_or_anchor", []) or []

            debate_section = f"""

==================================================

Debate Review

This is an independent review. The previous label is intentionally hidden.
Review both positions against the supplied evidence and reconstruct the answer.

Cross-examine both positions. Reject an argument that cites no current-image
evidence, treats missing support as conflict, or relies only on thematic
similarity. A longer argument is not stronger than a grounded one.

ENTAILS Advocate Evidence:
{entails_case if entails_case else "No decision-grade SUPPORT evidence submitted."}

CONTRADICTS Advocate Evidence:
{contradicts_case if contradicts_case else "No decision-grade CONFLICT evidence submitted."}

Neutral/Anchor Evidence Requiring Interpretation:
{neutral_case[:6] if neutral_case else "None."}

If a critique correctly identifies an unsupported claim, a missing visual
conflict, or a misunderstanding of the caption, do not retain the previous
decision unless the supplied evidence independently supports it.

Independent Visual Reviewer

Recommendation:
{agent1_critique.get("recommendation", "ABSTAIN")}

Observed Entity:
{agent1_critique.get("observed_entity", "None")}

Observed State:
{agent1_critique.get("observed_state", "None")}

Image Region:
{agent1_critique.get("image_region", "None")}

Claim Relation:
{agent1_critique.get("claim_relation", "UNRESOLVED")}

Reason:
{agent1_critique["reason"]}

-------------------------

Independent Linguistic Claim Auditor

Stance:
{agent2_critique["stance"]}

Visual condition required for ENTAILS:
{agent2_critique.get("support_requirement", "Unavailable")}

Visual condition required for CONTRADICTS:
{agent2_critique.get("conflict_requirement", "Unavailable")}

Figurative mechanism:
{agent2_critique.get("figurative_mechanism", "Unavailable")}

Caption ambiguity:
{agent2_critique.get("ambiguity", "None")}

Reason:
{agent2_critique["reason"]}

==================================================
"""
        else:
            debate_section = ""

        visual_summary = self._summarize_visual(visual_grounding)
        targeted_region_reason = None
        targeted_region_pairs = []
        if (
            isinstance(agent1_critique, dict)
            and agent1_critique.get("specific_evidence", False)
            and str(agent1_critique.get("reason", "")).strip()
        ):
            review_method = str(
                agent1_critique.get("review_method", "targeted_reinspection")
            ).strip()
            targeted_review = (
                f"[TARGETED VISUAL REINSPECTION - {review_method}] "
                f"Entity: {agent1_critique.get('observed_entity', 'unknown')}; "
                f"state: {agent1_critique.get('observed_state', 'unknown')}; "
                f"region: {agent1_critique.get('image_region', 'unknown')}; "
                f"relation: {agent1_critique.get('claim_relation', 'UNRESOLVED')}. "
                f"{agent1_critique['reason']}"
            )
            if review_method == "region_ocr":
                # This review was requested because the first pass could not
                # bind OCR text to regions. Mixing both summaries would retain
                # the exact ambiguity that the targeted review resolved.
                visual_summary = targeted_review
                targeted_region_reason = str(agent1_critique["reason"]).strip()
                targeted_region_pairs = agent1_critique.get("region_pairs", []) or []
            else:
                visual_summary = f"{targeted_review}\n{visual_summary}"
        language_summary = self._summarize_language(language_understanding)
        comparison_summary = self._summarize_comparison(comparison)
        feedback_section = (
            "Calibrated error-avoidance guidance:\n"
            f"{feedback.strip()}\n"
            if isinstance(feedback, str) and feedback.strip()
            else ""
        )
        comparison_and_guidance = "\n\n".join(
            part for part in (comparison_summary, feedback_section) if part
        )

        prompt = f"""<s>[INST]

{self.DEFAULT_PROMPT}

==================================================

Original Caption

{caption}

==================================================

Visual Grounding

{visual_summary}

==================================================

Language Understanding

{language_summary}

==================================================

Comparator Output

{comparison_summary}

{debate_section}

{feedback_section}

==================================================

Binary completion rule before writing the answer:
- You must output ENTAILS or CONTRADICTS, never UNCERTAIN or a third label.
- A high-confidence CONTRADICTS decision needs a concrete [VISUAL] conflict;
  missing evidence alone is not a conflict.
- A high-confidence ENTAILS decision needs a concrete [VISUAL] support.
- If evidence is incomplete, choose the more plausible binary label with
  confidence at or below 0.35. Do not say that no decision can be made.
- When Comparator Output contains SUPPORTED or CONFLICTING evidence, address
  that exact evidence rather than replacing it with a generic absence claim.

[/INST]
"""

        print("\n========== Arbiter ==========")

        if debate_section:
            print("[Arbiter] Debate revision round.")
        else:
            print("[Arbiter] Initial reasoning round.")

        if targeted_region_reason and targeted_region_pairs:
            scoring_started = time.time()
            try:
                label, raw_confidence, scores, response, elapsed, verification = (
                    self._verify_targeted_region_relation(
                        caption,
                        targeted_region_pairs,
                        language_understanding.get("claim_relation", {}),
                    )
                )
            except (RuntimeError, ValueError) as error:
                print(f"[Arbiter WARNING] Targeted region verification failed: {error}")
            else:
                evidence_quality = comparison.get("evidence_quality", 0.5)
                confidence = evidence_adjusted_confidence(
                    raw_confidence, evidence_quality
                )
                evidence_item = (
                    "[VISUAL][REGION_OCR] " + targeted_region_reason
                )
                visual_support = [evidence_item] if label == "ENTAILS" else []
                contradictions = (
                    [evidence_item] if label == "CONTRADICTS" else []
                )
                relation_word = (
                    "supports" if label == "ENTAILS" else "conflicts with"
                )
                spec_output = {
                    "label": label,
                    "explanation": (
                        "A targeted region-bound review found that the paired "
                        f"image outcomes {relation_word} the caption claim."
                    ),
                    "visual_support": visual_support,
                    "contradictions": contradictions,
                    "missing_evidence": [],
                    "confidence": confidence,
                    "debate_needed": confidence < self.CONFIDENCE_THRESHOLD,
                    "decision_method": "targeted_region_verifier",
                    "evidence_sources": {
                        "visual_support": "targeted_region_ocr",
                        "contradictions": "targeted_region_ocr",
                        "missing_evidence": "targeted_region_ocr",
                    },
                    "_label_was_forced": False,
                    "_confidence_was_invalid": False,
                    "_final_decision_valid": True,
                    "_format_retry_used": False,
                    "_binary_resolution_used": False,
                    "_retry_attempted": False,
                    "_retry_failed": False,
                    "_binary_resolution_scores": scores,
                    "_binary_resolution_raw_confidence": raw_confidence,
                    "_targeted_region_verification": verification,
                    "_relation_status": verification.get(
                        "evidence_relation", "INSUFFICIENT"
                    ),
                    "_revision_status": "DIRECTIONAL_PROPOSAL",
                    "_deficiencies": [],
                    "_directional_proposal": True,
                    "_evidence_quality": evidence_quality,
                    "_arbiter_assessment": response,
                    "_timing": {
                        "primary_seconds": round(elapsed, 4),
                        "assessment_seconds": 0.0,
                        "format_retry_seconds": 0.0,
                        "binary_resolution_seconds": round(elapsed, 4),
                        "total_seconds": round(time.time() - scoring_started, 4),
                    },
                }
                spec_output["_primary_decision"] = dict(spec_output)
                print(
                    "[Arbiter] Targeted region verifier: "
                    f"{label} (SUPPORT={scores['ENTAILS']:.3f}, "
                    f"CONFLICT={scores['CONTRADICTS']:.3f})."
                )
                return spec_output

        # First produce a short evidence audit, then score semantic support and
        # conflict with both A/B orders. Reversing the option order cancels a
        # stable preference for a label token or answer position.
        scoring_started = time.time()
        assessment_prompt = self._build_relation_assessment_prompt(
            caption,
            visual_summary,
            language_summary,
            comparison_and_guidance,
            debate_section,
        )
        assessment = "No generated assessment was available."
        assessment_seconds = 0.0
        citation_retry_response = ""
        citation_retry_seconds = 0.0
        citation_retry_used = False
        try:
            assessment, assessment_seconds = self._generate_response(
                assessment_prompt, max_new_tokens=112
            )
        except RuntimeError as error:
            print(f"[Arbiter WARNING] Evidence assessment failed: {error}")
        cited_evidence_ids = self._resolve_evidence_ids(assessment, comparison)
        if (
            not cited_evidence_ids
            and comparison.get("grounded_evidence_catalog")
        ):
            citation_retry_used = True
            try:
                citation_retry_response, citation_retry_seconds = (
                    self._generate_response(
                        self._build_citation_retry_prompt(
                            caption, comparison_and_guidance
                        ),
                        max_new_tokens=48,
                    )
                )
                cited_evidence_ids = self._resolve_evidence_ids(
                    citation_retry_response, comparison
                )
            except RuntimeError as error:
                print(f"[Arbiter WARNING] Evidence citation retry failed: {error}")
        try:
            label, raw_confidence, scores = self._score_semantic_relations(
                caption,
                visual_summary,
                language_summary,
                comparison_and_guidance,
                assessment,
            )
        except (RuntimeError, ValueError) as error:
            print(f"[Arbiter WARNING] Primary binary scoring failed: {error}")
        else:
            scoring_elapsed = time.time() - scoring_started
            primary_elapsed = max(
                0.0, scoring_elapsed - citation_retry_seconds
            )
            evidence_status = comparison.get("required_evidence_status")
            evidence_quality = comparison.get("evidence_quality", 0.5)
            confidence = evidence_adjusted_confidence(
                raw_confidence, evidence_quality
            )
            semantic_scored_label = label
            verified_label, verified_ids = self._unopposed_verified_relation(
                comparison
            )
            decision_method = "position_balanced_semantic"
            if verified_label:
                # Scored language preferences cannot overrule an unopposed
                # independently verified relation.  The Arbiter remains the
                # final selector and records the exact proof it relied on.
                label = verified_label
                cited_evidence_ids = list(verified_ids)
                confidence = max(confidence, 0.72)
                decision_method = "verified_relation_arbitration"
            relation_status, deficiencies = self._relation_status(
                label, assessment, comparison, cited_evidence_ids
            )
            unconstrained_label = semantic_scored_label
            revision_status = "DIRECTIONAL_PROPOSAL"
            if relation_status == "INSUFFICIENT":
                confidence = min(confidence, 0.35)
                revision_status = "NO_REVISION_INSUFFICIENT_EVIDENCE"
                if previous_decision and previous_decision.get("label") in {
                    "ENTAILS", "CONTRADICTS"
                }:
                    label = previous_decision["label"]
            visual_support = comparison.get("supporting_evidence", []) or []
            contradictions = comparison.get("contradicting_evidence", []) or []
            explanation = assessment.strip() or (
                "[COMPARATOR] No generated evidence assessment was available."
            )
            spec_output = {
                "label": label,
                "explanation": explanation,
                "visual_support": visual_support,
                "contradictions": contradictions,
                "missing_evidence": comparison.get("missing_evidence", []) or [],
                "confidence": confidence,
                "debate_needed": confidence < self.CONFIDENCE_THRESHOLD,
                "decision_method": decision_method,
                "evidence_sources": {
                    "visual_support": "visual_grounding_or_comparator",
                    "contradictions": "visual_grounding_or_comparator",
                    "missing_evidence": "missing_or_uncertain_evidence",
                },
                "_label_was_forced": False,
                "_confidence_was_invalid": False,
                "_final_decision_valid": True,
                "_format_retry_used": False,
                "_binary_resolution_used": False,
                "_retry_attempted": False,
                "_retry_failed": False,
                "_binary_resolution_scores": scores,
                "_binary_resolution_raw_confidence": raw_confidence,
                "_unconstrained_proposed_label": unconstrained_label,
                "_verified_relation_override": bool(verified_label),
                "_relation_status": relation_status,
                "_revision_status": revision_status,
                "_deficiencies": deficiencies,
                "_directional_proposal": relation_status in {
                    "SUPPORT", "CONFLICT"
                },
                "_evidence_quality": evidence_quality,
                "_arbiter_assessment": assessment,
                "_model_cited_evidence_ids": cited_evidence_ids,
                "_citation_retry_used": citation_retry_used,
                "_citation_retry_response": citation_retry_response,
                "_timing": {
                    "primary_seconds": round(primary_elapsed, 4),
                    "assessment_seconds": round(assessment_seconds, 4),
                    "citation_retry_seconds": round(
                        citation_retry_seconds, 4
                    ),
                    "format_retry_seconds": 0.0,
                    "binary_resolution_seconds": 0.0,
                    "total_seconds": round(scoring_elapsed, 4),
                },
            }
            spec_output["_primary_decision"] = dict(spec_output)
            print(
                "[Arbiter] Position-balanced semantic scores: "
                f"ENTAILS={scores['ENTAILS']:.3f}, "
                f"CONTRADICTS={scores['CONTRADICTS']:.3f}."
            )
            return spec_output

        response, elapsed = self._generate_response(prompt, max_new_tokens=112)

        print(f"Generation Time : {elapsed:.2f} sec")

        print("\n================ RAW ARBITER OUTPUT ================\n")
        print(response)
        print("\n====================================================\n")

        parsed = parse_arbiter_response(response)
        primary_output = self._to_spec_schema(parsed, response)
        primary_output["_retry_attempted"] = False
        primary_output["_retry_failed"] = False
        primary_output["_raw_primary_response"] = response
        primary_output["_timing"] = {
            "primary_seconds": round(elapsed, 4),
            "binary_resolution_seconds": 0.0,
            "total_seconds": round(elapsed, 4),
        }

        # Preserve the complete primary result even when a binary resolution
        # is needed, so later analysis never loses the original evidence.
        primary_snapshot = dict(primary_output)
        spec_output = primary_output

        if not primary_output["_final_decision_valid"]:
            print("[Arbiter] Invalid primary format. Attempting one strict format retry.")
            retry_prompt = self._build_format_retry_prompt(
                caption,
                visual_summary,
                language_summary,
                comparison_summary,
                debate_section,
            )
            retry_response, retry_elapsed = self._generate_response(
                retry_prompt,
                max_new_tokens=32,
            )
            retry_parsed = parse_arbiter_response(retry_response)
            retry_output = self._to_spec_schema(retry_parsed, retry_response)
            retry_output["_retry_attempted"] = True
            retry_output["_retry_failed"] = not retry_output["_final_decision_valid"]
            retry_output["_raw_format_retry_response"] = retry_response
            retry_output["_timing"] = {
                "primary_seconds": round(elapsed, 4),
                "format_retry_seconds": round(retry_elapsed, 4),
                "binary_resolution_seconds": 0.0,
                "total_seconds": round(elapsed + retry_elapsed, 4),
            }
            spec_output = retry_output

            if retry_output["_final_decision_valid"]:
                retry_output["decision_method"] = "format_retry"
                retry_output["_format_retry_used"] = True
                retry_output["_binary_resolution_used"] = False
                spec_output = retry_output

        if not spec_output["_final_decision_valid"]:
            print(
                "[Arbiter] Format retry failed. Resolving between the two "
                "allowed labels by scoring both completions."
            )
            resolution_start = time.time()
            resolution_prompt = self._build_binary_resolution_prompt(
                caption,
                visual_summary,
                language_summary,
                comparison_summary,
                debate_section,
            )

            try:
                label, confidence, scores = self._score_binary_labels(
                    resolution_prompt
                )
            except (RuntimeError, ValueError) as error:
                spec_output["_retry_attempted"] = True
                spec_output["_retry_failed"] = True
                spec_output["_binary_resolution_error"] = str(error)
                print(f"[Arbiter WARNING] Binary resolution failed: {error}")
            else:
                resolution_elapsed = time.time() - resolution_start
                spec_output = dict(spec_output)
                raw_confidence = confidence
                confidence = min(confidence, self.FALLBACK_CONFIDENCE_CAP)
                spec_output.update({
                    "label": label,
                    "confidence": confidence,
                    "debate_needed": confidence < self.CONFIDENCE_THRESHOLD,
                    "decision_method": "binary_resolution",
                    "_label_was_forced": False,
                    "_confidence_was_invalid": False,
                    "_final_decision_valid": True,
                    "_binary_resolution_used": True,
                    "_retry_attempted": True,
                    "_retry_failed": False,
                    "_binary_resolution_scores": scores,
                    "_binary_resolution_raw_confidence": raw_confidence,
                    "_timing": {
                        "primary_seconds": round(elapsed, 4),
                        "format_retry_seconds": round(
                            spec_output.get("_timing", {}).get("format_retry_seconds", 0.0),
                            4,
                        ),
                        "binary_resolution_seconds": round(resolution_elapsed, 4),
                        "total_seconds": round(
                            spec_output.get("_timing", {}).get("total_seconds", elapsed)
                            + resolution_elapsed,
                            4,
                        ),
                    },
                })
                print(
                    "[Arbiter] Binary-resolution scores: "
                    f"ENTAILS={scores['ENTAILS']:.3f}, "
                    f"CONTRADICTS={scores['CONTRADICTS']:.3f}."
                )

        spec_output["_primary_decision"] = primary_snapshot
        return spec_output
