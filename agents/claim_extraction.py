import time
import re
import json
from engine.runtime_accounting import record_generation
from engine.structured_decoder import DECODER_ID
from engine.output_contracts import CORE, INTERPRETATION, prefix_constraint, validate_shape, object_schema
try:
    import torch
except ImportError:
    torch = None

from utils.claim_parser import parse_claim_response
from engine.claim_contract import attach_claim_contract
from engine.reasoning_schema import attach_reasoning_profile
from engine.relation_schema import normalize_relation_family


class ClaimExtractionAgent:
    """
    Agent 2 : Language Understanding Agent

    Input:
        Caption only

    Output (FIXED schema Shrihan's orchestrator expects):
        surface_meaning (str), figurative_type (str, one of
        metaphor/sarcasm/humor/literal), intended_meaning (str),
        background_knowledge (str)

    This agent NEVER reasons about the image.
    """

    DEFAULT_PROMPT = """
Extract the caption's EXPRESSED claim using only its words. Do not imagine an
image or substitute sarcastic intended meaning for the expressed proposition.
Preserve entities, negation, numbers, comparison direction and panel/time scope.
Distinguish an event from its rate, degree or ease. Every modifier matters.
Return these short headings; use None for genuinely absent arguments:
Caption Proposition:
Claim Subject:
Claim Predicate:
Claim Object:
Claim Source:
Claim Target:
Asserted Property:
Relation Family: trajectory, pace, outcome, sentiment, safety, trust, association, quantity, or other
Reasoning Requirement: visual, text_binding, background, normative, or mixed
Comparison Direction:
Time or Panel Scope:
"""
    INTERPRETATION_PROMPT = """
Propose a figurative interpretation of this caption ONLY. Do not imagine the
image or decide entailment. Interpretation is a hypothesis, not a replacement
for the expressed claim. Use None where inapplicable. Return short headings:
Figurative Type: sarcasm, metaphor, humor, or literal
Linguistic Cue:
Polarity Reversal: yes, no, or unclear
Literal Meaning:
Underlying Message:
Alternative Interpretation:
Literal Polarity: positive, negative, neutral, mixed, or unclear
Intended Polarity: positive, negative, neutral, mixed, or unclear
Structural Reasoning Type: direct_state, ocr_region_binding, comparative_layout, temporal_causal_sequence, quoted_statement_and_reaction, affective_scene, symbol_attachment, background_required, or unresolved
Background Knowledge:
Confidence:
"""
    FIGURATIVE_TYPES = ("sarcasm", "metaphor", "humor", "literal")
    CLAIM_FRAME_FIELDS = (
        "caption_proposition", "claim_subject", "claim_predicate",
        "claim_object", "claim_source", "claim_target",
        "asserted_property", "relation_family", "expected_visual_state",
        "opposite_visual_state", "reasoning_requirement",
        "background_knowledge", "structural_reasoning_type",
        "figurative_mechanism_candidates", "literal_polarity",
        "intended_polarity", "comparison_direction", "evaluation_target",
        "time_or_panel_scope",
    )

    def __init__(self, mistral_model, tokenizer, research_decomposition=False):
        if torch is None:
            raise RuntimeError(
                "Agent 2 requires PyTorch. Run check_environment.py."
            )

        self.model = mistral_model
        self.tokenizer = tokenizer
        self.research_decomposition = bool(research_decomposition)

        print("[Agent2] Ready.")

        print("\n========== DEVICE MAP ==========")

        if hasattr(self.model, "hf_device_map"):

            for module, device in self.model.hf_device_map.items():
                print(f"{module:<40} {device}")

        else:
            print("Single device model.")

        print("================================\n")

    def _chat_prompt(self, instruction):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction.strip()}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @staticmethod
    def _normalize_figurative_type(raw_value: str, linguistic_notes=None):
        lines = [line.strip() for line in (raw_value or "").splitlines() if line.strip()]
        first_line = lines[0].lower().rstrip(".!,;:") if lines else ""

        if first_line in ClaimExtractionAgent.FIGURATIVE_TYPES:
            return first_line, False

        match = re.fullmatch(
            r"figurative\s+type\s*:\s*(sarcasm|metaphor|humor|literal)[.!;,]?",
            first_line,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).lower(), False

        # The model did not provide a usable figurative type. Preserve this
        # as an auditable failure instead of fabricating a metaphor label.
        # "unknown" is safe for downstream string processing and is tracked
        # through _figurative_type_was_guessed.
        return "unknown", True

    @classmethod
    def _parse_type_retry(cls, response: str):
        """Accept only one explicit type from the short recovery response."""
        figurative_type, was_guessed = cls._normalize_figurative_type(response)
        return None if was_guessed else figurative_type

    @staticmethod
    def _normalize_caption_polarity(value, asserted_property, intended_meaning):
        declared = str(value or "").strip().lower()
        text = " ".join(
            str(item or "").lower()
            for item in (asserted_property,)
        )
        negative = bool(re.search(
            r"\b(rotten|corrupt|negative|bad|hate|disliked|sad|angry|"
            r"dangerous|unsafe|failed|broken|decline|loss)\b",
            text,
        ))
        positive = bool(re.search(
            r"\b(good|positive|love|loved|happy|safe|successful|growth|"
            r"improve|trustworthy)\b",
            text,
        ))
        if negative and not positive:
            return "negative"
        if positive and not negative:
            return "positive"
        return declared if declared in {
            "positive", "negative", "neutral", "mixed", "unclear"
        } else "unclear"

    @record_generation("Mistral-7B-figurative-type")
    def _retry_figurative_type(self, caption: str):
        prompt = f"""
Classify the caption itself. Return exactly one lowercase word: sarcasm,
metaphor, humor, or literal. Do not infer a device that could exist only in an
unseen image and do not add an explanation.

Caption:
{caption}
"""
        inputs = self.tokenizer(
            self._chat_prompt(prompt),
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[:, inputs["input_ids"].shape[1]:]
        self._last_generation_diagnostics = {
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "generated_tokens": int(generated.shape[-1]),
        }
        response = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ).strip()
        return self._parse_type_retry(response), response

    def _score_figurative_type(self, caption: str):
        """Resolve only an invalid type with calibrated label likelihoods.

        This avoids treating a malformed generated phrase such as "none" as a
        valid dataset class. Valid primary classifications never take this path.
        """
        prompt = f"""
Classify the caption itself as sarcasm, metaphor, humor, or literal. The
figurative phenomenon may exist only in an unseen image, so use literal when
the caption contains no clear figurative device. Answer with one lowercase word.

Caption:
{caption}
"""
        prompt = self._chat_prompt(prompt)
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=True, return_tensors="pt"
        )["input_ids"][0].tolist()
        neutral_prompt = """
Answer with one lowercase word: sarcasm, metaphor, humor, or literal.
"""
        neutral_prompt = self._chat_prompt(neutral_prompt)

        def label_scores(prefix):
            prefix_ids = self.tokenizer(
                prefix, add_special_tokens=True, return_tensors="pt"
            )["input_ids"][0].tolist()
            scores = {}
            with torch.inference_mode():
                for label in self.FIGURATIVE_TYPES:
                    encoded = self.tokenizer(
                        prefix + label,
                        add_special_tokens=True,
                        return_tensors="pt",
                    ).to(self.model.device)
                    full_ids = encoded["input_ids"][0].tolist()
                    if full_ids[:len(prefix_ids)] != prefix_ids:
                        raise ValueError("Tokenizer changed the figurative-type prompt prefix.")
                    label_ids = full_ids[len(prefix_ids):]
                    if not label_ids:
                        raise ValueError("Figurative-type label tokenization is empty.")
                    logits = self.model(**encoded).logits[0]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    scores[label] = sum(
                        log_probs[len(prefix_ids) + offset - 1, token_id].item()
                        for offset, token_id in enumerate(label_ids)
                    ) / len(label_ids)
            return scores

        # The initial tokenization also validates the exact prompt path before
        # expensive scoring starts.
        if not prompt_ids:
            raise ValueError("Figurative-type prompt tokenization is empty.")
        raw_scores = label_scores(prompt)
        neutral_scores = label_scores(neutral_prompt)
        calibrated = {
            label: raw_scores[label] - neutral_scores[label]
            for label in self.FIGURATIVE_TYPES
        }
        probabilities = torch.softmax(
            torch.tensor([calibrated[label] for label in self.FIGURATIVE_TYPES]),
            dim=0,
        ).tolist()
        scores = {
            label: round(float(probability), 6)
            for label, probability in zip(self.FIGURATIVE_TYPES, probabilities)
        }
        selected = max(scores, key=scores.get)
        return selected, float(scores[selected]), scores

    def _retry_structured_claim(self, caption, invalid_groups=None, defective_fields=None, verification=None):
        instruction = (
            "Repair only the fields requested by the JSON schema from the original caption. Preserve "
            "expressed polarity, entity roles, quantities and scope. Do not "
            "imagine an image. Use JSON null for absent optional arguments and [] for absent lists."
        )
        requested = [field for field in (defective_fields or []) if field in CORE["properties"]]
        if verification and verification.get("claim_graph_fingerprint") and verification.get("input_errors"):
            if "INVALID_CORE_TYPES" not in verification["input_errors"]:
                # Graph defects belong to the graph repairer, not every CORE field.
                return {}, "", 0.0
        if verification and verification.get("claim_graph_fingerprint"):
            requested = [field for field in requested if field not in {"expected_visual_state", "opposite_visual_state"}]
            if not requested:
                return {}, "", 0.0
        if not requested:
            # Structural failures may precede a usable semantic audit. Repair
            # the affected typed group rather than returning to legacy headings.
            groups = {
                "immutable_proposition": ["caption_proposition", "claim_subject", "claim_predicate", "claim_object", "claim_source", "claim_target", "negation", "quantities", "claim_modifiers"],
                "relation": ["relation_family", "expected_visual_state", "opposite_visual_state"],
                "reasoning_profile": ["reasoning_requirement", "comparison_direction", "time_or_panel_scope"],
            }
            requested = list(dict.fromkeys(field for group in (invalid_groups or ["immutable_proposition", "relation"])
                for field in groups.get(group, [])))
        if not requested:
            raise ValueError("No identified defective claim fields to repair")
        if verification:
            instruction += "\nSource verification (fallible evidence; preserve unaffected fields): " + json.dumps({
                "input_errors": verification.get("input_errors", []),
                "answers": verification.get("independent_source_answers", []),
                "obligations": verification.get("obligations", []),
            })
        schema = object_schema({field: CORE["properties"][field] for field in requested})
        parsed, response, elapsed, _ = self._generate_section(
            instruction, caption, 512, schema=schema
        )
        return parsed, response, elapsed

    @record_generation("Mistral-7B")
    def _generate_section(self, instruction, caption, max_new_tokens, schema=None):
        if schema:
            instruction += "\nReturn JSON only, using these exact keys and types instead of headings: " + json.dumps(schema)
        prompt = self._chat_prompt(instruction + "\nCaption:\n" + caption)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_tokens = int(inputs["input_ids"].shape[1])
        if input_tokens > 3072:
            raise ValueError("Agent 2 context budget exceeded; caption not truncated")
        inputs = inputs.to(self.model.device)
        started = time.perf_counter()
        constrained = {"prefix_allowed_tokens_fn": prefix_constraint(self.tokenizer, schema)} if schema else {}
        if schema:
            from transformers import StoppingCriteriaList
            from engine.output_contracts import complete_json_stopper
            constrained["stopping_criteria"] = StoppingCriteriaList([
                complete_json_stopper(self.tokenizer, input_tokens, schema)])
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, **constrained, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        generated = output[:, input_tokens:]
        count = int(generated.shape[-1])
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        try:
            parsed = json.loads(response) if schema else parse_claim_response(response)
        except ValueError:
            parsed = {}
        schema_valid = validate_shape(parsed, schema) if schema else None
        if schema and not schema_valid:
            parsed = {}
        return parsed, response, elapsed, {
            "input_tokens": input_tokens, "generated_tokens": count,
            "max_new_tokens": max_new_tokens, "hit_token_limit": count >= max_new_tokens,
            "elapsed_seconds": round(elapsed, 4),
            "decoder": DECODER_ID if schema else "greedy_unconstrained_headings",
            "raw_response": response, "prompt": prompt, "schema": schema,
            "schema_valid": schema_valid,
        }

    @staticmethod
    def _claim_frame_quality(output):
        contract = output.get("claim_contract", {}) or {}
        relation_family = normalize_relation_family(
            output.get("relation_family", ""),
            output.get("caption_proposition", ""),
            output.get("expected_visual_state", ""),
            output.get("opposite_visual_state", ""),
        )
        return (
            int(contract.get("safe_for_automatic_directional_reasoning", False)),
            int(contract.get("safe_for_directional_reasoning", False)),
            int(contract.get("proposition_preserved", False)),
            int(contract.get("entity_frame_preserved", False)),
            int(contract.get("relation_pair_complete", False)),
            int(bool(relation_family and relation_family != "other")),
            -len(contract.get("warnings", []) or []),
        )

    @staticmethod
    def _to_spec_schema(parsed: dict, raw_response: str) -> dict:

        figurative_type_raw = parsed.get("figurative_type", "")
        background_knowledge = parsed.get("background_knowledge", "")
        caption_proposition = parsed.get("caption_proposition", "")
        explicit_claims = parsed.get("explicit_claims", []) or []
        if caption_proposition and not explicit_claims:
            explicit_claims = [caption_proposition]

        figurative_type, figurative_type_was_guessed = (
            ClaimExtractionAgent._normalize_figurative_type(
                figurative_type_raw
            )
        )
        relation_family_raw = parsed.get("relation_family", "")
        relation_family = normalize_relation_family(
            relation_family_raw,
            caption_proposition,
            parsed.get("expected_visual_state", ""),
            parsed.get("opposite_visual_state", ""),
        )

        return {
            "surface_meaning": parsed.get("literal_meaning", ""),
            "figurative_type": figurative_type,
            "intended_meaning": parsed.get("underlying_message", ""),
            "linguistic_cue": parsed.get("linguistic_cue", ""),
            "polarity_reversal": parsed.get("polarity_reversal", ""),
            "background_knowledge": background_knowledge or "Not specified",
            "non_literal_expressions": parsed.get("non_literal_expressions", []) or [],
            "caption_proposition": caption_proposition,
            "negation": parsed.get("negation", []),
            "quantities": parsed.get("quantities", []),
            "claim_modifiers": parsed.get("claim_modifiers", []),
            "claim_subject": parsed.get("claim_subject", ""),
            "claim_predicate": parsed.get("claim_predicate", ""),
            "claim_object": parsed.get("claim_object", ""),
            "claim_source": parsed.get("claim_source", ""),
            "claim_target": parsed.get("claim_target", ""),
            "asserted_property": parsed.get("asserted_property", ""),
            "transferred_property": parsed.get("transferred_property", ""),
            "incongruity": parsed.get("incongruity", ""),
            "caption_polarity": ClaimExtractionAgent._normalize_caption_polarity(
                parsed.get("literal_polarity", parsed.get("caption_polarity", "")),
                parsed.get("asserted_property", ""),
                parsed.get("underlying_message", ""),
            ),
            "alternative_interpretation": parsed.get(
                "alternative_interpretation", ""
            ),
            "relation_family": relation_family or relation_family_raw,
            "expected_visual_state": parsed.get("expected_visual_state", ""),
            "opposite_visual_state": parsed.get("opposite_visual_state", ""),
            "reasoning_requirement": parsed.get(
                "reasoning_requirement", ""
            ),
            "structural_reasoning_type": parsed.get(
                "structural_reasoning_type", ""
            ),
            "figurative_mechanism_candidates": parsed.get(
                "figurative_mechanism_candidates", ""
            ),
            "literal_polarity": parsed.get("literal_polarity", ""),
            "intended_polarity": parsed.get("intended_polarity", ""),
            "comparison_direction": parsed.get("comparison_direction", ""),
            "evaluation_target": parsed.get("evaluation_target", ""),
            "time_or_panel_scope": parsed.get("time_or_panel_scope", ""),
            "explicit_claims": explicit_claims,
            "implicit_claims": parsed.get("implicit_claims", []) or [],
            "linguistic_notes": parsed.get("linguistic_notes", []) or [],
            "language_confidence": parsed.get("confidence"),
            "_figurative_type_was_guessed": figurative_type_was_guessed,
            "_figurative_type_source": "primary" if not figurative_type_was_guessed else "unresolved",
            "_relation_family_raw": relation_family_raw,
            "_caption_semantic_audit": parsed.get("_caption_semantic_audit"),
            "claim_graph": parsed.get("claim_graph"),
            "_graph_generation": parsed.get("_graph_generation"),
            "_internal": parsed,
        }

    @staticmethod
    def _preserve_source_caption(output, caption):
        """Keep the source proposition immutable while retaining diagnostics.

        Agent 2 may interpret or decompose a caption, but downstream relation
        reasoning must never operate on a shortened generated paraphrase.
        """
        preserved = dict(output or {})
        generated = " ".join(
            str(preserved.get("caption_proposition") or "").split()
        )
        source = str(caption or "")
        preserved["generated_caption_proposition"] = generated
        preserved["caption_proposition"] = source
        preserved["source_caption_immutable"] = True
        preserved.setdefault("explicit_claims", [])
        if source and not preserved["explicit_claims"]:
            preserved["explicit_claims"] = [source]
        return preserved

    @classmethod
    def _merge_repaired_claim_fields(cls, base, retry_parsed, caption):
        """Keep only retry field groups that improve the source audit.

        Relation states and entity roles have cross-field dependencies, so the
        candidates include both individual fields and coherent groups.  The
        immutable caption is reattached after every candidate and therefore a
        superficially fluent but source-changing repair cannot win.
        """
        best = dict(base or {})
        best_quality = cls._claim_frame_quality(best)
        accepted_fields = []
        available = {
            field: retry_parsed.get(field)
            for field in cls.CLAIM_FRAME_FIELDS
            if retry_parsed.get(field) is not None
            and str(retry_parsed.get(field)).strip()
        }
        groups = [[field] for field in available]
        groups.extend((
            [
                field for field in (
                    "relation_family", "expected_visual_state",
                    "opposite_visual_state", "reasoning_requirement",
                    "background_knowledge",
                ) if field in available
            ],
            [
                field for field in (
                    "claim_subject", "claim_predicate", "claim_object",
                    "claim_source", "claim_target", "asserted_property",
                ) if field in available
            ],
            list(available),
        ))
        for fields in groups:
            if not fields:
                continue
            candidate = dict(best)
            for field in fields:
                candidate[field] = available[field]
            candidate["relation_family"] = normalize_relation_family(
                candidate.get("relation_family", ""),
                candidate.get("caption_proposition", ""),
                candidate.get("expected_visual_state", ""),
                candidate.get("opposite_visual_state", ""),
            ) or candidate.get("relation_family", "")
            candidate = attach_claim_contract(
                cls._preserve_source_caption(candidate, caption), caption
            )
            quality = cls._claim_frame_quality(candidate)
            if quality > best_quality:
                best = candidate
                best_quality = quality
                accepted_fields.extend(fields)
        return best, list(dict.fromkeys(accepted_fields))

    def analyze(self, caption, feedback=None):
        prompt_text = self.DEFAULT_PROMPT

        if feedback:

            prompt_text += f"""

========================================================
FEEDBACK FROM PREVIOUS ROUND
========================================================

Your previous language understanding was considered insufficient.

Feedback:

{feedback}

Reconsider the caption carefully.

Do NOT imagine the image.

Focus especially on:
- figurative meaning
- implied message
- possible metaphor, sarcasm, humor, or a literal caption
- hidden assumptions
- contextual interpretation
- whether another figurative interpretation better explains the caption

Return the SAME output format as before.
"""

        parsed, response, elapsed, core_diagnostics = self._generate_section(
            prompt_text, caption, 512, schema=object_schema({key: value for key, value in CORE["properties"].items()
                if key not in {"expected_visual_state", "opposite_visual_state"}})
        )
        interpretation, interpretation_response, interpretation_elapsed, interpretation_diagnostics = (
            self._generate_section(self.INTERPRETATION_PROMPT, caption, 384, schema=INTERPRETATION)
        )
        # Interpretation cannot overwrite source roles, states or proposition.
        interpretation_fields = {
            "figurative_type", "linguistic_cue", "polarity_reversal",
            "literal_meaning", "underlying_message", "alternative_interpretation",
            "literal_polarity", "intended_polarity", "structural_reasoning_type",
            "background_knowledge", "confidence",
        }
        parsed.update({key: value for key, value in interpretation.items()
                       if key in interpretation_fields})
        elapsed += interpretation_elapsed
        generation_diagnostics = {
            "core": core_diagnostics, "interpretation": interpretation_diagnostics,
            "hit_token_limit": core_diagnostics["hit_token_limit"] or interpretation_diagnostics["hit_token_limit"],
            "generated_tokens": core_diagnostics["generated_tokens"] + interpretation_diagnostics["generated_tokens"],
            "max_new_tokens": 896,
        }

        from engine.claim_semantics import audit_core
        from engine.claim_graph import attach_graph, repair_graph, bind_readings
        elapsed += attach_graph(self, caption, parsed,
                                research_decomposition=getattr(self, "research_decomposition", False))
        parsed["_caption_semantic_audit"] = audit_core(self, caption, parsed)
        parsed["claim_graph"] = bind_readings(parsed["claim_graph"],
            parsed["_caption_semantic_audit"].get("independent_source_answers", []))
        elapsed += parsed["_caption_semantic_audit"]["_generation_seconds"]
        generation_diagnostics["semantic_audit"] = parsed["_caption_semantic_audit"]["_generation_diagnostics"]
        spec_output = attach_reasoning_profile(attach_claim_contract(
            self._preserve_source_caption(
                self._to_spec_schema(parsed, response), caption
            ),
            caption,
        ))
        spec_output["_claim_retry_attempted"] = False
        spec_output["_claim_retry_success"] = False
        spec_output["_claim_retry_seconds"] = 0.0
        if not spec_output["claim_contract"].get(
            "safe_for_directional_reasoning", False
        ):
            spec_output["_claim_retry_attempted"] = True
            primary_quality = self._claim_frame_quality(spec_output)
            try:
                retry_parsed, retry_response, retry_elapsed = (
                    self._retry_structured_claim(
                        caption,
                        spec_output["claim_contract"].get(
                            "invalid_field_groups", []
                        ),
                        defective_fields=(spec_output.get("_caption_semantic_audit") or {}).get("defective_fields"),
                        verification=spec_output.get("_caption_semantic_audit"),
                    )
                )
            except (RuntimeError, ValueError) as error:
                spec_output["_claim_retry_error"] = str(error)
                print(f"[Agent2 WARNING] Claim-frame recovery failed: {error}")
            else:
                elapsed += retry_elapsed
                repaired_fields = [key for key in retry_parsed if key in CORE["properties"]]
                repaired = dict(spec_output)
                repaired.update({key: retry_parsed[key] for key in repaired_fields})
                elapsed += repair_graph(self, caption, repaired,
                                        verification=spec_output.get("_caption_semantic_audit"))
                repaired["_caption_semantic_audit"] = audit_core(self, caption, repaired)
                recovery_audit = dict(repaired["_caption_semantic_audit"])
                elapsed += repaired["_caption_semantic_audit"]["_generation_seconds"]
                repaired = attach_claim_contract(self._preserve_source_caption(repaired, caption), caption)
                if self._claim_frame_quality(repaired) > primary_quality:
                    spec_output = attach_reasoning_profile(repaired)
                    final_valid = bool(
                        repaired["claim_contract"].get("fully_valid", False)
                    )
                    spec_output["_claim_retry_success"] = final_valid
                    spec_output["_claim_retry_improved"] = True
                    spec_output["_claim_retry_repaired_fields"] = repaired_fields
                    spec_output["_claim_retry_directionally_safe"] = bool(
                        repaired["claim_contract"].get(
                            "safe_for_directional_reasoning", False
                        )
                    )
                    print("[Agent2] Accepted improved structured-claim recovery.")
                else:
                    print("[Agent2] Rejected non-improving structured-claim recovery.")
                spec_output["_claim_retry_attempted"] = True
                spec_output["_claim_retry_seconds"] = round(retry_elapsed, 4)
                spec_output["_raw_claim_retry_response"] = retry_response
                spec_output["_claim_retry_semantic_audit"] = recovery_audit
                spec_output["_claim_retry_graph"] = repaired.get("claim_graph")
                spec_output["_graph_repair_history"] = repaired.get("_graph_repair_history", [])
                spec_output["_claim_retry_proposed_fields"] = {key: retry_parsed[key] for key in repaired_fields}

        spec_output["_generation_seconds"] = round(elapsed, 4)
        spec_output["_generation_diagnostics"] = generation_diagnostics
        spec_output["_raw_interpretation_response"] = interpretation_response
        spec_output["_claim_extraction_version"] = "source_graph_bound_readings_v4"

        spec_output["_figurative_type_retry_attempted"] = False
        spec_output["_figurative_type_retry_failed"] = False
        if spec_output["_figurative_type_was_guessed"]:
            spec_output["_figurative_type_retry_attempted"] = True
            try:
                resolved_type, resolution_confidence, resolution_scores = (
                    self._score_figurative_type(caption)
                )
            except (RuntimeError, ValueError) as error:
                spec_output["_figurative_type_retry_failed"] = True
                spec_output["_figurative_type_resolution_error"] = str(error)
                print(f"[Agent2 WARNING] Figurative-type recovery failed: {error}")
            else:
                spec_output.update({
                    "figurative_type": resolved_type,
                    "_figurative_type_was_guessed": False,
                    "_figurative_type_source": "scored_recovery",
                    "_figurative_type_retry_failed": False,
                    "_figurative_type_resolution_confidence": resolution_confidence,
                    "_figurative_type_resolution_scores": resolution_scores,
                })
                spec_output = attach_reasoning_profile(spec_output)
                print(
                    "[Agent2] Figurative-type recovery scores: "
                    + ", ".join(
                        f"{label}={score:.3f}"
                        for label, score in resolution_scores.items()
                    )
                )

        print("\n================ SPEC-COMPLIANT OUTPUT =====================\n")
        print({k: v for k, v in spec_output.items() if k != "_internal"})
        print("\n====================================================\n")

        return spec_output

    @staticmethod
    def _audit_field(critique_prompt, name):
        match = re.search(
            rf"(?im)^\s*{re.escape(name)}\s*:\s*(.+?)\s*$",
            str(critique_prompt or ""),
        )
        return " ".join(match.group(1).split()) if match else ""


    def critique(self, caption, critique_prompt, _format_retry=False):
        from engine.claim_witness import audit_claim_witness
        return audit_claim_witness(self, caption, critique_prompt)
