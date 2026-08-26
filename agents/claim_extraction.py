import time
import re
try:
    import torch
except ImportError:
    torch = None

from utils.claim_parser import parse_claim_response
from engine.claim_contract import attach_claim_contract
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
You are Agent 2. Analyze only the caption; never imagine an image or verify
whether it is true. The dataset's figurative phenomenon can occur in the image,
the caption, or both. Therefore a caption may be literal. Do not guess an
unseen visual phenomenon.

Before choosing the type, check:
- sarcasm: does the literal wording reverse the likely intended evaluation?
- metaphor: does a concrete domain describe a different abstract situation?
- humor: is the main mechanism absurdity, incongruity, or a punchline?
- literal: does the caption itself contain no clear figurative device?

Return exactly these concise sections:
Figurative Type: sarcasm, metaphor, humor, or literal
Linguistic Cue: short quoted phrase or cue
Polarity Reversal: yes, no, or unclear, followed by a short reason
Literal Meaning: one sentence
Underlying Message: one sentence
Caption Proposition: one declarative sentence preserving the caption's key
entities, numbers, comparison, and polarity
Claim Subject: the entity or group whose state is asserted
Claim Predicate: the main state, action, quality, or relation
Claim Object: the object or target of the predicate, or None
Claim Source: who expresses, causes, or owns the claim, or Same as subject
Claim Target: who or what receives the action or evaluation, or None
Asserted Property: the exact state, relation, or evaluation being claimed
Transferred Property: for metaphor, the property transferred from source to
target; otherwise None
Incongruity: for sarcasm or humor, the exact expectation/reality mismatch;
otherwise None
Caption Polarity: positive, negative, neutral, mixed, or unclear
Alternative Interpretation: one plausible competing reading, or None
Relation Family: trajectory, pace, outcome, sentiment, safety, trust,
association, quantity, or other
Expected Visual State: one short observable state that would support the claim
Opposite Visual State: one short observable state that would conflict with it
Background Knowledge: one short sentence or None
Confidence: one decimal from 0 to 1
"""
    FIGURATIVE_TYPES = ("sarcasm", "metaphor", "humor", "literal")
    CLAIM_FRAME_FIELDS = (
        "caption_proposition", "claim_subject", "claim_predicate",
        "claim_object", "claim_source", "claim_target",
        "asserted_property", "relation_family", "expected_visual_state",
        "opposite_visual_state",
    )

    def __init__(self, mistral_model, tokenizer):
        if torch is None:
            raise RuntimeError(
                "Agent 2 requires PyTorch. Run check_environment.py."
            )

        self.model = mistral_model
        self.tokenizer = tokenizer

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
            for item in (asserted_property, intended_meaning)
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

    def _retry_structured_claim(self, caption):
        prompt = f"""
Repair only the structured claim for this caption. Preserve every explicit
entity, number, negation, comparison, and polarity. Use None when a source,
object, or target is absent. Relation Family must be exactly one of:
trajectory, pace, outcome, sentiment, safety, trust, association, quantity,
or other. Return exactly these headings and no commentary:

Caption Proposition:
Claim Subject:
Claim Predicate:
Claim Object:
Claim Source:
Claim Target:
Asserted Property:
Relation Family:
Expected Visual State:
Opposite Visual State:

Caption: {caption}
"""
        inputs = self.tokenizer(
            self._chat_prompt(prompt),
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        ).to(self.model.device)
        started = time.time()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                repetition_penalty=1.05,
                no_repeat_ngram_size=6,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(
            generated[0], skip_special_tokens=True
        ).strip()
        return parse_claim_response(response), response, time.time() - started

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
            "claim_subject": parsed.get("claim_subject", ""),
            "claim_predicate": parsed.get("claim_predicate", ""),
            "claim_object": parsed.get("claim_object", ""),
            "claim_source": parsed.get("claim_source", ""),
            "claim_target": parsed.get("claim_target", ""),
            "asserted_property": parsed.get("asserted_property", ""),
            "transferred_property": parsed.get("transferred_property", ""),
            "incongruity": parsed.get("incongruity", ""),
            "caption_polarity": ClaimExtractionAgent._normalize_caption_polarity(
                parsed.get("caption_polarity", ""),
                parsed.get("asserted_property", ""),
                parsed.get("underlying_message", ""),
            ),
            "alternative_interpretation": parsed.get(
                "alternative_interpretation", ""
            ),
            "relation_family": relation_family or relation_family_raw,
            "expected_visual_state": parsed.get("expected_visual_state", ""),
            "opposite_visual_state": parsed.get("opposite_visual_state", ""),
            "explicit_claims": explicit_claims,
            "implicit_claims": parsed.get("implicit_claims", []) or [],
            "linguistic_notes": parsed.get("linguistic_notes", []) or [],
            "language_confidence": parsed.get("confidence"),
            "_figurative_type_was_guessed": figurative_type_was_guessed,
            "_figurative_type_source": "primary" if not figurative_type_was_guessed else "unresolved",
            "_relation_family_raw": relation_family_raw,
            "_internal": parsed,
        }

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

        prompt = self._chat_prompt(f"""
{prompt_text}

Caption:
{caption}
""")
        

        

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
        ).to(self.model.device)

        print("\n========== Agent 2 ==========")

        start = time.time()

        with torch.inference_mode():

            output = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                repetition_penalty=1.08,
                no_repeat_ngram_size=8,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        elapsed = time.time() - start

        print(f"Generation Time : {elapsed:.2f} sec")

        generated = output[:, inputs["input_ids"].shape[1]:]

        response = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ).strip()

        print("\n================ RAW MISTRAL OUTPUT ================\n")
        print(response)
        print("\n====================================================\n")

        parsed = parse_claim_response(response)

        spec_output = attach_claim_contract(
            self._to_spec_schema(parsed, response), caption
        )
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
                    self._retry_structured_claim(caption)
                )
            except (RuntimeError, ValueError) as error:
                spec_output["_claim_retry_error"] = str(error)
                print(f"[Agent2 WARNING] Claim-frame recovery failed: {error}")
            else:
                elapsed += retry_elapsed
                repaired = dict(spec_output)
                for field in self.CLAIM_FRAME_FIELDS:
                    value = retry_parsed.get(field)
                    if value is not None and str(value).strip():
                        repaired[field] = value
                repaired["relation_family"] = normalize_relation_family(
                    repaired.get("relation_family", ""),
                    repaired.get("caption_proposition", ""),
                    repaired.get("expected_visual_state", ""),
                    repaired.get("opposite_visual_state", ""),
                ) or repaired.get("relation_family", "")
                repaired = attach_claim_contract(repaired, caption)
                if (
                    repaired["claim_contract"].get(
                        "safe_for_directional_reasoning", False
                    )
                    and self._claim_frame_quality(repaired) > primary_quality
                ):
                    spec_output = repaired
                    spec_output["_claim_retry_success"] = True
                    print("[Agent2] Accepted improved structured-claim recovery.")
                else:
                    print("[Agent2] Rejected non-improving structured-claim recovery.")
                spec_output["_claim_retry_attempted"] = True
                spec_output["_claim_retry_seconds"] = round(retry_elapsed, 4)
                spec_output["_raw_claim_retry_response"] = retry_response

        spec_output["_generation_seconds"] = round(elapsed, 4)

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

    @staticmethod
    def _unusable_requirement(value):
        normalized = " ".join(str(value or "").casefold().split()).strip(" .")
        return (
            normalized in {"", "none", "n/a", "unavailable", "unknown"}
            or normalized.startswith("none (")
        )

    @classmethod
    def _ground_visual_requirements(
        cls, support_requirement, conflict_requirement,
        figurative_mechanism, critique_prompt,
    ):
        expected_state = cls._audit_field(
            critique_prompt, "Expected visual state"
        )
        opposite_state = cls._audit_field(
            critique_prompt, "Opposite visual state"
        )
        claim_subject = cls._audit_field(
            critique_prompt, "Claim subject"
        ) or "the claim subject"
        asserted_property = cls._audit_field(
            critique_prompt, "Asserted property"
        )
        intended_meaning = cls._audit_field(
            critique_prompt, "Intended meaning"
        )
        if cls._unusable_requirement(support_requirement):
            support_requirement = (
                expected_state
                if not cls._unusable_requirement(expected_state)
                else intended_meaning
            )
        if cls._unusable_requirement(conflict_requirement):
            conflict_requirement = (
                opposite_state
                if not cls._unusable_requirement(opposite_state)
                else f"a visible state opposite to {intended_meaning or 'the caption meaning'}"
            )
        support_requirement = str(support_requirement).strip().rstrip(".")
        conflict_requirement = str(conflict_requirement).strip().rstrip(".")
        if "metaphor" in str(figurative_mechanism).casefold():
            property_text = (
                asserted_property or intended_meaning or "the asserted property"
            )
            support_requirement = (
                f"{support_requirement}. A visible symbol attached to "
                f"{claim_subject} supports the claim only when its condition "
                f"or conventional association expresses {property_text}."
            )
            conflict_requirement = (
                f"{conflict_requirement}. A visible symbol attached to "
                f"{claim_subject} conflicts when its condition or conventional "
                f"association expresses the opposite of {property_text}."
            )
        else:
            support_requirement = (
                f"{support_requirement}. An explicitly bound label, analogy, "
                "or symbol may instantiate the same semantic roles."
            )
            conflict_requirement = (
                f"{conflict_requirement}. An explicitly bound label, analogy, "
                "or symbol may instantiate the opposite role relation."
            )
        return support_requirement.strip(), conflict_requirement.strip()

    @classmethod
    def _validate_visual_requirements(
        cls, support_requirement, conflict_requirement, critique_prompt,
    ):
        """Reject claim requirements that cease to be mutually directional."""
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "in", "is", "it", "of", "on", "or", "the", "to",
            "visible", "visibly", "image", "caption", "claim", "state",
            "condition", "object", "person", "symbol", "label", "role",
        }

        def tokens(value):
            result = set()
            for token in re.findall(r"[a-z0-9]+", str(value or "").casefold()):
                if token in stopwords or len(token) < 3:
                    continue
                for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
                    if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                        token = token[:-len(suffix)]
                        break
                result.add(token)
            return result

        support_tokens = tokens(support_requirement)
        conflict_tokens = tokens(conflict_requirement)
        expected_tokens = tokens(
            cls._audit_field(critique_prompt, "Expected visual state")
        )
        opposite_tokens = tokens(
            cls._audit_field(critique_prompt, "Opposite visual state")
        )
        expected_only = expected_tokens - opposite_tokens
        opposite_only = opposite_tokens - expected_tokens
        errors = []
        if " ".join(str(support_requirement).casefold().split()) == " ".join(
            str(conflict_requirement).casefold().split()
        ):
            errors.append("IDENTICAL_REQUIREMENTS")
        if expected_only and not support_tokens.intersection(expected_only):
            errors.append("SUPPORT_DROPPED_EXPECTED_STATE")
        if opposite_only and not conflict_tokens.intersection(opposite_only):
            errors.append("CONFLICT_DROPPED_OPPOSITE_STATE")
        if (
            support_tokens
            and conflict_tokens
            and support_tokens == conflict_tokens
            and "IDENTICAL_REQUIREMENTS" not in errors
        ):
            errors.append("NON_OPPOSING_REQUIREMENTS")
        expected_text = str(
            cls._audit_field(critique_prompt, "Expected visual state") or ""
        ).casefold()
        outcome_terms = {
            "admit", "honest", "truthful", "success", "successful",
            "effective", "effectively", "result", "outcome", "recover",
            "recovery", "fail", "failure", "last", "disappear",
        }
        required_outcomes = tokens(expected_text).intersection(
            {next(iter(tokens(term)), term) for term in outcome_terms}
        )
        if required_outcomes and not support_tokens.intersection(required_outcomes):
            errors.append("CLAIM_OUTCOME_DROPPED")
        return not errors, errors

    def critique(self, caption, critique_prompt):

        prompt = self._chat_prompt(f"""
You are the independent linguistic claim auditor in a multimodal reasoning
system. You are not shown another agent's answer.

Audit the supplied structured analysis using ONLY the original caption.

Rules:

- Analyze ONLY the caption.
- Do NOT imagine the image.
- Do NOT use outside knowledge unless required to interpret the caption.
- ENDORSE only when entities, numbers, negation, target, polarity, and the
  figurative meaning are preserved.
- CHALLENGE when one of those fields changed.
- ABSTAIN when the supplied analysis is too incomplete to audit.
- The image may realize a literal caption through labels, analogy, or a visual
  metaphor. Requirements must therefore describe semantic roles, not demand
  only the literal real-world object.
- For metaphor, support and conflict requirements must name the visible symbol,
  its attachment to the claim subject, and the expected or opposite property.
- Support Requirement and Conflict Requirement must never be None. They must
  be mutually opposing, visually testable conditions.

Return exactly these six lines, without brackets or angle brackets:
Stance: ENDORSE, CHALLENGE, or ABSTAIN
Support Requirement: observable condition that would support the caption meaning
Conflict Requirement: observable condition that would conflict with it
Figurative Mechanism: literal, metaphor, sarcasm, humor, or unresolved, with a short cue
Ambiguity: the main competing caption interpretation, or None
Reason: whether the structured claim preserved the exact caption meaning

====================================================

Caption:

{caption}

====================================================

Structured Caption Analysis:

{critique_prompt}
""")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        ).to(self.model.device)

        print("\n========== Agent 2 Critique ==========")

        with torch.inference_mode():

            output = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[:, inputs["input_ids"].shape[1]:]

        response = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ).strip()

        print("\n================ AGENT 2 CRITIQUE ================\n")
        print(response)
        print("\n=================================================\n")

        # Do not turn an unparseable response into a fabricated challenge.
        stance = "UNRESOLVED"
        match = re.search(
            r"stance\s*:\s*[<\[\(\"']*\s*(endorse|challenge|abstain)\b",
            response,
            flags=re.IGNORECASE,
        )
        if match:
            stance = match.group(1).upper()
        

        reason_match = re.search(
            r"reason\s*:\s*(.+)", response, flags=re.IGNORECASE | re.DOTALL
        )
        reason = reason_match.group(1).strip() if reason_match else ""
        support_match = re.search(
            r"support requirement\s*:\s*(.+)", response, flags=re.IGNORECASE
        )
        conflict_match = re.search(
            r"conflict requirement\s*:\s*(.+)", response, flags=re.IGNORECASE
        )
        mechanism_match = re.search(
            r"figurative mechanism\s*:\s*(.+)", response, flags=re.IGNORECASE
        )
        ambiguity_match = re.search(
            r"ambiguity\s*:\s*(.+)", response, flags=re.IGNORECASE
        )
        support_requirement = (
            support_match.group(1).strip() if support_match else ""
        )
        conflict_requirement = (
            conflict_match.group(1).strip() if conflict_match else ""
        )
        figurative_mechanism = (
            mechanism_match.group(1).strip() if mechanism_match else ""
        )
        ambiguity = ambiguity_match.group(1).strip() if ambiguity_match else ""
        support_requirement, conflict_requirement = (
            self._ground_visual_requirements(
                support_requirement,
                conflict_requirement,
                figurative_mechanism,
                critique_prompt,
            )
        )
        requirements_valid, requirement_errors = (
            self._validate_visual_requirements(
                support_requirement, conflict_requirement, critique_prompt
            )
        )
        format_valid = bool(
            stance != "UNRESOLVED"
            and reason
            and not self._unusable_requirement(support_requirement)
            and not self._unusable_requirement(conflict_requirement)
            and figurative_mechanism
            and ambiguity_match
        )

        return {
            "stance": stance,
            "reason": reason or response.strip(),
            "specific_evidence": len(reason.split()) >= 5,
            "support_requirement": support_requirement,
            "conflict_requirement": conflict_requirement,
            "figurative_mechanism": figurative_mechanism,
            "ambiguity": ambiguity,
            "requirements_source": "caption_audit_with_role_equivalence",
            "requirements_valid": requirements_valid,
            "requirement_errors": requirement_errors,
            "_format_valid": format_valid,
            "_raw_response": response,
        }
