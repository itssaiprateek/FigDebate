import re
import time
try:
    import torch
except ImportError:
    torch = None
from utils.parser import parse_visual_response


class VisualGroundingAgent:
    """
    Agent 1:
    Performs visual grounding using a preloaded LLaVA model.
    Extracts detailed visual evidence only.

    Returns the FIXED schema Shrihan's orchestrator expects:
        visual_description (str), objects (list[str]),
        scene_type (str), symbolic_tone (str)
    Everything else the internal parser produces is kept under
    "_internal" for debugging / feedback-loop use, but is NOT part
    of the contract with the orchestrator.
    """

    DEFAULT_PROMPT = """
Inspect only the image. You are not given a caption. Report visible facts and
written text, never an entailment label or hidden interpretation.

Return exactly these seven headings. Keep each value concise.

Literal Scene: one factual sentence describing the complete image
Objects: important visible people and objects
Visible Text: exact printed words in quotation marks, with their region or
attached object; write None only when no words are readable
Visual Facts: directly observed actions, appearances, and states
Visual Relations: actual spatial or object-to-text bindings; write a complete
observation such as "left bottle has text 'X'", never a template such as
"Left/Right, Top/Bottom"
Scene Type: two to five words
Confidence: one decimal from 0 to 1

For memes, charts, and comparisons, inspect every region. Preserve which exact
text belongs to each person, object, panel, or side. Mark uncertain OCR with
"uncertain" rather than inventing letters.
"""

    FACT_RECOVERY_PROMPT = """
Inspect the complete image again. Return exactly these five lines:
Literal Scene: one factual sentence
Objects: visible people and important objects
Visual Facts: directly observed actions, appearances, and states
Visual Relations: complete spatial or object-to-object observations, or None
Scene Type: two to five words
Do not interpret symbolism and do not discuss a caption.
"""

    OCR_RECOVERY_PROMPT = """
Inspect all readable writing in the complete image, including labels attached
to people or objects and text in separate panels. Return exactly:
Visible Text:
- [region or attached object] "exact printed words"
Text Bindings:
- [region or object] "exact phrase" labels or describes that visible item
Write None under both headings only when no characters are readable. Do not
describe clothing, people, or objects as text.
"""

    def __init__(self, llava_model):
        if torch is None:
            raise RuntimeError(
                "Agent 1 requires PyTorch. Run check_environment.py."
            )

        self.processor = llava_model.processor
        self.model = llava_model.model

        print("[Agent1] Ready.")

    def _move_inputs(self, inputs):
        """Move token IDs and image tensors without casting integer IDs."""
        return {
            key: (
                value.to(self.model.device, dtype=torch.float16)
                if torch.is_floating_point(value)
                else value.to(self.model.device)
            )
            for key, value in inputs.items()
        }

    @staticmethod
    def _is_observed_relation(value):
        text = " ".join(str(value or "").split()).strip()
        normalized = re.sub(r"[^a-z0-9 ]", " ", text.lower())
        normalized = " ".join(normalized.split())
        if not normalized or normalized in {
            "left right",
            "top bottom",
            "left right top bottom",
            "tilted rotated",
            "trend arrow contrast",
            "object to object",
        }:
            return False
        if re.fullmatch(
            r"(?:left|right|top|bottom|tilted|rotated|trend|arrow|contrast|"
            r"object|to|and|or|white|red|first|second)(?:\s+(?:left|right|top|"
            r"bottom|tilted|rotated|trend|arrow|contrast|object|to|and|or|"
            r"white|red|first|second))*",
            normalized,
        ):
            return False
        return bool(
            re.search(
                r"\b(is|are|has|have|contains?|shows?|reads?|says|labels?|"
                r"points?|pointing|faces?|stands?|sits?|lies?|located|attached|above|"
                r"below|beside|between|next|tilted|rotated|rises?|falls?)\b",
                normalized,
            )
        )

    @staticmethod
    def _format_heading_count(raw_response):
        return len(re.findall(
            r"^(?:Literal Scene|Objects|Visible Text|Visual Facts|"
            r"Visual Relations|Scene Type|Confidence)\s*:",
            str(raw_response or ""),
            flags=re.IGNORECASE | re.MULTILINE,
        ))

    @staticmethod
    def _merge_parsed(primary, *retries):
        output = dict(primary or {})
        list_fields = (
            "people", "objects", "actions", "visible_text",
            "symbolic_elements", "possible_visual_metaphors",
            "visual_facts", "visual_relations", "uncertain_observations",
        )
        for key in list_fields:
            combined = []
            seen = set()
            for source in (primary, *retries):
                for item in (source or {}).get(key, []) or []:
                    normalized = " ".join(str(item).lower().split())
                    if normalized and normalized not in seen:
                        combined.append(item)
                        seen.add(normalized)
            output[key] = combined
        for key in ("literal_scene", "environment", "scene_type"):
            if not output.get(key):
                for source in retries:
                    if (source or {}).get(key):
                        output[key] = source[key]
                        break
        if output.get("confidence") is None:
            for source in retries:
                if (source or {}).get("confidence") is not None:
                    output["confidence"] = source["confidence"]
                    break
        return output

    def _generate_response(self, image, prompt_text, max_new_tokens):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = self._move_inputs(inputs)
        started = time.time()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                no_repeat_ngram_size=8,
                use_cache=True,
            )
        generated_tokens = output[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]
        return response, time.time() - started

    @staticmethod
    def _to_spec_schema(parsed: dict, raw_response: str) -> dict:

        objects = parsed.get("objects", []) or []

        visible_text = []
        reclassified_facts = []
        reclassified_relations = []
        object_names = {
            re.sub(r"[^a-z0-9 ]", " ", str(item).lower()).strip()
            for item in objects
        }
        direction_pattern = re.compile(
            r"\b(arrow|upward|downward|left|right|above|below|tilted|rotated|trend)\b",
            flags=re.IGNORECASE,
        )
        observation_pattern = re.compile(
            r"\b(is|are|wearing|holding|sitting|standing|pointing|appears|shows|displayed)\b",
            flags=re.IGNORECASE,
        )
        for item in parsed.get("visible_text", []) or []:
            normalized = re.sub(r"[^a-z0-9 ]", " ", str(item).lower()).strip()
            has_explicit_text_signal = bool(
                re.search(r"[\"']|\d|[$#@]", str(item))
            )
            is_observation = (
                not has_explicit_text_signal
                and (
                    normalized in object_names
                    or observation_pattern.search(str(item))
                    or direction_pattern.search(str(item))
                )
            )
            if not is_observation:
                visible_text.append(item)
            elif direction_pattern.search(str(item)):
                reclassified_relations.append(item)
            else:
                reclassified_facts.append(item)

        visual_relations = [
            item
            for item in (
                (parsed.get("visual_relations", []) or [])
                + reclassified_relations
            )
            if VisualGroundingAgent._is_observed_relation(item)
            and not (
                re.search(
                    r"^no .+ (detected|provided|found)\.?$",
                    str(item).strip(),
                    flags=re.IGNORECASE,
                )
                or re.search(
                    r"^(left/right|top/bottom|tilted/rotated|trend|arrow|contrast|"
                    r"object-to-object)\s*:\s*(yes|no|none|\d+)\.?$",
                    str(item).strip(),
                    flags=re.IGNORECASE,
                )
                or str(item).strip().endswith("/")
            )
        ]
        visual_facts = (
            (parsed.get("visual_facts", []) or []) + reclassified_facts
        )

        def unique(items):
            result = []
            seen = set()
            for item in items:
                key = str(item).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    result.append(item)
            return result

        visible_text = unique(visible_text)
        visual_relations = unique(visual_relations)
        visual_facts = unique(visual_facts)

        symbolic = parsed.get("symbolic_elements", []) or []
        metaphors = parsed.get("possible_visual_metaphors", []) or []

        symbolic_tone_parts = [
            s for s in symbolic
            if s and s.strip().lower() not in ("none", "")
        ]
        symbolic_tone = "; ".join(symbolic_tone_parts) if symbolic_tone_parts else "None"

        scene_type = parsed.get("scene_type", "").strip() or "Unspecified"

        description_parts = [
            parsed.get("literal_scene", ""),
            *(parsed.get("people", []) or []),
            *(parsed.get("actions", []) or []),
            parsed.get("environment", ""),
            *visible_text,
            *visual_relations,
            *visual_facts,
        ]
        visual_description = " ".join(p for p in description_parts if p).strip()

        heading_count = VisualGroundingAgent._format_heading_count(raw_response)
        format_valid = bool(heading_count >= 4)
        factual_grounding_present = bool(
            parsed.get("literal_scene")
            and (objects or visual_facts or visual_relations)
        )
        relation_binding_present = any(
            re.search(
                r"\b(left|right|top|bottom|first|second|above|below|attached|"
                r"labels?|reads?|says)\b",
                str(item),
                flags=re.IGNORECASE,
            )
            for item in visual_relations
        )
        schema_issues = []
        if not format_valid:
            schema_issues.append("structured_headings_incomplete")
        if not factual_grounding_present:
            schema_issues.append("factual_grounding_missing")
        raw_placeholder_relation = bool(re.search(
            r"^Visual Relations\s*:\s*(?:Left\s*/\s*Right|"
            r"Top\s*/\s*Bottom|Left/Right,\s*Top/Bottom)\s*$",
            str(raw_response or ""),
            flags=re.IGNORECASE | re.MULTILINE,
        ))
        if (
            (parsed.get("visual_relations") and not visual_relations)
            or raw_placeholder_relation
        ):
            schema_issues.append("visual_relations_are_placeholders")

        return {
            "visual_description": visual_description,
            "objects": objects,
            "scene_type": scene_type,
            "symbolic_tone": symbolic_tone,
            "visual_facts": visual_facts,
            "visual_relations": visual_relations,
            "visible_text": visible_text,
            "visible_text_count": len(visible_text),
            "visual_fact_count": len(visual_facts),
            "visual_relation_count": len(visual_relations),
            "visual_hypotheses": [
                item for item in metaphors
                if item and item.strip().lower() not in ("none", "unclear", "")
            ],
            "possible_visual_metaphors": metaphors,
            "uncertain_observations": parsed.get("uncertain_observations", []) or [],
            "visual_confidence": parsed.get("confidence"),
            "schema_complete": bool(format_valid and factual_grounding_present),
            "schema_format_valid": format_valid,
            "factual_grounding_present": factual_grounding_present,
            "ocr_usable": bool(visible_text),
            "relation_binding_present": relation_binding_present,
            "schema_heading_count": heading_count,
            "schema_issues": schema_issues,
            # kept for debugging / feedback_loop.py failure classification,
            # NOT part of the orchestrator contract
            "_internal": parsed,
        }

    def analyze(self, image, feedback=None):

        prompt_text = self.DEFAULT_PROMPT

        if feedback:

            prompt_text += f"""

    ========================================================
    FEEDBACK FROM PREVIOUS ROUND
    ========================================================

    Your previous visual grounding was considered insufficient.

    Feedback:

    {feedback}

    Carefully inspect the image again.

    Do NOT invent objects.

    Focus especially on:
    - objects that may have been overlooked
    - relationships between objects
    - text inside the image
    - charts, arrows, icons, or signs
    - uncertainty where appropriate

Do not invent new objects simply to satisfy the feedback.
Do not turn a possible symbolic interpretation into a visual fact.

    Return the SAME output format as before.

    """

        def generate(current_prompt, max_new_tokens):
            return self._generate_response(
                image, current_prompt, max_new_tokens
            )

        print("\n========== Agent 1 ==========")

        if torch.cuda.is_available():
            print(
                f"GPU Memory Before : {torch.cuda.memory_allocated()/1024**3:.2f} GB"
            )

        print("Generating visual evidence...")
        response, elapsed = generate(prompt_text, 220)

        print(f"Generation Time : {elapsed:.2f} sec")

        if torch.cuda.is_available():
            print(
                f"GPU Memory After  : {torch.cuda.memory_allocated()/1024**3:.2f} GB"
            )

        print("\n================ RAW LLaVA OUTPUT ================\n")
        print(response)
        print("\n==================================================\n")

        parsed = parse_visual_response(response)
        spec_output = self._to_spec_schema(parsed, response)
        retry_attempted = False
        retry_success = False
        recovery_responses = []
        recovery_parsed = []
        likely_text_surface = bool(re.search(
            r"\b(sign|poster|caption|label|screen|chart|meme|advertisement|"
            r"packaging|bottle|board|banner)\b",
            " ".join([
                parsed.get("literal_scene", ""),
                parsed.get("scene_type", ""),
                " ".join(parsed.get("objects", []) or []),
            ]),
            flags=re.IGNORECASE,
        ))
        needs_fact_recovery = not spec_output["schema_complete"]
        needs_ocr_recovery = bool(
            needs_fact_recovery
            or (likely_text_surface and not spec_output["ocr_usable"])
        )

        if needs_fact_recovery:
            retry_attempted = True
            fact_response, fact_elapsed = generate(
                self.FACT_RECOVERY_PROMPT, 150
            )
            elapsed += fact_elapsed
            recovery_responses.append(("FACT RECOVERY", fact_response))
            recovery_parsed.append(parse_visual_response(fact_response))

        if needs_ocr_recovery:
            retry_attempted = True
            ocr_response, ocr_elapsed = generate(
                self.OCR_RECOVERY_PROMPT, 150
            )
            elapsed += ocr_elapsed
            recovery_responses.append(("OCR RECOVERY", ocr_response))
            recovery_parsed.append(parse_visual_response(ocr_response))

        if recovery_responses:
            print("\n================ RAW LLaVA RECOVERY ================\n")
            for name, recovery_response in recovery_responses:
                print(f"{name}:\n{recovery_response}\n")
            print("====================================================\n")
            parsed = self._merge_parsed(parsed, *recovery_parsed)
            parsed["raw_output"] = "\n\n".join([
                f"PRIMARY:\n{response}",
                *(
                    f"{name}:\n{recovery_response}"
                    for name, recovery_response in recovery_responses
                ),
            ])
            spec_output = self._to_spec_schema(parsed, parsed["raw_output"])
            retry_success = bool(
                spec_output["schema_complete"]
                and (not needs_ocr_recovery or spec_output["ocr_usable"])
            )

        spec_output["_schema_retry_attempted"] = retry_attempted
        spec_output["_schema_retry_success"] = retry_success
        spec_output["_generation_seconds"] = round(elapsed, 4)

        print("\n================ SPEC-COMPLIANT OUTPUT ===================\n")
        print({k: v for k, v in spec_output.items() if k != "_internal"})
        print("\n==================================================\n")

        return spec_output

    def recover_for_claim(
        self,
        image,
        visual_output,
        claim_relation,
        recovery_reason="targeted_grounding_recovery",
    ):
        """Acquire a fresh observation record for a disputed claim."""
        claim_relation = claim_relation or {}
        prompt = f"""
Reinspect the image as a factual evidence collector. The immutable claim frame
below is supplied only to identify relevant entities and regions. Do not choose
ENTAILS or CONTRADICTS and do not invent an expected object.

Claim subject: {claim_relation.get('subject', 'unavailable')}
Claim property: {claim_relation.get('asserted_property', 'unavailable')}
Expected state: {claim_relation.get('expected_visual_state', 'unavailable')}
Opposite state: {claim_relation.get('opposite_visual_state', 'unavailable')}
Expected cues: {', '.join(claim_relation.get('expected_visual_cues', []) or []) or 'unavailable'}
Opposite cues: {', '.join(claim_relation.get('opposite_visual_cues', []) or []) or 'unavailable'}

Return exactly these headings:
Literal Scene: one complete factual sentence
Objects: relevant visible people and objects
Visible Text: exact quoted text with its region or attached object, or None
Visual Facts: directly observed states and actions
Visual Relations: complete entity-to-state, object-to-text, panel, or spatial bindings
Symbolic Elements: recognizable visible symbols and a tentative conventional association, or None
Possible Visual Metaphors: visible transformation or juxtaposition only, or None
Scene Type: two to five words
Confidence: one decimal from 0 to 1

For every text label, state what visible person, object, or region it labels.
For every symbol, state what visible entity it is attached to. Missing evidence
must be reported as uncertain, not converted into an opposite observation.
"""
        response, elapsed = self._generate_response(image, prompt, 260)
        parsed = parse_visual_response(response)
        recovery_responses = [("TARGETED RECOVERY", response)]
        recovery_parsed = [parsed]
        targeted = self._to_spec_schema(parsed, response)

        needs_ocr = bool(
            not targeted.get("ocr_usable")
            and (
                recovery_reason in {
                    "insufficient_visual_evidence",
                    "unresolved_text_layout_binding",
                    "unresolved_text_relation_semantics",
                }
                or re.search(
                    r"\b(sign|poster|label|screen|chart|meme|bottle|board)\b",
                    targeted.get("visual_description", ""),
                    flags=re.IGNORECASE,
                )
            )
        )
        if needs_ocr:
            ocr_response, ocr_elapsed = self._generate_response(
                image, self.OCR_RECOVERY_PROMPT, 160
            )
            elapsed += ocr_elapsed
            recovery_responses.append(("TARGETED OCR", ocr_response))
            recovery_parsed.append(parse_visual_response(ocr_response))

        original_parsed = (visual_output or {}).get("_internal", {}) or {}
        combined = self._merge_parsed(original_parsed, *recovery_parsed)
        combined_raw = "\n\n".join(
            f"{name}:\n{text}" for name, text in recovery_responses
        )
        combined["raw_output"] = combined_raw
        recovered = self._to_spec_schema(combined, combined_raw)
        recovered["_targeted_recovery_attempted"] = True
        recovered["_targeted_recovery_success"] = bool(
            recovered.get("schema_complete")
            and recovered.get("factual_grounding_present")
        )
        recovered["_targeted_recovery_reason"] = recovery_reason
        recovered["_targeted_recovery_seconds"] = round(elapsed, 4)
        recovered["_generation_seconds"] = round(
            float((visual_output or {}).get("_generation_seconds", 0.0))
            + elapsed,
            4,
        )
        return recovered

    @staticmethod
    def _comparison_crops(image):
        """Separate side labels from bottom outcomes in comparison graphics."""
        if not hasattr(image, "size") or not hasattr(image, "crop"):
            return []
        width, height = image.size
        if width <= 1 or height <= 1:
            return []
        x_midpoint = width // 2
        x_overlap = max(1, int(width * 0.08))
        object_bottom = max(1, int(height * 0.43))
        outcome_top = max(0, int(height * 0.65))
        return [
            (
                "left object label",
                image.crop((0, 0, min(width, x_midpoint + x_overlap), object_bottom)),
            ),
            (
                "left bottom outcome",
                image.crop((0, outcome_top, min(width, x_midpoint + x_overlap), height)),
            ),
            (
                "right object label",
                image.crop((max(0, x_midpoint - x_overlap), 0, width, object_bottom)),
            ),
            (
                "right bottom outcome",
                image.crop((max(0, x_midpoint - x_overlap), outcome_top, width, height)),
            ),
        ]

    @staticmethod
    def _usable_region_text(text):
        normalized = " ".join(str(text or "").split()).strip(" .:-\"'").lower()
        if not normalized:
            return False
        placeholders = {
            "text", "label", "object", "object label", "outcome", "unknown",
            "unreadable", "unclear", "none", "n/a", "not visible",
        }
        return normalized not in placeholders and len(normalized) >= 3

    @staticmethod
    def _clean_object_label(text):
        """Remove a predicate accidentally clipped into a category heading."""
        cleaned = " ".join(str(text).split()).strip(" ,.:;")
        lowered = cleaned.lower()
        predicate_markers = (
            " and you're",
            " and you are",
            " and they",
            " and it",
            " and we",
            " but you",
            " while you",
        )
        positions = [
            lowered.find(marker)
            for marker in predicate_markers
            if lowered.find(marker) >= 0
        ]
        if positions:
            cleaned = cleaned[: min(positions)].rstrip(" ,.:;")
        lowered = cleaned.lower().rstrip(".,:;")
        for suffix in (" and", " but", " while"):
            if lowered.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].rstrip(" ,.:;")
                break
        return cleaned

    def _read_region_text(self, image, region_name):
        if "object label" in region_name:
            instruction = (
                "Transcribe only the shortest visible category or identity "
                "heading for the compared object. Exclude sentences about "
                "what it is doing, what someone is trying to do, and any "
                "duration or outcome text."
            )
        else:
            instruction = (
                "Transcribe only the large bottom outcome or result text. "
                "Exclude object descriptions and action or intention text."
            )
        prompt_text = f"""
Inspect only this {region_name} crop of a comparison image. {instruction}
Copy only visibly printed words. Do not infer hidden words. Return exactly one
line:
Text: exact text, or None
"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = self._move_inputs(inputs)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                repetition_penalty=1.10,
                no_repeat_ngram_size=8,
                use_cache=True,
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()
        match = re.search(r"text\s*:\s*(.+)", response, flags=re.IGNORECASE | re.DOTALL)
        text = match.group(1).strip() if match else response
        if not text or text.lower().rstrip(".") == "none":
            return None
        text = " ".join(text.split())[:300]
        if "object label" in region_name:
            text = self._clean_object_label(text)
        return text

    @staticmethod
    def _text_relation_crops(image):
        """Return two overlapping regions for non-tabular text memes."""
        if not hasattr(image, "size") or not hasattr(image, "crop"):
            return []
        width, height = image.size
        if width <= 1 or height <= 1:
            return []
        overlap_x = max(1, int(width * 0.10))
        overlap_y = max(1, int(height * 0.10))
        if width >= height:
            midpoint = width // 2
            return [
                ("left region", image.crop((0, 0, min(width, midpoint + overlap_x), height))),
                ("right region", image.crop((max(0, midpoint - overlap_x), 0, width, height))),
            ]
        midpoint = height // 2
        return [
            ("top region", image.crop((0, 0, width, min(height, midpoint + overlap_y)))),
            ("bottom region", image.crop((0, max(0, midpoint - overlap_y), width, height))),
        ]

    def _read_text_relation_crop(self, image, region_name):
        response, elapsed = self._generate_response(
            image,
            f"""
Inspect only the {region_name}. Transcribe every readable printed phrase and
state which visible person, object, sign, or area it labels. Do not infer
missing words and do not interpret the caption. Return exactly one line:
Text Binding: exact visible phrase -> attached visible entity, or None
""",
            90,
        )
        match = re.search(
            r"text binding\s*:\s*(.+)", response, flags=re.IGNORECASE
        )
        text = " ".join((match.group(1) if match else response).split())[:350]
        if not self._usable_region_text(text):
            return None, elapsed
        return f"{region_name}: {text}", elapsed

    @staticmethod
    def _critique_review_method(critique_prompt):
        if "FIGURATIVE_SYMBOL_REINSPECTION" in critique_prompt:
            return "symbolic_visual_reinspection"
        if "UNRESOLVED_TEXT_RELATION_SEMANTICS" in critique_prompt:
            return "text_relation_visual_reinspection"
        if "UNCORROBORATED_STRUCTURED_RELATION" in critique_prompt:
            return "structured_relation_visual_reinspection"
        return "targeted_visual_reinspection"

    @staticmethod
    def _parse_critique_response(response, review_method):
        headings = (
            "Recommendation", "Observed Entity", "Observed State",
            "Image Region", "Claim Relation", "Reason",
        )

        def field(name):
            following = "|".join(re.escape(item) for item in headings)
            match = re.search(
                rf"(?ims)^\s*{re.escape(name)}\s*:\s*(.*?)"
                rf"(?=^\s*(?:{following})\s*:|\Z)",
                str(response or ""),
            )
            return " ".join(match.group(1).split()).strip() if match else ""

        recommendation_text = field("Recommendation")
        recommendation_match = re.search(
            r"\b(ENTAILS|CONTRADICTS|ABSTAIN)\b",
            recommendation_text,
            flags=re.IGNORECASE,
        )
        recommendation = (
            recommendation_match.group(1).upper()
            if recommendation_match else "ABSTAIN"
        )
        relation_text = field("Claim Relation")
        relation_match = re.search(
            r"\b(SUPPORT|CONFLICT|UNRESOLVED)\b",
            relation_text,
            flags=re.IGNORECASE,
        )
        claim_relation = (
            relation_match.group(1).upper()
            if relation_match else "UNRESOLVED"
        )
        observed_entity = field("Observed Entity")
        observed_state = field("Observed State")
        image_region = field("Image Region")
        reason = field("Reason")
        expected_relation = {
            "ENTAILS": "SUPPORT", "CONTRADICTS": "CONFLICT",
            "ABSTAIN": "UNRESOLVED",
        }[recommendation]
        placeholders = {"", "none", "unavailable", "unknown"}
        format_valid = bool(
            recommendation_match
            and relation_match
            and observed_entity
            and observed_state
            and image_region
            and reason
            and claim_relation == expected_relation
        )
        absence_claim = bool(re.search(
            r"\b(no clear (?:indication|relation)|no evidence|not shown|"
            r"cannot be determined|do not have a clear visual relation)\b",
            reason,
            flags=re.IGNORECASE,
        ))
        specific_evidence = bool(
            format_valid
            and recommendation in {"ENTAILS", "CONTRADICTS"}
            and observed_entity.casefold() not in placeholders
            and observed_state.casefold() not in placeholders
            and image_region.casefold() not in placeholders
            and len(reason.split()) >= 5
            and not absence_claim
        )
        return {
            "stance": "UNRESOLVED",
            "recommendation": recommendation,
            "reason": reason or str(response or "").strip(),
            "specific_evidence": specific_evidence,
            "observed_entity": observed_entity,
            "observed_state": observed_state,
            "image_region": image_region,
            "claim_relation": claim_relation,
            "review_method": review_method,
            "_format_valid": format_valid,
            "_raw_response": str(response or "").strip(),
        }

    @staticmethod
    def _critique_quality(parsed):
        return sum(bool(parsed.get(field)) for field in (
            "observed_entity", "observed_state", "image_region", "reason"
        )) + 4 * int(bool(parsed.get("_format_valid"))) + 2 * int(
            bool(parsed.get("specific_evidence"))
        )

    def critique(self, image, critique_prompt):

        if "UNRESOLVED_TEXT_LAYOUT_BINDING" in critique_prompt:
            region_texts = {}
            for region_name, crop in self._comparison_crops(image):
                region_text = self._read_region_text(crop, region_name)
                if self._usable_region_text(region_text):
                    region_texts[region_name] = region_text
            pair_regions = {
                "left object label",
                "left bottom outcome",
                "right object label",
                "right bottom outcome",
            }
            if pair_regions.issubset(region_texts):
                region_pairs = [
                    {
                        "side": "left",
                        "object_text": region_texts["left object label"],
                        "outcome_text": region_texts["left bottom outcome"],
                    },
                    {
                        "side": "right",
                        "object_text": region_texts["right object label"],
                        "outcome_text": region_texts["right bottom outcome"],
                    },
                ]
                reason = (
                    "Region-bound outcome pairs (the bottom text is the displayed "
                    "result for the object on the same side): "
                    f"LEFT bottom outcome = '{region_texts['left bottom outcome']}' "
                    f"for object label '{region_texts['left object label']}'; "
                    f"RIGHT bottom outcome = '{region_texts['right bottom outcome']}' "
                    f"for object label '{region_texts['right object label']}'."
                )
            elif len(region_texts) >= 2:
                region_pairs = []
                reason = "; ".join(
                    f"{name.capitalize()} region reads: {text}"
                    for name, text in region_texts.items()
                )
            else:
                region_pairs = []
                reason = ""
            if reason and region_pairs:
                print("\n========== Agent 1 Region Review ==========\n")
                print(reason)
                print("\n===========================================\n")
                return {
                    "stance": "UNRESOLVED",
                    "recommendation": "ABSTAIN",
                    "reason": reason,
                    "specific_evidence": bool(region_pairs),
                    "observed_entity": "comparison regions",
                    "observed_state": reason,
                    "image_region": "left and right comparison regions",
                    "claim_relation": "UNRESOLVED",
                    "review_method": "region_ocr",
                    "region_pairs": region_pairs,
                    "_format_valid": True,
                }
            if reason:
                critique_prompt += (
                    "\nPartial crop OCR (not yet a verified pair):\n"
                    + reason
                    + "\nUse it only if the full-image reinspection confirms "
                    "which visible entity each phrase labels.\n"
                )

        crop_ocr_candidates = []
        crop_ocr_seconds = 0.0
        if "UNRESOLVED_TEXT_RELATION_SEMANTICS" in critique_prompt:
            for region_name, crop in self._text_relation_crops(image):
                candidate, elapsed = self._read_text_relation_crop(
                    crop, region_name
                )
                crop_ocr_seconds += elapsed
                if candidate:
                    crop_ocr_candidates.append(candidate)
            if crop_ocr_candidates:
                critique_prompt += (
                    "\nUnverified regional OCR candidates:\n- "
                    + "\n- ".join(crop_ocr_candidates)
                    + "\nConfirm each phrase and attachment in the complete image "
                    "before using it as evidence.\n"
                )

        prompt_text = f"""
You are the independent visual reviewer in a multimodal reasoning system.

You are not shown another agent's answer. Evaluate the caption relation using
only the image and the supplied immutable claim frame.

Rules:
- Inspect the image carefully again.
- Do NOT use outside knowledge.
- Do NOT imagine missing objects.
- Use only visible visual evidence.
- Recommend a label only when a specific visible relation supports it.
- Use ABSTAIN when the image is not directionally decisive.
- A printed label or visible symbol may instantiate a caption role without the
  literal object being present, but only when its attachment is visible.
- Do not repeat the supplied analysis or any headings other than the six
  response fields below.

Return exactly these six lines, without brackets or angle brackets:
Recommendation: ENTAILS, CONTRADICTS, or ABSTAIN
Observed Entity: exact visible person or object, or None
Observed State: exact directly observed state or relation, or None
Image Region: location of that evidence, or None
Claim Relation: SUPPORT, CONFLICT, or UNRESOLVED
Reason: one concise explanation linking the observation to the claim state

====================================================

{critique_prompt}
"""
        print("\n========== Agent 1 Critique ==========")

        response, primary_seconds = self._generate_response(
            image, prompt_text, 170
        )

        print("\n================ AGENT 1 CRITIQUE ================\n")
        print(response)
        print("\n=================================================\n")
        review_method = self._critique_review_method(critique_prompt)
        parsed = self._parse_critique_response(response, review_method)
        retry_response = ""
        retry_seconds = 0.0
        retry_used = not parsed["_format_valid"]
        if retry_used:
            retry_prompt = f"""
Inspect the image and repair the visual review. Do not repeat this instruction.
Use the claim frame below only to decide what visible state must be checked.

{critique_prompt}

Return exactly six single-line fields. Every field is mandatory. Image Region
must name a visible location. Recommendation and Claim Relation must agree:
ENTAILS=SUPPORT, CONTRADICTS=CONFLICT, ABSTAIN=UNRESOLVED.
Recommendation: ENTAILS, CONTRADICTS, or ABSTAIN
Observed Entity: exact visible person, object, printed label, or symbol
Observed State: exact visible state, action, attachment, or bound text
Image Region: exact location in the image
Claim Relation: SUPPORT, CONFLICT, or UNRESOLVED
Reason: one sentence connecting the visible observation to expected or opposite state
"""
            retry_response, retry_seconds = self._generate_response(
                image, retry_prompt, 150
            )
            retry_parsed = self._parse_critique_response(
                retry_response, review_method
            )
            if self._critique_quality(retry_parsed) > self._critique_quality(parsed):
                parsed = retry_parsed
        parsed.update({
            "_format_retry_used": retry_used,
            "_format_retry_success": bool(retry_used and parsed["_format_valid"]),
            "_raw_primary_response": response.strip(),
            "_raw_retry_response": retry_response.strip(),
            "_region_ocr_candidates": crop_ocr_candidates,
            "_generation_seconds": round(
                primary_seconds + retry_seconds + crop_ocr_seconds, 4
            ),
        })
        return parsed
    
