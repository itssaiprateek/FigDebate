from copy import deepcopy
import re
import time
try:
    import torch
except ImportError:
    torch = None
from utils.visual_parser import parse_visual_response
from agents.visual_adapter import (
    AtomicVisualQuestionController,
    VisualAnswer,
    VisualQuestion,
    normalized,
    question_region,
    split_atomic_items,
)


class VisualGroundingAgent:
    """
    Agent 1:
    Performs visual grounding using a preloaded vision model.
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

    def __init__(self, vision_model):
        self.runtime = vision_model
        self.processor = vision_model.processor
        self.model = vision_model.model
        self.backend = getattr(vision_model, "backend", "legacy_llava")
        self.uses_atomic_adapter = bool(
            getattr(vision_model, "supports_atomic_questions", False)
        )
        if torch is None and not self.uses_atomic_adapter:
            raise RuntimeError(
                "The legacy Agent 1 path requires PyTorch. "
                "Run check_environment.py."
            )
        self.question_controller = AtomicVisualQuestionController()
        self._last_generation_diagnostics = {}

        print(f"[Agent1] Ready ({self.backend}).")

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
        if (
            getattr(self, "uses_atomic_adapter", False)
            and hasattr(self, "runtime")
            and hasattr(self.runtime, "generate")
        ):
            response, elapsed = self.runtime.generate(
                image, prompt_text, max_new_tokens=max_new_tokens
            )
            self._last_generation_diagnostics = dict(
                getattr(self.runtime, "_last_generation_diagnostics", {}) or {}
            )
            return response, elapsed

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
        generated_count = int(generated_tokens.shape[1])
        eos_value = getattr(
            getattr(self.model, "generation_config", None), "eos_token_id", None
        )
        eos_ids = set(eos_value if isinstance(eos_value, (list, tuple)) else [eos_value])
        eos_ids.discard(None)
        last_token = (
            int(generated_tokens[0, -1].item()) if generated_count else None
        )
        self._last_generation_diagnostics = {
            "generated_tokens": generated_count,
            "max_new_tokens": int(max_new_tokens),
            "hit_token_limit": bool(generated_count >= int(max_new_tokens)),
            "ended_by_eos": bool(last_token in eos_ids),
        }
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

    def _run_atomic_question(self, image, question):
        valid_question, question_error = self.question_controller.validate_question(
            question.text, question.question_type
        )
        if not valid_question:
            return VisualAnswer(
                question_id=question.question_id,
                question_type=question.question_type,
                question=question.text,
                answer="",
                status="INVALID_QUESTION",
                valid=False,
                error=question_error,
                raw_response="",
                elapsed_seconds=0.0,
                generation_diagnostics={},
            )

        prompt = self.question_controller.build_prompt(
            question.text, question.question_type
        )
        raw, elapsed = self._generate_response(
            image, prompt, question.max_new_tokens
        )
        diagnostics = dict(self._last_generation_diagnostics or {})
        answer, status, valid, error = self.question_controller.validate_answer(
            raw,
            question.text,
            question.question_type,
            diagnostics=diagnostics,
        )
        retry_attempted = not valid
        retry_success = False
        raw_responses = [raw]
        if retry_attempted:
            retry_raw, retry_elapsed = self._generate_response(
                image,
                self.question_controller.retry_prompt(
                    question.text, question.question_type
                ),
                question.max_new_tokens,
            )
            elapsed += retry_elapsed
            retry_diagnostics = dict(self._last_generation_diagnostics or {})
            retry_answer, retry_status, retry_valid, retry_error = (
                self.question_controller.validate_answer(
                    retry_raw,
                    question.text,
                    question.question_type,
                    diagnostics=retry_diagnostics,
                )
            )
            raw_responses.append(retry_raw)
            diagnostics = {
                "primary": diagnostics,
                "retry": retry_diagnostics,
            }
            if retry_valid:
                answer, status, valid, error = (
                    retry_answer, retry_status, retry_valid, retry_error
                )
                retry_success = True

        return VisualAnswer(
            question_id=question.question_id,
            question_type=question.question_type,
            question=question.text,
            answer=answer,
            status=status,
            valid=valid,
            error=error,
            raw_response="\n\nRETRY:\n".join(raw_responses),
            elapsed_seconds=round(elapsed, 4),
            retry_attempted=retry_attempted,
            retry_success=retry_success,
            generation_diagnostics=diagnostics,
        )

    def answer_visual_question(
        self,
        image,
        question,
        question_type=None,
        question_id="agent1_visual_question",
    ):
        """Answer one neutral visual question through a validated public API."""
        question_type = question_type or self.question_controller.infer_question_type(
            question
        )
        max_tokens = {
            "count": 12,
            "yes_no": 18,
            "ocr": 120,
            "relation": 80,
        }.get(question_type, 70)
        answer = self._run_atomic_question(
            image,
            VisualQuestion(
                question_id=question_id,
                question_type=question_type,
                text=" ".join(str(question or "").split()),
                max_new_tokens=max_tokens,
            ),
        )
        return answer.to_dict()

    @staticmethod
    def _atomic_schema(answers, backend="atomic_visual"):
        by_id = {answer.question_id: answer for answer in answers}

        def observed(question_id):
            answer = by_id.get(question_id)
            return answer.answer if answer and answer.status == "OBSERVED" else ""

        literal_scene = " ".join(observed("initial_scene").split())
        object_answer = observed("initial_objects").split("\n\n", 1)[0]
        object_answer = object_answer.splitlines()[0] if object_answer else ""
        objects = split_atomic_items(
            object_answer, comma_separated=True, maximum=12
        )
        visible_text = split_atomic_items(observed("initial_ocr"), maximum=24)
        ocr_lines = [normalized(item) for item in visible_text]
        objects = [
            item for item in objects
            if normalized(item) not in {"text", "label", "labels"}
            and not normalized(item).endswith(" text")
            and not (
                len(normalized(item).split()) <= 2
                and any(
                    re.search(rf"\b{re.escape(normalized(item))}\b", line)
                    for line in ocr_lines
                )
            )
        ]
        visual_facts = split_atomic_items(observed("initial_facts"), maximum=8)
        visual_relations = split_atomic_items(
            observed("initial_relations"), maximum=8
        )
        text_bindings = []
        full_ocr_text = " ".join(visible_text)
        for answer in answers:
            if (
                answer.question_id.startswith("initial_region_ocr_")
                and answer.status == "OBSERVED"
            ):
                region = answer.question_id.removeprefix(
                    "initial_region_ocr_"
                ).replace("_", " ")
                region_items = split_atomic_items(answer.answer, maximum=12)
                region_items = [
                    item for item in region_items
                    if len(normalized(item)) > 1
                    or re.search(
                        rf"(?<![A-Za-z0-9]){re.escape(normalized(item))}(?![A-Za-z0-9])",
                        full_ocr_text,
                        flags=re.IGNORECASE,
                    )
                ]
                if not region_items:
                    continue
                binding = (
                    f"{region} contains text: {'; '.join(region_items)}"
                )
                text_bindings.append(binding)
                visible_text.append(binding)
        visual_relations = list(dict.fromkeys([
            *text_bindings, *visual_relations
        ]))
        visual_relations = [
            item for item in visual_relations
            if not re.search(
                r"^(?:no|none|not)\b|\bno (?:spatial )?relationship\b",
                item,
                flags=re.IGNORECASE,
            )
        ]
        symbolic_cues = split_atomic_items(
            observed("initial_symbolic_cues"), maximum=6
        )
        scene_type_words = observed("initial_scene_type").split()
        scene_type = " ".join(scene_type_words[:5]).strip() or "Unspecified"
        required_answers = [answer for answer in answers if answer.question_id in {
            "initial_scene", "initial_objects", "initial_facts", "initial_scene_type"
        }]
        format_valid = bool(
            len(required_answers) == 4 and all(answer.valid for answer in required_answers)
        )
        factual_grounding_present = bool(
            literal_scene and (objects or visual_facts or visual_relations)
        )
        schema_issues = []
        if not format_valid:
            schema_issues.append("atomic_contract_incomplete")
        if not factual_grounding_present:
            schema_issues.append("factual_grounding_missing")
        for answer in answers:
            if not answer.valid:
                schema_issues.append(
                    f"{answer.question_id}:{answer.error or answer.status.lower()}"
                )
        relation_binding_present = any(
            re.search(
                r"\b(?:left|right|top|bottom|above|below|inside|attached|"
                r"belongs|labels?|reads?|panel|next to|between)\b",
                item,
                flags=re.IGNORECASE,
            )
            for item in visual_relations
        )
        visual_description = " ".join(
            item for item in (
                literal_scene,
                *visible_text,
                *visual_facts,
                *visual_relations,
            ) if item
        ).strip()
        valid_count = sum(answer.valid for answer in answers)
        deterministic_raw = "\n".join(
            (
                f"Literal Scene: {literal_scene or 'None'}",
                f"Objects: {'; '.join(objects) or 'None'}",
                f"Visible Text: {'; '.join(visible_text) or 'None'}",
                f"Visual Facts: {'; '.join(visual_facts) or 'None'}",
                f"Visual Relations: {'; '.join(visual_relations) or 'None'}",
                f"Scene Type: {scene_type}",
                "Confidence: unavailable (model self-confidence is not used)",
            )
        )
        return {
            "visual_description": visual_description,
            "objects": objects,
            "scene_type": scene_type,
            "symbolic_tone": "; ".join(symbolic_cues) or "None",
            "visual_facts": visual_facts,
            "visual_relations": visual_relations,
            "visible_text": visible_text,
            "visible_text_count": len(visible_text),
            "visual_fact_count": len(visual_facts),
            "visual_relation_count": len(visual_relations),
            "visual_hypotheses": symbolic_cues,
            "possible_visual_metaphors": symbolic_cues,
            "uncertain_observations": [
                answer.question_id
                for answer in answers
                if answer.status in {"UNCLEAR", "INVALID_RESPONSE"}
            ],
            "visual_confidence": None,
            "schema_complete": bool(format_valid and factual_grounding_present),
            "schema_format_valid": format_valid,
            "factual_grounding_present": factual_grounding_present,
            "ocr_usable": bool(visible_text),
            "relation_binding_present": relation_binding_present,
            "schema_heading_count": 7,
            "schema_issues": list(dict.fromkeys(schema_issues)),
            "schema_source": "deterministic_atomic_adapter_v1",
            "agent1_backend": backend,
            "adapter_completion_rate": round(valid_count / len(answers), 4),
            "_schema_retry_attempted": any(
                answer.retry_attempted for answer in answers
            ),
            "_schema_retry_success": all(
                not answer.retry_attempted or answer.retry_success
                for answer in answers
            ),
            "_generation_seconds": round(
                sum(answer.elapsed_seconds for answer in answers), 4
            ),
            "_internal": {
                "raw_output": deterministic_raw,
                "atomic_answers": [answer.to_dict() for answer in answers],
            },
        }

    def _analyze_atomic(self, image, feedback=None):
        questions = list(self.question_controller.STANDARD_QUESTIONS)
        if feedback:
            feedback_question = VisualQuestion(
                "feedback_reinspection",
                "open",
                "Reinspect the complete image for a directly visible important "
                "fact, object, text binding, or relation that may have been missed. "
                "Do not assume the feedback is correct. Answer NONE if nothing new "
                "is directly visible.",
                80,
            )
            questions.append(feedback_question)
        answers = []
        print("\n========== Agent 1: atomic visual grounding ==========")
        for question in questions:
            dependency = None
            if question.question_id == "initial_ocr":
                dependency = next(
                    (item for item in answers if item.question_id == "initial_text_presence"),
                    None,
                )
            elif question.question_id == "initial_symbolic_cues":
                dependency = next(
                    (item for item in answers if item.question_id == "initial_symbolic_presence"),
                    None,
                )
            if (
                dependency
                and dependency.valid
                and re.match(r"^NO\b", dependency.answer, flags=re.IGNORECASE)
            ):
                answer = VisualAnswer(
                    question_id=question.question_id,
                    question_type=question.question_type,
                    question=question.text,
                    answer="NONE",
                    status="ABSENT",
                    valid=True,
                    error="",
                    raw_response="Skipped after validated absence gate.",
                    elapsed_seconds=0.0,
                    generation_diagnostics={"skipped_by": dependency.question_id},
                )
            else:
                answer = self._run_atomic_question(image, question)
            answers.append(answer)
            print(
                f"[{question.question_id}] {answer.status}: "
                f"{answer.answer or answer.error}"
            )
        text_presence = next(
            (item for item in answers if item.question_id == "initial_text_presence"),
            None,
        )
        if (
            text_presence
            and text_presence.valid
            and re.match(r"^YES\b", text_presence.answer, flags=re.IGNORECASE)
        ):
            for region_name, crop in self._text_relation_crops(image):
                region_id = re.sub(r"[^a-z0-9]+", "_", region_name.casefold()).strip("_")
                region_question = VisualQuestion(
                    f"initial_region_ocr_{region_id}",
                    "ocr",
                    "Transcribe every readable phrase in this cropped image "
                    "region. Copy exact visible characters only. Answer NONE "
                    "if the crop contains no readable text.",
                    120,
                )
                answer = self._run_atomic_question(crop, region_question)
                answers.append(answer)
                print(
                    f"[{region_question.question_id}] {answer.status}: "
                    f"{answer.answer or answer.error}"
                )
        output = self._atomic_schema(answers, backend=self.backend)
        feedback_answer = next(
            (item for item in answers if item.question_id == "feedback_reinspection"),
            None,
        )
        if feedback_answer and feedback_answer.status == "OBSERVED":
            output["visual_facts"] = list(dict.fromkeys([
                *output["visual_facts"], feedback_answer.answer
            ]))
            output["visual_fact_count"] = len(output["visual_facts"])
            output["visual_description"] = " ".join(
                [output["visual_description"], feedback_answer.answer]
            ).strip()
        return output

    def analyze(self, image, feedback=None):
        if getattr(self, "uses_atomic_adapter", False):
            return self._analyze_atomic(image, feedback=feedback)

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

        print("\n================ RAW LEGACY VISION OUTPUT ================\n")
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
            print("\n================ RAW LEGACY VISION RECOVERY ================\n")
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
        if getattr(self, "uses_atomic_adapter", False):
            # Preserve a complete caption-blind grounding and acquire only the
            # missing claim-targeted observation.  Repeating all nine visual
            # questions is expensive and can replace good evidence with a
            # slightly different second description.
            recovered = (
                deepcopy(visual_output)
                if (visual_output or {}).get("schema_complete", False)
                else self._analyze_atomic(image)
            )
            base_generation_seconds = float(
                recovered.get("_generation_seconds", 0.0)
            )
            subject = " ".join(
                str(claim_relation.get("subject") or "the claim subject").split()
            )
            targeted = self.answer_visual_question(
                image,
                f"What directly visible state, action, text, or relation is shown for {subject}?",
                question_type="open",
                question_id="targeted_claim_reinspection",
            )
            if targeted.get("status") == "OBSERVED":
                recovered["visual_facts"] = list(dict.fromkeys([
                    *recovered.get("visual_facts", []), targeted["answer"]
                ]))
                recovered["visual_fact_count"] = len(recovered["visual_facts"])
                recovered["visual_description"] = " ".join([
                    recovered.get("visual_description", ""), targeted["answer"]
                ]).strip()
            recovered["_targeted_recovery_attempted"] = True
            recovered["_targeted_recovery_success"] = bool(
                recovered.get("schema_complete")
                and targeted.get("status") == "OBSERVED"
            )
            recovered["_targeted_recovery_reason"] = recovery_reason
            recovered["_targeted_recovery_seconds"] = round(
                float(targeted.get("elapsed_seconds", 0.0)), 4
            )
            recovered["_generation_seconds"] = round(
                base_generation_seconds
                + float(targeted.get("elapsed_seconds", 0.0)),
                4,
            )
            recovered.setdefault("_internal", {})[
                "targeted_claim_reinspection"
            ] = targeted
            return recovered
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

        # A targeted recovery is a new observation generation.  Do not union
        # stale, disputed observations from the earlier pass into it.
        combined = self._merge_parsed(
            recovery_parsed[0] if recovery_parsed else {},
            *recovery_parsed[1:],
        )
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
        response, _ = self._generate_response(image, prompt_text, 100)
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

    @staticmethod
    def _initial_text_regions(image):
        """Use non-overlapping halves so OCR cannot leak across a binding."""
        if not hasattr(image, "size") or not hasattr(image, "crop"):
            return []
        width, height = image.size
        if width <= 1 or height <= 1:
            return []
        if width >= height:
            midpoint = width // 2
            return [
                ("left region", image.crop((0, 0, midpoint, height))),
                ("right region", image.crop((midpoint, 0, width, height))),
            ]
        midpoint = height // 2
        return [
            ("top region", image.crop((0, 0, width, midpoint))),
            ("bottom region", image.crop((0, midpoint, width, height))),
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
    def _extract_atomic_question(critique_prompt):
        match = re.search(
            r"(?im)^Review question\s*:\s*(.+)$", str(critique_prompt or "")
        )
        if match:
            return " ".join(match.group(1).split()).strip()
        match = re.search(
            r"(?im)^(?:Question|Verification request)\s*:\s*(.+)$",
            str(critique_prompt or ""),
        )
        if match:
            return " ".join(match.group(1).split()).strip()
        return "What exact visible state or relation is shown for the claim subject?"

    @staticmethod
    def _question_id(critique_prompt):
        match = re.search(
            r"(?im)^Question ID\s*:\s*([A-Za-z0-9_.-]+)",
            str(critique_prompt or ""),
        )
        return match.group(1) if match else "agent1_visual_observation"

    @staticmethod
    def _claim_frame_excerpt(critique_prompt):
        wanted = (
            "Original caption", "Claim subject", "Claim predicate",
            "Asserted property", "Expected visual state",
            "Opposite visual state", "Intended meaning", "Relation family",
        )
        lines = []
        for name in wanted:
            match = re.search(
                rf"(?im)^\s*{re.escape(name)}\s*:\s*(.+)$",
                str(critique_prompt or ""),
            )
            if match:
                lines.append(f"{name}: {' '.join(match.group(1).split())}")
        return "\n".join(lines) or "Claim frame: unavailable"

    @staticmethod
    def _parse_observation_response(response):
        headings = (
            "Observation Status", "Observed Entity", "Observed State",
            "Image Region", "Reason",
        )

        def field(name):
            following = "|".join(re.escape(item) for item in headings)
            match = re.search(
                rf"(?ims)^\s*{re.escape(name)}\s*:\s*(.*?)"
                rf"(?=^\s*(?:{following})\s*:|\Z)",
                str(response or ""),
            )
            return " ".join(match.group(1).split()).strip() if match else ""

        status_match = re.search(
            r"\b(OBSERVED|UNCLEAR)\b", field("Observation Status"),
            flags=re.IGNORECASE,
        )
        status = status_match.group(1).upper() if status_match else ""
        entity = field("Observed Entity")
        state = field("Observed State")
        region = field("Image Region")
        reason = field("Reason")
        placeholders = {"", "none", "unknown", "unavailable"}
        if status == "UNCLEAR":
            format_valid = bool(status_match and reason)
        else:
            format_valid = bool(
                status == "OBSERVED"
                and entity.casefold() not in placeholders
                and state.casefold() not in placeholders
                and region.casefold() not in placeholders
                and reason
            )
        return {
            "observation_status": status or "INVALID",
            "observed_entity": entity,
            "observed_state": state,
            "image_region": region,
            "reason": reason,
            "_format_valid": format_valid,
            "_raw_response": str(response or "").strip(),
        }

    @staticmethod
    def _parse_relation_response(response):
        raw = str(response or "").strip()
        text = re.sub(
            r"^```(?:text)?\s*|\s*```$", "", raw,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        # Local instruction-tuned models commonly omit a requested field name
        # while still returning the exact bounded enum.  Canonicalize that
        # harmless surface variation, but reject explanations, multiple lines,
        # and every value outside the closed relation vocabulary.
        match = re.fullmatch(
            r"(?:Claim\s+Relation\s*:\s*)?"
            r"(SUPPORT|CONFLICT|UNRESOLVED)\s*[.!]?",
            text,
            flags=re.IGNORECASE,
        )
        return {
            "claim_relation": match.group(1).upper() if match else "UNRESOLVED",
            "_format_valid": bool(match),
            "_invalid_enum": bool(text and not match),
            "_normalized_from_bare_enum": bool(
                match and not re.match(
                    r"^Claim\s+Relation\s*:", text, flags=re.IGNORECASE
                )
            ),
            "_raw_response": raw,
        }

    @staticmethod
    def _response_status(observation, relation, diagnostics):
        if (diagnostics or {}).get("hit_token_limit"):
            return "TRUNCATED_RESPONSE"
        if not observation.get("_raw_response"):
            return "QUESTION_NOT_ANSWERED"
        if relation.get("_invalid_enum"):
            return "INVALID_ENUM"
        if (
            observation.get("observation_status") == "UNCLEAR"
            and relation.get("claim_relation") in {"SUPPORT", "CONFLICT"}
        ):
            return "INCONSISTENT_FIELDS"
        if not observation.get("_format_valid") or not relation.get("_format_valid"):
            return "FORMAT_FAILURE"
        if observation.get("observation_status") == "UNCLEAR":
            return "VALID_ABSTENTION"
        value = relation.get("claim_relation")
        if value not in {"SUPPORT", "CONFLICT", "UNRESOLVED"}:
            return "INVALID_ENUM"
        if value == "UNRESOLVED":
            return "SEMANTICALLY_UNRESOLVED"
        return "VALID_DIRECTIONAL_ANSWER"

    @staticmethod
    def _critique_quality(parsed):
        return sum(bool(parsed.get(field)) for field in (
            "observed_entity", "observed_state", "image_region", "reason"
        )) + 4 * int(bool(parsed.get("_format_valid"))) + 2 * int(
            bool(parsed.get("specific_evidence"))
        )

    @staticmethod
    def _prompt_field(prompt, name):
        match = re.search(
            rf"(?im)^\s*{re.escape(name)}\s*:\s*(.+)$", str(prompt or "")
        )
        return " ".join(match.group(1).split()) if match else ""

    def _atomic_critique(self, image, critique_prompt):
        question = self._extract_atomic_question(critique_prompt)
        question_id = self._question_id(critique_prompt)
        question_type = self.question_controller.infer_question_type(question)
        answer = self.answer_visual_question(
            image,
            question,
            question_type=question_type,
            question_id=question_id,
        )
        answer_status = answer.get("status")
        observation_status = (
            "OBSERVED" if answer_status == "OBSERVED"
            else "ABSENT" if answer_status == "ABSENT"
            else "UNCLEAR"
        )
        subject = self._prompt_field(critique_prompt, "Claim subject") or (
            "question-targeted visual entity"
        )
        observed_state = answer.get("answer") if observation_status == "OBSERVED" else ""
        relation_value = "UNRESOLVED"
        relation_raw = ""
        relation_seconds = 0.0
        relation_diagnostics = {}
        recommendation = "ABSTAIN"
        relation_parsed = {
            "claim_relation": "UNRESOLVED",
            "_format_valid": True,
            "_invalid_enum": False,
            "_raw_response": "Claim Relation: UNRESOLVED",
        }
        relation_retry_raw = ""
        if answer.get("valid") and observation_status == "OBSERVED":
            relation_prompt = f"""
Inspect the image and use the factual observation below. Compare it with the
immutable caption conditions. Do not use absence as conflict and do not guess.

Factual observation: {observed_state}
{self._claim_frame_excerpt(critique_prompt)}

Return exactly one line:
Claim Relation: SUPPORT, CONFLICT, or UNRESOLVED

SUPPORT means the observed state matches the expected caption condition.
CONFLICT means the observed state directly matches the stated opposite.
UNRESOLVED means neither direction is directly established.
"""
            relation_raw, relation_seconds = self._generate_response(
                image, relation_prompt, 20
            )
            relation_diagnostics = dict(
                getattr(self, "_last_generation_diagnostics", {}) or {}
            )
            relation_parsed = self._parse_relation_response(relation_raw)
            if (
                not relation_parsed.get("_format_valid", False)
                or relation_diagnostics.get("hit_token_limit", False)
            ):
                retry_raw, retry_seconds = self._generate_response(
                    image,
                    relation_prompt
                    + "\nYour previous format was invalid. Output only the one Claim Relation line.",
                    12,
                )
                retry_parsed = self._parse_relation_response(retry_raw)
                relation_retry_raw = retry_raw
                relation_seconds += retry_seconds
                if retry_parsed.get("_format_valid", False):
                    relation_raw = retry_raw
                    relation_parsed = retry_parsed
                    relation_diagnostics = dict(
                        getattr(self, "_last_generation_diagnostics", {}) or {}
                    )
            relation_value = relation_parsed.get(
                "claim_relation", "UNRESOLVED"
            )
            recommendation = {
                "SUPPORT": "ENTAILS",
                "CONFLICT": "CONTRADICTS",
            }.get(relation_value, "ABSTAIN")

        observation_contract = {
            "observation_status": (
                "OBSERVED" if observation_status == "OBSERVED" else "UNCLEAR"
            ),
            "_format_valid": bool(answer.get("valid")),
            "_raw_response": answer.get("raw_response", ""),
        }
        response_status = self._response_status(
            observation_contract, relation_parsed, relation_diagnostics
        )
        format_valid = bool(
            answer.get("valid") and relation_parsed.get("_format_valid", False)
        )
        parser_errors = []
        observation_error = str(answer.get("error") or "").strip()
        if not answer.get("valid"):
            parser_errors.append(
                "observation:" + (observation_error or "invalid_answer")
            )
        if not relation_parsed.get("_format_valid", False):
            parser_errors.append("relation:invalid_relation_enum")
        return {
            "stance": "UNRESOLVED",
            "recommendation": recommendation,
            "reason": (
                f"Atomic visual answer: {observed_state}"
                if observed_state else answer.get("error") or "Visual detail is unclear."
            ),
            "specific_evidence": bool(
                observation_status == "OBSERVED"
                and relation_value in {"SUPPORT", "CONFLICT"}
            ),
            "observed_entity": subject if observation_status == "OBSERVED" else "",
            "observed_state": observed_state,
            "image_region": question_region(question) if observation_status == "OBSERVED" else "",
            "claim_relation": relation_value,
            "witness_contract": {
                "schema_version": "1.0",
                "question_id": question_id,
                "question_type": question_type,
                "question": question,
                "answer_status": answer_status,
                "entity_present": observation_status == "OBSERVED",
                "observation": observed_state,
                "region": (
                    question_region(question)
                    if observation_status == "OBSERVED" else ""
                ),
                "direction_assigned": relation_value in {"SUPPORT", "CONFLICT"},
                "relation_candidate": relation_value,
            },
            "review_method": self._critique_review_method(critique_prompt),
            "observation_status": observation_status,
            "response_status": response_status,
            "question_id": question_id,
            "question": question,
            "parser_errors": parser_errors,
            "observation_parser_error": observation_error,
            "relation_parser_error": (
                "" if relation_parsed.get("_format_valid", False)
                else "invalid_relation_enum"
            ),
            "generation_diagnostics": {
                "observation": answer.get("generation_diagnostics", {}),
                "relation": relation_diagnostics,
            },
            "_format_valid": format_valid,
            "_format_retry_used": bool(answer.get("retry_attempted")),
            "_format_retry_success": bool(answer.get("retry_success")),
            "_raw_response": "\n".join(
                item for item in (answer.get("raw_response", ""), relation_raw) if item
            ),
            "_raw_primary_response": answer.get("raw_response", ""),
            "_raw_retry_response": "",
            "_raw_relation_response": relation_raw,
            "_raw_relation_retry_response": relation_retry_raw,
            "_region_ocr_candidates": [],
            "_generation_seconds": round(
                float(answer.get("elapsed_seconds", 0.0)) + relation_seconds, 4
            ),
        }

    def critique(self, image, critique_prompt):
        if getattr(self, "uses_atomic_adapter", False):
            return self._atomic_critique(image, critique_prompt)

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
                    "observation_status": "OBSERVED",
                    "response_status": "VALID_ABSTENTION",
                    "question_id": "agent1_region_binding",
                    "question": (
                        "Which exact text belongs to each visible comparison region?"
                    ),
                    "parser_errors": [],
                    "generation_diagnostics": {},
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
            for region_name, crop in self._initial_text_regions(image):
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

        question = self._extract_atomic_question(critique_prompt)
        question_id = self._question_id(critique_prompt)
        claim_frame = self._claim_frame_excerpt(critique_prompt)
        prompt_text = f"""
Inspect the image and answer only this one visual question:
{question}

Do not choose ENTAILS or CONTRADICTS. Do not repeat the supplied claim. Report
only a directly visible state, action, text attachment, symbol attachment, or
spatial relation. If the requested detail cannot be seen, use UNCLEAR.

Return exactly these five single-line fields:
Observation Status: OBSERVED or UNCLEAR
Observed Entity: exact visible person, object, label, or symbol; None if unclear
Observed State: exact visible state or relation; None if unclear
Image Region: exact visible location; None if unclear
Reason: short factual observation or why the detail is unclear
"""
        print("\n========== Agent 1 Critique ==========")

        response, primary_seconds = self._generate_response(
            image, prompt_text, 110
        )
        primary_diagnostics = dict(
            getattr(self, "_last_generation_diagnostics", {}) or {}
        )

        print("\n================ AGENT 1 OBSERVATION ================\n")
        print(response)
        print("\n=================================================\n")
        review_method = self._critique_review_method(critique_prompt)
        observation = self._parse_observation_response(response)
        observation_retry_response = ""
        retry_seconds = 0.0
        retry_used = not observation["_format_valid"]
        if retry_used:
            retry_prompt = f"""
Repair only the missing visual-observation fields for this one question:
{question}

Do not classify the caption. If the evidence is not visible, use UNCLEAR.
Return exactly:
Observation Status: OBSERVED or UNCLEAR
Observed Entity: exact visible entity, or None
Observed State: exact visible state or relation, or None
Image Region: exact visible location, or None
Reason: one short factual sentence
"""
            observation_retry_response, retry_seconds = self._generate_response(
                image, retry_prompt, 90
            )
            retry_diagnostics = dict(
                getattr(self, "_last_generation_diagnostics", {}) or {}
            )
            retry_observation = self._parse_observation_response(
                observation_retry_response
            )
            if retry_observation["_format_valid"]:
                observation = retry_observation
        else:
            retry_diagnostics = {}

        relation_response = ""
        relation_retry_response = ""
        relation_seconds = 0.0
        relation_retry_seconds = 0.0
        if observation.get("observation_status") == "UNCLEAR":
            relation = {
                "claim_relation": "UNRESOLVED",
                "_format_valid": True,
                "_raw_response": "Claim Relation: UNRESOLVED",
            }
            relation_diagnostics = {}
            relation_retry_diagnostics = {}
        elif observation.get("_format_valid"):
            relation_prompt = f"""
Classify the supplied direct visual observation against the immutable claim
frame. Do not add visual details and do not choose a final dataset label.

{claim_frame}

Observed Entity: {observation.get('observed_entity')}
Observed State: {observation.get('observed_state')}
Image Region: {observation.get('image_region')}

Return exactly one line:
Claim Relation: SUPPORT, CONFLICT, or UNRESOLVED
"""
            relation_response, relation_seconds = self._generate_response(
                image, relation_prompt, 28
            )
            relation_diagnostics = dict(
                getattr(self, "_last_generation_diagnostics", {}) or {}
            )
            relation = self._parse_relation_response(relation_response)
            if not relation["_format_valid"]:
                relation_retry_prompt = f"""
Use only this observation: {observation.get('observed_entity')} —
{observation.get('observed_state')} in {observation.get('image_region')}.

{claim_frame}

Return one line only: Claim Relation: SUPPORT, CONFLICT, or UNRESOLVED
"""
                relation_retry_response, relation_retry_seconds = (
                    self._generate_response(image, relation_retry_prompt, 20)
                )
                relation_retry_diagnostics = dict(
                    getattr(self, "_last_generation_diagnostics", {}) or {}
                )
                retry_relation = self._parse_relation_response(
                    relation_retry_response
                )
                if retry_relation["_format_valid"]:
                    relation = retry_relation
            else:
                relation_retry_diagnostics = {}
        else:
            relation = {
                "claim_relation": "UNRESOLVED",
                "_format_valid": False,
                "_raw_response": "",
            }
            relation_diagnostics = {}
            relation_retry_diagnostics = {}

        relation_value = relation.get("claim_relation", "UNRESOLVED")
        recommendation = {
            "SUPPORT": "ENTAILS",
            "CONFLICT": "CONTRADICTS",
            "UNRESOLVED": "ABSTAIN",
        }.get(relation_value, "ABSTAIN")
        combined_diagnostics = {
            "observation_primary": primary_diagnostics,
            "observation_retry": retry_diagnostics,
            "relation_primary": relation_diagnostics,
            "relation_retry": relation_retry_diagnostics,
        }
        any_truncated = any(
            item.get("hit_token_limit")
            for item in combined_diagnostics.values() if item
        )
        status_diagnostics = {"hit_token_limit": any_truncated}
        response_status = self._response_status(
            observation, relation, status_diagnostics
        )
        format_valid = bool(
            observation.get("_format_valid") and relation.get("_format_valid")
        )
        specific_evidence = bool(
            format_valid
            and response_status == "VALID_DIRECTIONAL_ANSWER"
            and observation.get("observation_status") == "OBSERVED"
        )
        parsed = {
            "stance": "UNRESOLVED",
            "recommendation": recommendation,
            "reason": observation.get("reason") or "Visual question was not answered.",
            "specific_evidence": specific_evidence,
            "observed_entity": observation.get("observed_entity", ""),
            "observed_state": observation.get("observed_state", ""),
            "image_region": observation.get("image_region", ""),
            "claim_relation": relation_value,
            "review_method": review_method,
            "observation_status": observation.get("observation_status"),
            "response_status": response_status,
            "question_id": question_id,
            "question": question,
            "parser_errors": [
                name for name, valid in (
                    ("observation_contract_invalid", observation.get("_format_valid")),
                    ("relation_contract_invalid", relation.get("_format_valid")),
                ) if not valid
            ],
            "generation_diagnostics": combined_diagnostics,
            "_format_valid": format_valid,
            "_raw_response": "\n".join(
                item for item in (response, relation_response) if item
            ).strip(),
        }
        parsed.update({
            "_format_retry_used": retry_used,
            "_format_retry_success": bool(retry_used and parsed["_format_valid"]),
            "_raw_primary_response": response.strip(),
            "_raw_retry_response": observation_retry_response.strip(),
            "_raw_relation_response": relation_response.strip(),
            "_raw_relation_retry_response": relation_retry_response.strip(),
            "_region_ocr_candidates": crop_ocr_candidates,
            "_generation_seconds": round(
                primary_seconds + retry_seconds + relation_seconds
                + relation_retry_seconds + crop_ocr_seconds,
                4,
            ),
        })
        return parsed
    
