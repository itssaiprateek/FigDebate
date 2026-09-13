"""Pinned local Qwen runtime used only by the optional judge stage."""

import gc
import math
import os
import time
from engine.runtime_accounting import record_generation
from engine.structured_decoder import DECODER_ID

from engine.runtime_profile import resolve_runtime_profile


JUDGE_MODEL_ID = "Qwen/Qwen3.5-4B"
JUDGE_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
JUDGE_MODEL_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
JUDGE_MODEL_DIRECTORY = os.path.join("models", "judge", "Qwen3.5-4B")
JUDGE_MODEL_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
)


def default_judge_model_path():
    configured = os.environ.get("FIGDEBATE_MODEL_ROOT")
    if configured:
        return os.path.join(os.path.abspath(configured), "judge", "Qwen3.5-4B")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, JUDGE_MODEL_DIRECTORY)


class QwenJudgeModel:
    """Load the independent multimodal judge locally in deterministic 4-bit mode."""

    def __init__(self, model_path=None, hardware_profile="8gb"):
        try:
            import torch
            from transformers import (
                AutoModelForMultimodalLM,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as error:
            raise RuntimeError(
                "The Qwen judge requires the validated FigDebate environment. "
                "Run check_environment.py before inference."
            ) from error

        self.hardware_profile = resolve_runtime_profile(hardware_profile)
        self.model_path = os.path.abspath(model_path or default_judge_model_path())
        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.isfile(config_path):
            raise RuntimeError(
                "The local judge model is missing. Expected config.json under "
                f"{self.model_path}. Download the pinned {JUDGE_MODEL_ID} revision first."
            )
        metadata_path = os.path.join(
            self.model_path,
            ".cache",
            "huggingface",
            "download",
            "config.json.metadata",
        )
        if not os.path.isfile(metadata_path):
            raise RuntimeError(
                "The judge revision metadata is missing. Re-download the model with "
                "the pinned revision before running a judge experiment."
            )
        with open(metadata_path, "r", encoding="utf-8") as handle:
            local_revision = handle.readline().strip()
        if local_revision != JUDGE_MODEL_REVISION:
            raise RuntimeError(
                f"Judge revision mismatch: {local_revision or 'missing'}; "
                f"expected {JUDGE_MODEL_REVISION}."
            )

        print(f"Loading independent Qwen judge from {self.model_path}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            min_pixels=65_536,
            max_pixels=self.hardware_profile.judge_max_pixels,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self._last_generation_diagnostics = {}
        print("Independent Qwen judge loaded")

    def count_text_tokens(self, text):
        return len(self.processor.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _release_generation_memory(torch):
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"[Judge][cleanup-warning] {str(error).splitlines()[0]}")
                return False
        return True

    @staticmethod
    def _image_size(image):
        size = getattr(image, "size", None)
        if not size or len(size) != 2:
            return None
        return int(size[0]), int(size[1])

    @classmethod
    def _fit_image_to_pixel_budget(cls, image, max_pixels):
        size = cls._image_size(image)
        if not size or not max_pixels:
            return image, False
        width, height = size
        if width * height <= int(max_pixels):
            return image, False
        scale = math.sqrt(float(max_pixels) / float(width * height))
        target = (max(1, int(width * scale)), max(1, int(height * scale)))
        from PIL import Image

        return image.resize(target, Image.Resampling.LANCZOS), True

    @staticmethod
    def _is_cuda_oom(error, torch):
        return isinstance(error, torch.OutOfMemoryError) or (
            isinstance(error, RuntimeError)
            and "out of memory" in str(error).casefold()
        )

    @staticmethod
    def _cuda_memory(torch):
        if not torch.cuda.is_available():
            return {}
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            divisor = 1024 ** 3
            return {
                "allocated_gb": round(
                    torch.cuda.memory_allocated() / divisor, 4
                ),
                "reserved_gb": round(
                    torch.cuda.memory_reserved() / divisor, 4
                ),
                "free_gb": round(free_bytes / divisor, 4),
                "total_gb": round(total_bytes / divisor, 4),
            }
        except RuntimeError:
            return {}

    def _install_prefill_cleanup(self, torch, use_cache):
        """Release unused prefill allocations once; retain weights and KV state.

        The matched diagnostic established memory headroom, not a decode-speed
        improvement. This never lowers image resolution or changes model inputs.
        """
        enabled = bool(getattr(self.hardware_profile, "judge_release_prefill_workspace", False))
        measurement = {"enabled": enabled, "performed": False}
        if not enabled or not use_cache or not torch.cuda.is_available():
            return None, measurement

        def after_forward(module, args, output):
            if measurement["performed"]:
                return
            measurement["performed"] = True
            measurement["before"] = self._cuda_memory(torch)
            started = time.perf_counter()
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                measurement["succeeded"] = True
            finally:
                measurement["cleanup_seconds"] = time.perf_counter() - started
                measurement["after"] = self._cuda_memory(torch)

        return self.model.register_forward_hook(after_forward), measurement

    def _generate_once(
        self, image, prompt, max_new_tokens, *, use_cache
    ):
        """Generate once and surface asynchronous CUDA failures immediately."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        from engine.output_contracts import prefix_constraint

        class Deadline(StoppingCriteria):
            def __init__(self, seconds):
                self.started = time.perf_counter()
                self.seconds = seconds
                self.expired = False
                self.first_token_seconds = None
                self.steps = 0

            def __call__(self, input_ids, scores, **kwargs):
                elapsed = time.perf_counter() - self.started
                if self.first_token_seconds is None:
                    self.first_token_seconds = elapsed
                self.steps += 1
                self.expired = elapsed > self.seconds
                return self.expired

        inputs = None
        generated = None
        completion_ids = None
        prefill_hook = None
        prefill_cleanup = {}
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an independent multimodal evidence reviewer. "
                    "Follow the role, evidence rules, and JSON contract in "
                    "the user instruction exactly."
                ),
            },
            {
                "role": "user",
                "content": [
                    *([{"type": "image", "image": image}] if image is not None else []),
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            prompt_length = int(inputs["input_ids"].shape[1])
            total_limit = self.hardware_profile.judge_total_tokens
            self._last_generation_diagnostics = {
                "input_tokens": prompt_length, "max_new_tokens": int(max_new_tokens),
                "total_token_budget": total_limit,
                "image_size": list(self._image_size(image) or ()),
                "torch_seed": torch.initial_seed(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "parameter_devices": sorted({str(p.device) for p in self.model.parameters()}),
                "quantization": str(getattr(self.model.config, "quantization_config", None)),
                "cold_call": not getattr(self, "_generation_started_before", False),
            }
            from engine.judge_telemetry import runtime_snapshot
            self._last_generation_diagnostics["runtime_before"] = runtime_snapshot()
            provenance = dict(self._last_generation_diagnostics)
            self._generation_started_before = True
            if prompt_length + int(max_new_tokens) > total_limit:
                raise ValueError(
                    f"context budget exceeded: input={prompt_length}, "
                    f"output={max_new_tokens}, limit={total_limit}"
                )
            inputs = inputs.to(self.device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            from engine.case_budget import require_time
            remaining = require_time(self)
            deadline = Deadline(min(self.hardware_profile.judge_max_seconds, remaining)
                                if remaining is not None else self.hardware_profile.judge_max_seconds)
            generation_options = {"stopping_criteria": StoppingCriteriaList([deadline])}
            callback_seconds = {}
            schema = getattr(self, "_active_output_schema", None)
            if schema:
                from engine.judge_telemetry import measured_callback
                generation_options["prefix_allowed_tokens_fn"] = measured_callback(prefix_constraint(
                    self.processor.tokenizer, schema, compact=self.hardware_profile.judge_compact_json),
                    callback_seconds, "grammar")
                from engine.output_contracts import complete_json_stopper
                generation_options["stopping_criteria"].append(
                    measured_callback(complete_json_stopper(self.processor.tokenizer, prompt_length, schema),
                                      callback_seconds, "completion_check"))
            prefill_hook, prefill_cleanup = self._install_prefill_cleanup(torch, use_cache)
            self._last_generation_diagnostics["prefill_workspace_cleanup"] = prefill_cleanup
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    **generation_options,
                    use_cache=bool(use_cache),
                )
            if deadline.expired:
                # Keep failure diagnostics, never promote partial text to proof.
                partial_ids = generated[:, prompt_length:]
                self._last_generation_diagnostics.update(
                    timeout_seconds=deadline.seconds,
                    elapsed_seconds=time.perf_counter() - started,
                    termination_reason="TIMEOUT",
                    generated_tokens=int(partial_ids.shape[-1]),
                    first_token_seconds=deadline.first_token_seconds,
                    decode_steps=deadline.steps,
                    memory_at_timeout=self._cuda_memory(torch),
                    partial_response=self.processor.batch_decode(partial_ids,
                        skip_special_tokens=True, clean_up_tokenization_spaces=False)[0],
                    use_cache=bool(use_cache), partial_output_discarded=True)
                self._last_generation_diagnostics["runtime_after"] = runtime_snapshot()
                self._last_generation_diagnostics["callback_seconds"] = callback_seconds
                del partial_ids
                raise TimeoutError("judge generation timed out before a complete qualified response")
            # Generation kernels are asynchronous.  Synchronizing here makes
            # an OOM belong to this attempt instead of a later cleanup/seed.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            prompt_length = int(inputs["input_ids"].shape[1])
            completion_ids = generated[:, prompt_length:]
            response = self.processor.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            generated_tokens = int(completion_ids.shape[-1])
            diagnostics = {
                **provenance,
                "decoder": DECODER_ID if schema else None,
                "compact_json": self.hardware_profile.judge_compact_json,
                "runtime_after": runtime_snapshot(),
                "callback_seconds": callback_seconds,
                "input_tokens": prompt_length,
                "generated_tokens": generated_tokens,
                "max_new_tokens": int(max_new_tokens),
                "hit_token_limit": generated_tokens >= int(max_new_tokens),
                "termination_reason": "TOKEN_LIMIT" if generated_tokens >= int(max_new_tokens) else "STOPPED",
                "first_token_seconds": deadline.first_token_seconds,
                "decode_steps": deadline.steps,
                "peak_allocated_gb": (
                    round(torch.cuda.max_memory_allocated() / 1024 ** 3, 4)
                    if torch.cuda.is_available() else None
                ),
                "elapsed_seconds": round(elapsed, 4),
                "use_cache": bool(use_cache),
                "prefill_workspace_cleanup": prefill_cleanup,
                "image_size": list(self._image_size(image) or ()),
            }
            return response, elapsed, diagnostics
        finally:
            if prefill_hook is not None:
                prefill_hook.remove()
            completion_ids = None
            generated = None
            inputs = None
            messages = None
            self._release_generation_memory(torch)

    @record_generation("Qwen3.5-4B")
    def generate(self, image, prompt, max_new_tokens=None, json_schema=None):
        import torch

        self._active_output_schema = json_schema
        if max_new_tokens is None:
            max_new_tokens = self.hardware_profile.judge_output_tokens

        if hasattr(image, "convert"):
            image = image.convert("RGB")
        original_size = self._image_size(image)
        primary_budget = self.hardware_profile.judge_max_pixels
        primary_image, primary_resized = self._fit_image_to_pixel_budget(
            image, primary_budget
        )
        primary_mode = (
            "bounded_full_frame" if primary_resized else "full_detail"
        )
        # A bounded context is qualified with cache first. Cache-free fallback
        # retains identical inputs but has its own explicit wall-time bound.
        attempts = [(primary_image, True, primary_mode)]
        attempts.append(
            (primary_image, False, f"{primary_mode}_cache_free")
        )
        for budget in self.hardware_profile.oom_retry_pixels:
            if int(budget) >= int(primary_budget):
                continue
            retry_image, resized = self._fit_image_to_pixel_budget(image, budget)
            if resized:
                attempts.append(
                    (retry_image, False, f"oom_scaled_{budget}_pixels")
                )

        failures = []
        self._last_generation_diagnostics = {}
        for attempt_number, (attempt_image, use_cache, mode) in enumerate(
            attempts, start=1
        ):
            self._release_generation_memory(torch)
            memory_before = self._cuda_memory(torch)
            try:
                response, elapsed, diagnostics = self._generate_once(
                    attempt_image,
                    prompt,
                    max_new_tokens,
                    use_cache=use_cache,
                )
            except (torch.OutOfMemoryError, RuntimeError) as error:
                if not self._is_cuda_oom(error, torch):
                    self._last_generation_diagnostics.update({
                        "failed_attempts": failures, "error": str(error),
                        "memory_before": memory_before,
                    })
                    raise
                failures.append({
                    "attempt": attempt_number,
                    "mode": mode,
                    "image_size": list(
                        self._image_size(attempt_image) or ()
                    ),
                    "memory_before": memory_before,
                    "error": str(error).splitlines()[0],
                })
                self._release_generation_memory(torch)
                print(
                    "[Judge][VRAM] CUDA OOM during "
                    f"{mode}; temporary tensors cleared."
                )
                continue
            diagnostics.update({
                "attempt": attempt_number,
                "inference_mode": mode,
                "hardware_profile": self.hardware_profile.name,
                "judge_max_pixels": primary_budget,
                "original_image_size": list(original_size or ()),
                "resolution_reduced": bool(
                    primary_resized or mode.startswith("oom_scaled_")
                ),
                "operating_limit_applied": bool(primary_resized),
                "oom_recovery_used": bool(failures),
                "memory_before": memory_before,
                "memory_after_cleanup": self._cuda_memory(torch),
                "failed_attempts": failures,
            })
            self._last_generation_diagnostics = diagnostics
            if mode.startswith("oom_scaled_"):
                print(
                    "[Judge][VRAM] Recovered with adaptive image budget: "
                    f"{diagnostics['image_size']} (original "
                    f"{diagnostics['original_image_size']})."
                )
            return response, elapsed

        self._last_generation_diagnostics = {
            "hardware_profile": self.hardware_profile.name,
            "judge_max_pixels": primary_budget,
            "original_image_size": list(original_size or ()),
            "oom_recovery_used": True,
            "resolution_reduced": any(
                mode.startswith("oom_scaled_")
                for _, _, mode in attempts
            ),
            "failed_attempts": failures,
            "memory_after_cleanup": self._cuda_memory(torch),
        }
        raise RuntimeError(
            "The Qwen judge could not process this case within the configured "
            "VRAM profile after cache-free and adaptive-resolution attempts."
        )
