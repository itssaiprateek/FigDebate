"""Pinned Qwen3-VL runtime used by FigDebate Agent 1."""

from __future__ import annotations

import gc
import math
from pathlib import Path
import time


VISION_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
VISION_MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
VISION_MODEL_DIRECTORY = "Qwen3-VL-4B-Instruct"
VISION_MODEL_ARCHITECTURE = "Qwen3VLForConditionalGeneration"
# Images up to this measured 8 GB operating limit retain their original pixels.
# Larger full frames are bounded, while OCR relation crops are still created
# from the original image by VisualGroundingAgent.
PRIMARY_MAX_PIXELS = 2_359_296
# These lower ceilings are used only after CUDA reports an out-of-memory error.
OOM_RETRY_MAX_PIXELS = (1_048_576, 589_824, 262_144)
VISION_MODEL_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


def local_vision_model_path() -> Path:
    return Path(__file__).resolve().parent / "vision" / VISION_MODEL_DIRECTORY


def vision_model_source():
    """Prefer the prepared project-local model and pin any Hub fallback."""
    local_path = local_vision_model_path()
    if all((local_path / filename).is_file() for filename in VISION_MODEL_FILES):
        return str(local_path), {"local_files_only": True}
    return VISION_MODEL_ID, {"revision": VISION_MODEL_REVISION}


class Qwen3VLVisionModel:
    """Load Qwen3-VL 4B once in NF4 and expose deterministic generation."""

    backend = "qwen3_vl_4b_instruct"
    supports_atomic_questions = True
    model_id = VISION_MODEL_ID
    model_revision = VISION_MODEL_REVISION
    quantization = "nf4_4bit_double_quant_bf16_compute"

    def __init__(self):
        try:
            import torch
            from transformers import (
                AutoModelForMultimodalLM,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as error:
            raise RuntimeError(
                "Qwen3-VL requires the validated PyTorch, Transformers, "
                "Accelerate, and bitsandbytes runtime. Run "
                "setup_environment.py before inference."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen3-VL requires a CUDA-capable GPU in the validated "
                "FigDebate runtime."
            )

        source, source_kwargs = vision_model_source()
        print(f"Loading Qwen3-VL 4B Instruct from {source} in 4-bit NF4...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.processor = AutoProcessor.from_pretrained(source, **source_kwargs)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            source,
            **source_kwargs,
            quantization_config=quantization_config,
            device_map="auto",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        if next(self.model.parameters()).device.type != "cuda":
            raise RuntimeError(
                "Qwen3-VL was not placed on CUDA. Close other GPU programs "
                "and verify the validated 4-bit environment."
            )
        device_map = getattr(self.model, "hf_device_map", {}) or {}
        if any(str(device) in {"cpu", "disk"} for device in device_map.values()):
            raise RuntimeError(
                "Qwen3-VL was partially offloaded to CPU or disk; the "
                "validated FigDebate runtime requires a complete CUDA load."
            )
        self._last_generation_diagnostics = {}
        print("Qwen3-VL 4B Instruct loaded")

    @staticmethod
    def _image_size(image):
        size = getattr(image, "size", None)
        if isinstance(size, tuple) and len(size) == 2:
            return int(size[0]), int(size[1])
        return None

    @classmethod
    def _fit_image_to_pixel_budget(cls, image, max_pixels):
        """Return an aspect-preserving copy only when the image exceeds a budget."""
        size = cls._image_size(image)
        if not size:
            return image, False
        width, height = size
        pixels = width * height
        if pixels <= int(max_pixels):
            return image, False
        scale = math.sqrt(float(max_pixels) / float(pixels))
        target = (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        )
        from PIL import Image

        return image.resize(target, Image.Resampling.LANCZOS), True

    @staticmethod
    def _cuda_memory(torch):
        if not torch.cuda.is_available():
            return {}
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        divisor = 1024 ** 3
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / divisor, 4),
            "reserved_gb": round(torch.cuda.memory_reserved() / divisor, 4),
            "free_gb": round(free_bytes / divisor, 4),
            "total_gb": round(total_bytes / divisor, 4),
        }

    @staticmethod
    def _release_cuda(torch):
        """Release only unreachable temporary tensors; model weights stay loaded."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_generation_memory(self):
        """Public sample-boundary cleanup used by the stagewise runner."""
        import torch

        self._release_cuda(torch)

    def _generate_once(self, image, prompt, max_new_tokens, *, use_cache):
        """Run one attempt and always release its temporary CUDA tensors."""
        import torch

        inputs = None
        generated = None
        completion_ids = None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": str(prompt)},
                ],
            }
        ]
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    use_cache=bool(use_cache),
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            prompt_length = int(inputs["input_ids"].shape[-1])
            completion_ids = generated[:, prompt_length:]
            answer = self.processor.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            generated_count = int(completion_ids.shape[-1])
            diagnostics = {
                "backend": self.backend,
                "quantization": self.quantization,
                "generated_tokens": generated_count,
                "max_new_tokens": int(max_new_tokens),
                "hit_token_limit": generated_count >= int(max_new_tokens),
                "elapsed_seconds": round(elapsed, 4),
                "use_cache": bool(use_cache),
                "input_tokens": prompt_length,
                "image_size": list(self._image_size(image) or ()),
                "peak_allocated_gb": (
                    round(torch.cuda.max_memory_allocated() / 1024 ** 3, 4)
                    if torch.cuda.is_available() else None
                ),
            }
            return answer, elapsed, diagnostics
        finally:
            # Explicit deletion matters in a long atomic-question sequence:
            # CUDA's allocator cannot reuse live references even after empty_cache.
            completion_ids = None
            generated = None
            inputs = None
            messages = None
            self._release_cuda(torch)

    def generate(self, image, prompt, max_new_tokens=96):
        """Generate deterministically within the measured 8 GB visual budget."""
        import torch

        original_size = self._image_size(image)
        primary_image, primary_resized = self._fit_image_to_pixel_budget(
            image, PRIMARY_MAX_PIXELS
        )
        primary_mode = (
            "bounded_full_frame" if primary_resized else "full_detail"
        )
        attempts = [
            (primary_image, True, primary_mode),
            # A cache-free retry preserves the identical prepared image while
            # reducing generation-state memory. It is slower, not lower quality.
            (primary_image, False, f"{primary_mode}_cache_free"),
        ]
        for max_pixels in OOM_RETRY_MAX_PIXELS:
            retry_image, resized = self._fit_image_to_pixel_budget(image, max_pixels)
            if resized:
                attempts.append(
                    (retry_image, False, f"oom_scaled_{max_pixels}_pixels")
                )

        failures = []
        for attempt_number, (attempt_image, use_cache, mode) in enumerate(
            attempts, start=1
        ):
            self._release_cuda(torch)
            memory_before = self._cuda_memory(torch)
            try:
                answer, elapsed, diagnostics = self._generate_once(
                    attempt_image,
                    prompt,
                    max_new_tokens,
                    use_cache=use_cache,
                )
            except torch.OutOfMemoryError as error:
                failures.append({
                    "attempt": attempt_number,
                    "mode": mode,
                    "image_size": list(self._image_size(attempt_image) or ()),
                    "memory_before": memory_before,
                    "error": str(error).splitlines()[0],
                })
                self._release_cuda(torch)
                print(
                    "[Agent1][VRAM] CUDA OOM during "
                    f"{mode}; temporary tensors cleared."
                )
                continue
            diagnostics.update({
                "attempt": attempt_number,
                "inference_mode": mode,
                "full_detail": mode.startswith("full_detail"),
                "oom_recovery_used": bool(failures),
                "operating_limit_applied": bool(primary_resized),
                "resolution_reduced": bool(
                    primary_resized or mode.startswith("oom_scaled_")
                ),
                "primary_max_pixels": PRIMARY_MAX_PIXELS,
                "original_image_size": list(original_size or ()),
                "memory_before": memory_before,
                "memory_after_cleanup": self._cuda_memory(torch),
                "failed_attempts": failures,
            })
            self._last_generation_diagnostics = diagnostics
            if mode.startswith("oom_scaled_"):
                print(
                    "[Agent1][VRAM] Recovered with adaptive image budget: "
                    f"{diagnostics['image_size']} (original "
                    f"{diagnostics['original_image_size']})."
                )
            elif primary_resized:
                print(
                    "[Agent1][VRAM] Applied measured 8 GB full-frame budget: "
                    f"{diagnostics['image_size']} (original "
                    f"{diagnostics['original_image_size']}); OCR crops remain "
                    "sourced from the original image."
                )
            return answer, elapsed

        self._last_generation_diagnostics = {
            "backend": self.backend,
            "quantization": self.quantization,
            "oom_recovery_used": True,
            "operating_limit_applied": bool(primary_resized),
            "resolution_reduced": bool(
                primary_resized or any(
                    item[2].startswith("oom_scaled_") for item in attempts
                )
            ),
            "primary_max_pixels": PRIMARY_MAX_PIXELS,
            "original_image_size": list(original_size or ()),
            "failed_attempts": failures,
            "memory_after_cleanup": self._cuda_memory(torch),
        }
        raise RuntimeError(
            "Qwen3-VL could not process this image within 8 GB VRAM after "
            "full-detail, cache-free, and adaptive-resolution attempts."
        )
