"""Pinned local Qwen runtime used only by the optional judge stage."""

import os
import time


JUDGE_MODEL_ID = "Qwen/Qwen3.5-4B"
JUDGE_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
JUDGE_MODEL_DIRECTORY = os.path.join("models", "judge", "Qwen3.5-4B")


def default_judge_model_path():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, JUDGE_MODEL_DIRECTORY)


class QwenJudgeModel:
    """Load the independent multimodal judge locally in deterministic 4-bit mode."""

    def __init__(self, model_path=None):
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
            max_pixels=1_048_576,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print("Independent Qwen judge loaded")

    def generate(self, image, prompt, max_new_tokens=384):
        import torch

        if hasattr(image, "convert"):
            image = image.convert("RGB")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an independent multimodal appellate judge. "
                    "Follow the supplied evidence and JSON contract exactly."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        inputs = inputs.to(self.device)
        started = time.time()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        elapsed = time.time() - started
        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = generated[:, prompt_length:]
        response = self.processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return response, elapsed
