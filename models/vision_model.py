class LlavaModel:
    def __init__(self):
        try:
            import torch
            from transformers import (
                LlavaForConditionalGeneration,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as error:
            raise RuntimeError(
                "LLaVA requires the validated PyTorch/Transformers runtime. "
                "Run check_environment.py before inference."
            ) from error
        print("Loading LLaVA...")

        model_id = "llava-hf/llava-1.5-7b-hf"
        model_revision = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            revision=model_revision,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # Match the original LLaVA image-padding behavior and populate the
        # processor metadata required by recent Transformers releases.
        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=model_revision
        )
        self.processor.image_processor.do_pad = True
        self.processor.patch_size = getattr(
            self.model.config.vision_config, "patch_size", 14
        )
        feature_strategy = getattr(
            self.model.config, "vision_feature_select_strategy", "default"
        )
        self.processor.vision_feature_select_strategy = feature_strategy
        # LLaVA 1.5 uses a CLIP vision backbone with one CLS token. The
        # processor subtracts that token for the default feature strategy, so
        # this must remain 1 to produce the model's 576 image placeholders.
        self.processor.num_additional_image_tokens = 1

        self.model.eval()

        print("LLaVA loaded")
