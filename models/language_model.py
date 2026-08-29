class MistralModel:
    def __init__(self):
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )
        except ImportError as error:
            raise RuntimeError(
                "Mistral requires the validated PyTorch/Transformers runtime. "
                "Run check_environment.py before inference."
            ) from error
        print("Loading Mistral...")

        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        model_revision = "63a8b081895390a26e140280378bc85ec8bce07a"
        from models.hub_source import cached_snapshot_or_hub
        model_source, source_kwargs = cached_snapshot_or_hub(
            model_id, model_revision
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source, **source_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            **source_kwargs,
            quantization_config=bnb_config,
            device_map="auto",
        )

        self.model.eval()

        print("Mistral loaded")
