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

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=model_revision
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_revision,
            quantization_config=bnb_config,
            device_map="auto",
        )

        self.model.eval()

        print("Mistral loaded")
