"""Small CPU natural-language-inference verifier for grounded text pairs."""

class NliVerifier:
    MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
    REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"

    def __init__(self):
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as error:
            raise RuntimeError(
                "The NLI diagnostic requires PyTorch and Transformers."
            ) from error
        self._torch = torch
        print("Loading targeted NLI verifier on CPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID,
            revision=self.REVISION,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_ID,
            revision=self.REVISION,
        ).to("cpu")
        self.model.eval()
        print("Targeted NLI verifier loaded.")

    def predict(self, premise, hypothesis):
        return self.predict_batch([(premise, hypothesis)])[0]

    def predict_batch(self, pairs, batch_size=16):
        """Score premise/hypothesis pairs in deterministic CPU batches."""
        pairs = list(pairs or [])
        outputs = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            premises = [str(pair[0]) for pair in batch]
            hypotheses = [str(pair[1]) for pair in batch]
            inputs = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with self._torch.inference_mode():
                rows = self._torch.softmax(
                    self.model(**inputs).logits, dim=-1
                ).tolist()
            for probabilities in rows:
                outputs.append({
                    str(self.model.config.id2label[index]).lower(): float(probability)
                    for index, probability in enumerate(probabilities)
                })
        return outputs

    @staticmethod
    def resolve_binary(probabilities):
        contradiction = float(probabilities.get("contradiction", 0.0))
        entailment = float(probabilities.get("entailment", 0.0))
        neutral = float(probabilities.get("neutral", 0.0))
        selected = max(contradiction, entailment)
        if neutral >= selected or contradiction + entailment <= 0.0:
            return None, 0.5
        label = "CONTRADICTS" if contradiction > entailment else "ENTAILS"
        binary_confidence = selected / (contradiction + entailment)
        return label, binary_confidence
