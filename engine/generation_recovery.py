"""Detect unproductive repetition without treating a stopped stream as evidence."""


def repeated_tail(tokens, repeats=6):
    tokens = list(tokens)
    for width in range(1, 9):
        if len(tokens) >= max(24, width * repeats):
            tail = tokens[-width * repeats:]
            if tail == tail[-width:] * repeats:
                return True
    return False


def repetition_stopper(prompt_length):
    from transformers import StoppingCriteria
    class RepetitionStopper(StoppingCriteria):
        triggered = False
        def __call__(self, input_ids, scores, **kwargs):
            count = int(input_ids.shape[-1]) - prompt_length
            if count >= 24 and count % 8 == 0:
                self.triggered = repeated_tail(input_ids[0, -min(count, 48):].tolist())
            return self.triggered
    return RepetitionStopper()


def repeated_text(text):
    return repeated_tail(str(text).split())
