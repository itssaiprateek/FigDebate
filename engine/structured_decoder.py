"""Single shared grammar backend; fail closed, never unconstrained fallback.

Uses llguidance's public tokenizer and matcher API (MIT). The small prefix
adapter supports the project's batch-one greedy generation, not beam search.
"""
from copy import copy, deepcopy
from importlib.metadata import version
import json
import numpy as np
from llguidance import LLTokenizer, LLMatcher

DECODER_ID = "llguidance-1.8.0-prefix-v2-bounded-whitespace-32"
MAX_STRUCTURAL_WHITESPACE = 32


def prefix_constraint(tokenizer, schema, compact=False):
    if version("llguidance") != "1.8.0":
        raise RuntimeError("Install the pinned llguidance==1.8.0 backend")
    data = getattr(tokenizer, "_figdebate_llguidance_data", None)
    if data is None:
        backend = copy(tokenizer.backend_tokenizer)
        backend.no_padding()
        backend.no_truncation()
        data = LLTokenizer(backend.to_str(), n_vocab=len(tokenizer),
                           eos_token=tokenizer.eos_token_id)
        tokenizer._figdebate_llguidance_data = data
    # SentencePiece completion tokens can carry leading whitespace. JSON allows
    # it; llguidance's bare JSON compiler does not allow it outside the object.
    # Bound structural formatting, never whitespace inside JSON string values.
    # Otherwise greedy decoding can spend its entire budget on indentation.
    constrained = deepcopy(schema)
    whitespace_limit = 1 if compact else MAX_STRUCTURAL_WHITESPACE
    constrained["x-guidance"] = {**constrained.get("x-guidance", {}),
                                  "whitespace_pattern": rf"[\x20\x0A\x0D\x09]{{1,{whitespace_limit}}}"}
    padding = '/[ \\t\\r\\n]{0,' + str(whitespace_limit) + '}/'
    grammar = 'start: ' + padding + ' body ' + padding + '\nbody: %json ' + json.dumps(constrained)
    matcher = LLMatcher(data, LLMatcher.grammar_from_lark(grammar))
    if matcher.is_error() or matcher.get_grammar_warnings():
        raise ValueError("Unsupported grammar: " + str(matcher.get_error() or matcher.get_grammar_warnings()))
    previous = None
    options = None

    def allowed(batch_id, token_ids):
        nonlocal previous, options
        if batch_id != 0:
            raise ValueError("Structured generation requires batch size one")
        current = token_ids.tolist()
        if previous is not None:
            if current[:len(previous)] != previous:
                raise ValueError("Structured generation does not support prefix rewrites or beams")
            if current == previous:
                return options
            if not matcher.consume_tokens(current[len(previous):]):
                raise ValueError("Invalid generated token: " + matcher.get_error())
        previous = current
        options = np.flatnonzero(np.frombuffer(matcher.compute_logit_bias(), dtype=np.uint8)).tolist()
        if matcher.is_error() or not options:
            raise ValueError("Grammar has no valid continuation: " + matcher.get_error())
        return options

    return allowed
