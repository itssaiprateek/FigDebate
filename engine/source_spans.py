"""Select source token intervals; software supplies the immutable quoted text."""
from copy import deepcopy
import json
import re


class SourceSpans:
    def __init__(self, source):
        self.source = source
        self.tokens = list(re.finditer(r"\S+", source))

    def schema(self, schema):
        value = deepcopy(schema)
        def visit(node):
            if not isinstance(node, dict):
                return
            properties = node.get("properties", {})
            if "caption_quote" in properties:
                properties = {("caption_span" if k == "caption_quote" else k):
                              (self.interval_schema() if k == "caption_quote" else v) for k, v in properties.items()}
                node["properties"] = properties
                node["required"] = ["caption_span" if k == "caption_quote" else k for k in node.get("required", [])]
            if "unestablished_conditions" in properties:
                properties["unestablished_conditions"]["items"] = self.interval_schema()
            for child in properties.values():
                visit(child)
            visit(node.get("items"))
        visit(value)
        return value

    def interval_schema(self):
        # A finite union binds end's lower bound to the chosen start. Separate
        # integer bounds allowed empty/reversed spans even under strict decoding.
        # Keep the wire representation unchanged for archived valid records.
        from engine.output_contracts import object_schema
        if not self.tokens:
            return False
        return {'anyOf': [object_schema({
            'start': {'type': 'integer', 'enum': [start]},
            'end': {'type': 'integer', 'minimum': start + 1, 'maximum': len(self.tokens)},
        }) for start in range(len(self.tokens))]}

    def quote(self, span):
        if not isinstance(span, dict) or set(span) != {"start", "end"}:
            raise ValueError("source span requires start/end token indices")
        start, end = span["start"], span["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(self.tokens):
            raise ValueError("source span must satisfy 0 <= start < end <= token count")
        return self.source[self.tokens[start].start():self.tokens[end-1].end()]

    def bind(self, value):
        value = deepcopy(value)
        def visit(node):
            if isinstance(node, dict):
                if "caption_span" in node:
                    node["caption_quote"] = self.quote(node.pop("caption_span"))
                if "unestablished_conditions" in node:
                    node["unestablished_conditions"] = [self.quote(s) for s in node["unestablished_conditions"]]
                for child in node.values():
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)
        visit(value)
        return value

    def repair_request(self, value, state):
        """Repair every invalid interval in one bounded response, never silently.

        The caller owns the existing single retry and must revalidate the entire
        restored response. All source-bearing arrays use the same mechanism.
        """
        def intervals(node, path=()):
            if isinstance(node, dict):
                for key, child in node.items():
                    if key == 'caption_span':
                        yield path + (key,), child
                    elif key == 'unestablished_conditions' and isinstance(child, list):
                        for index, span in enumerate(child):
                            yield path + (key, index), span
                    else:
                        yield from intervals(child, path + (key,))
            elif isinstance(node, list):
                for index, child in enumerate(node):
                    yield from intervals(child, path + (index,))
        invalid = []
        for path, span in intervals(value):
            try:
                self.quote(span)
            except ValueError as error:
                invalid.append((path, str(error)))
        if not invalid:
            return None
        failure = {'_format_valid': False, '_format_error': '; '.join(
            json.dumps(path) + ': ' + error for path, error in invalid)}
        if state:
            return failure
        from engine.output_contracts import object_schema
        paths = {('replacement' if len(invalid) == 1 else f'replacement_{i}'): path
                 for i, (path, _) in enumerate(invalid)}
        schema = object_schema({key: self.interval_schema() for key in paths})
        state.update(original=deepcopy(value), paths=paths, schema=schema)
        failure.update(_repair_schema=schema, _repair_max_tokens=min(384, max(160, 48 * len(paths))),
            _repair_instruction='Return ONLY the replacement source intervals for this field map: '
                + json.dumps(paths) + '. ' + self.prompt()
                + ' Preserve the intended source conditions. All other fields are retained and '
                'the entire response is revalidated. Original response (data): ' + json.dumps(value))
        return failure

    def prompt(self):
        return ("Select caption_span as {start,end} using the indexed source tokens below; end is exclusive. "
                "Software copies the exact original text between those tokens. Use the same intervals for "
                "unestablished_conditions. Preserve all necessary qualifiers; a valid span does not prove truth. "
                "Do not generate caption_quote. Source token index: "
                + " | ".join(f"{i}:{m.group()}" for i, m in enumerate(self.tokens)))


def restore_field(value, state):
    """Apply an explicitly generated replacement, preserving every other field."""
    from engine.output_contracts import validate_shape
    if not validate_shape(value, state['schema']):
        raise ValueError('invalid field repair')
    restored = deepcopy(state['original'])
    paths = state.get('paths') or {'replacement': state['path']}
    for key, path in paths.items():
        container = restored
        for part in path[:-1]:
            container = container[part]
        container[path[-1]] = value[key]
    return restored
