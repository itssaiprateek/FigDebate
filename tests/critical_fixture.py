from engine.output_contracts import CORE


def core(caption="The meeting was calm."):
    value = {key: None if definition.get("type") == ["string", "null"] else
             [] if definition.get("type") == "array" else "meeting"
             for key, definition in CORE["properties"].items()}
    value.update(caption_proposition=caption, relation_family="other", reasoning_requirement="visual",
                 expected_visual_state="The meeting is calm.", opposite_visual_state="The meeting is chaotic.")
    return value
