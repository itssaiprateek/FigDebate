"""Source-anchored claim obligations, adapted from DSG (arXiv:2310.18235).

The graph is a fallible language interpretation, never image evidence. Flat
requirements are a compatibility projection, not independently generated facts.
"""
from copy import deepcopy
import hashlib
import json
from engine.output_contracts import TEXT, TEXT_LIST, NULLABLE_TEXT, object_schema, validate_shape
from engine.claim_validation import requirement_errors

NODE = object_schema({
    "id": TEXT, "source_quote": TEXT, "subject": TEXT, "predicate": TEXT,
    "object": NULLABLE_TEXT, "qualifiers": TEXT_LIST,
    "reading": {"type": "string", "enum": ["EXPRESSED", "IDIOMATIC", "UNRESOLVED"]},
    "interpreted_proposition": TEXT, "support_condition": TEXT,
    "conflict_condition": TEXT, "depends_on": TEXT_LIST,
})
SCHEMA = object_schema({
    "composition": {"type": "string", "enum": ["SINGLE", "ALL", "ANY", "UNRESOLVED"]},
    "nodes": {"type": "array", "minItems": 1, "maxItems": 8, "items": NODE},
})

# IDs and truth conditions are compiler output, not language-model inventions.
DRAFT_NODE = object_schema({
    key: value for key, value in NODE["properties"].items()
    if key not in {"id", "support_condition", "conflict_condition", "depends_on"}
} | {"depends_on_indices": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 8}}})
DRAFT_SCHEMA = object_schema({"composition": SCHEMA["properties"]["composition"],
    "nodes": {"type": "array", "minItems": 1, "maxItems": 8, "items": DRAFT_NODE}})
CONDITION_ORIGIN = "proposition_truth_conditions_v1"


def truth_conditions(proposition):
    """P and not-P are hypothetical tests, never observations or image proof."""
    quoted = json.dumps(proposition.strip(), ensure_ascii=False)
    return f"The following proposition is true: {quoted}", f"The following proposition is false: {quoted}"


def conditions_are_compiled(graph):
    return bool(isinstance(graph, dict) and graph.get("condition_origin") == CONDITION_ORIGIN
                and isinstance(graph.get("nodes"), list) and graph["nodes"]
                and all(isinstance(n, dict) and isinstance(n.get("interpreted_proposition"), str)
                        and (n.get("support_condition"), n.get("conflict_condition")) ==
                        truth_conditions(n.get("interpreted_proposition", "")) for n in graph["nodes"]))


def source_identity_graph(caption):
    """Lossless fallback, explicitly NOT a certified atomic decomposition.

    A small model's approval cannot certify equivalence of a paraphrase. Keep
    the complete source as one obligation when decomposition is unqualified.
    Figurative interpretation and image truth remain the tribunal's job.
    """
    support, conflict = truth_conditions(caption)
    node = dict(id="C1", source_quote=caption, subject=caption, predicate="asserts",
                object=None, qualifiers=[], reading="EXPRESSED",
                interpreted_proposition=caption, support_condition=support,
                conflict_condition=conflict, depends_on=[])
    graph = dict(composition="SINGLE", nodes=[node], source_caption=caption,
                 schema_version="2.1", condition_origin=CONDITION_ORIGIN,
                 representation="UNDECOMPOSED_SOURCE", semantic_verified=False,
                 decomposition_status="UNQUALIFIED_USE_FULL_CAPTION")
    graph["errors"] = validate_graph(caption, {k: graph[k] for k in ("composition", "nodes")})
    graph["fingerprint"] = fingerprint(graph)
    graph["source_spans"] = {"C1": {"start": 0, "end": len(caption), "text": caption}}
    return graph


def protect_source_obligation(caption, graph):
    """Do not promote model paraphrases using a same-model PASS as proof.

    The current decomposer failed live semantic qualification. Its proposal
    remains archived; only a source-identical single obligation is admitted.
    This is not a claim that multiple assertions have become one atomic fact.
    """
    nodes = graph.get("nodes", [])
    identical = (not graph.get("errors") and graph.get("composition") == "SINGLE"
                 and len(nodes) == 1 and nodes[0].get("source_quote") == caption
                 and nodes[0].get("interpreted_proposition") == caption
                 and nodes[0].get("reading") == "EXPRESSED" and not nodes[0].get("depends_on"))
    # Rebuild even a source-identical proposal: matching prose alone cannot
    # certify the model's separate subject/predicate metadata.
    admitted = source_identity_graph(caption)
    if identical:
        admitted["decomposition_status"] = "SOURCE_IDENTICAL_SINGLE_OBLIGATION"
    return admitted


def compile_draft(value):
    if not validate_shape(value, DRAFT_SCHEMA):
        return {}
    nodes = []
    for index, draft in enumerate(value["nodes"], 1):
        # Reject malformed dependency indices, rather than rounding/remapping.
        if any(type(parent) not in (int, float) or int(parent) != parent or parent >= index
               for parent in draft["depends_on_indices"]):
            return {}
        node = {k: deepcopy(v) for k, v in draft.items() if k != "depends_on_indices"}
        node.update(id=f"C{index}", depends_on=[f"C{int(p)}" for p in draft["depends_on_indices"]])
        node["support_condition"], node["conflict_condition"] = truth_conditions(node["interpreted_proposition"])
        nodes.append(node)
    return {"composition": value["composition"], "nodes": nodes}


def fingerprint(graph):
    content = {key: graph.get(key) for key in ("source_caption", "composition", "nodes")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_graph(caption, value):
    if not validate_shape(value, SCHEMA):
        return ["INVALID_GRAPH_SHAPE"]
    errors, ids = [], [node["id"] for node in value["nodes"]]
    if ids != [f"C{index + 1}" for index in range(len(ids))]:
        errors.append("INVALID_OR_DUPLICATE_NODE_IDS")
    if value["composition"] == "UNRESOLVED":
        errors.append("UNRESOLVED_COMPOSITION")
    if value["composition"] == "SINGLE" and len(ids) != 1:
        errors.append("SINGLE_REQUIRES_ONE_NODE")
    seen = set()
    for node in value["nodes"]:
        prefix = node["id"] + ":"
        if not node["source_quote"].strip() or node["source_quote"] not in caption:
            errors.append(prefix + "INVALID_SOURCE_QUOTE")
        if not node["subject"].strip() or not node["predicate"].strip():
            errors.append(prefix + "MISSING_BINDING")
        if any(not q.strip() or q not in caption for q in node["qualifiers"]):
            errors.append(prefix + "UNANCHORED_QUALIFIER")
        # Topological order makes cycles, self-links and missing IDs explicit.
        if any(parent not in seen for parent in node["depends_on"]):
            errors.append(prefix + "INVALID_DEPENDENCY")
        if node["reading"] == "UNRESOLVED":
            errors.append(prefix + "UNRESOLVED_READING")
        if not node["interpreted_proposition"].strip():
            errors.append(prefix + "MISSING_PROPOSITION")
        errors.extend(prefix + error for error in requirement_errors(node["support_condition"], node["conflict_condition"]))
        seen.add(node["id"])
    return errors


def projected_requirements(graph):
    """Truth-functional projection; natural-language graph correctness is separate."""
    if graph.get("errors") or not graph.get("nodes"):
        return "", ""
    nodes = graph["nodes"]
    if graph["composition"] == "SINGLE":
        return nodes[0]["support_condition"], nodes[0]["conflict_condition"]
    support_op, conflict_op = (("ALL", "AT LEAST ONE") if graph["composition"] == "ALL"
                               else ("AT LEAST ONE", "ALL"))
    def render(operator, field):
        return operator + " of these conditions holds: " + "; ".join(
            f"[{node['id']}] {node[field]}" for node in nodes)
    return render(support_op, "support_condition"), render(conflict_op, "conflict_condition")


def resolve_nodes(graph, records, visible_evidence_ids):
    """Check bindings and aggregate model-proposed node relations, not their truth."""
    errors, relations = [], {}
    nodes = {node["id"]: node for node in graph.get("nodes", [])}
    for record in records:
        if (not isinstance(record, dict) or set(record) != {"claim_node_id", "relation", "evidence_ids"}
                or not isinstance(record["claim_node_id"], str)
                or record["relation"] not in {"SUPPORT", "CONFLICT", "UNRESOLVED"}
                or not isinstance(record["evidence_ids"], list)
                or not all(isinstance(item, str) for item in record["evidence_ids"])):
            errors.append("INVALID_NODE_RESOLUTION")
            continue
        node_id = record["claim_node_id"]
        if node_id not in nodes or node_id in relations:
            errors.append("UNKNOWN_OR_DUPLICATE_CLAIM_NODE")
            continue
        citations = record["evidence_ids"]
        if record["relation"] != "UNRESOLVED" and (not citations or not set(citations) <= set(visible_evidence_ids)):
            errors.append("UNBOUND_NODE_EVIDENCE")
        relations[node_id] = record["relation"]
    for node_id, relation in relations.items():
        if relation != "UNRESOLVED" and any(relations.get(parent) != "SUPPORT" for parent in nodes[node_id]["depends_on"]):
            errors.append("UNRESOLVED_NODE_DEPENDENCY")
    values = [relations.get(node_id, "UNRESOLVED") for node_id in nodes]
    aggregate = "UNRESOLVED"
    if values and not errors and not graph.get("errors"):
        if graph["composition"] in {"SINGLE", "ALL"}:
            if "CONFLICT" in values:
                aggregate = "CONFLICT"
            elif all(value == "SUPPORT" for value in values):
                aggregate = "SUPPORT"
        elif graph["composition"] == "ANY":
            if "SUPPORT" in values:
                aggregate = "SUPPORT"
            elif all(value == "CONFLICT" for value in values):
                aggregate = "CONFLICT"
    return {"relation": aggregate, "errors": errors, "node_relations": relations,
            "status": "BLOCKED_UPSTREAM" if graph.get("errors") else "INVALID_RESOLUTION" if errors
                      else "RESOLVED" if aggregate != "UNRESOLVED" else "UNRESOLVED",
            "upstream_errors": list(graph.get("errors", [])),
            "missing_node_ids": [key for key in nodes if key not in relations],
            "graph_fingerprint": graph.get("fingerprint"), "image_truth_verified": False}


def bind_readings(graph, answers):
    """Bind fallible interpretations without changing truth-condition nodes."""
    graph = deepcopy(graph)
    digest = hashlib.sha256(graph.get("source_caption", "").encode()).hexdigest()
    readings = {r["question_id"]: deepcopy(r) for r in graph.get("readings", [])}
    for answer in answers:
        if not answer.get("question_id") or answer.get("source_caption_sha256") != digest:
            raise ValueError("Reading is not bound to this source caption")
        readings[answer["question_id"]] = deepcopy(answer)
    graph["readings"] = list(readings.values())
    graph["reading_fingerprint"] = hashlib.sha256(json.dumps(
        graph["readings"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return graph


def attach_graph(agent, caption, fields, verification=None, research_decomposition=False):
    fields.setdefault("generated_caption_proposition", fields.get("caption_proposition"))
    fields["caption_proposition"] = caption
    if not research_decomposition:
        graph = source_identity_graph(caption)
        fields["claim_graph"] = graph
        fields["expected_visual_state"], fields["opposite_visual_state"] = projected_requirements(graph)
        fields["_graph_generation"] = {
            "status": "NOT_RUN_UNQUALIFIED_DECOMPOSER", "seconds": 0.0,
            "admitted_graph_fingerprint": graph["fingerprint"],
            "admission_method": "source_identity_with_separate_fallible_readings"}
        return 0.0
    instruction = (
        "Decompose the original caption into atomic bound propositions, preserving every role, "
        "negation, number, modifier and scope. Each source_quote "
        "and qualifier must be an exact caption substring. Separate distinct entity/property assertions. "
        "SINGLE means one proposition, ALL a conjunction, ANY a disjunction. Use UNRESOLVED for "
        "unsupported or ambiguous logical structure; never simplify a quantifier to force a parse. "
        "Each interpreted_proposition must be a complete bound assertion, including every qualifier. "
        "Do not split a subject from its property or create an empty event from a copula. "
        "An idiom may use an IDIOMATIC reading with an explicit interpreted proposition; "
        "do not require the literal physical event. Do not substitute sarcastic intended polarity for "
        "the expressed proposition. depends_on_indices contains 1-based indices of EARLIER nodes only "
        "whose truth is necessary to evaluate this node. Use [] for independent assertions; sentence "
        "order is not a logical dependency. Do not invent causes, scenarios, or image observations. "
        "IDs and hypothetical true/false conditions are assigned by software. "
        "The graph is a hypothesis, not a verdict. Keep each field concise. "
        "Use JSON null for absent objects and [] for absent qualifiers/dependencies; never the string None.\n"
        "Existing extraction is fallible context, not authority; resolve any disagreement from the original caption: "
        + json.dumps({k: fields.get(k) for k in ("claim_subject", "claim_predicate", "claim_object",
            "claim_modifiers", "negation", "quantities", "surface_meaning", "intended_meaning",
            "literal_meaning", "underlying_message", "alternative_interpretation")}, ensure_ascii=False)
    )
    if verification:
        instruction += "\nPreviously identified defects (fallible, not desired answers): " + json.dumps({
            "input_errors": verification.get("input_errors", []),
            "obligations": verification.get("obligations", []),
        })
    draft, raw, seconds, diagnostics = agent._generate_section(instruction, caption, 1024, schema=DRAFT_SCHEMA)
    value = compile_draft(draft)
    errors = validate_graph(caption, value)
    if not validate_shape(value, SCHEMA):
        value = {}
    graph = dict(deepcopy(value), source_caption=caption, schema_version="2.0", errors=errors,
                 semantic_verified=False, condition_origin=CONDITION_ORIGIN)
    graph["fingerprint"] = fingerprint(graph)
    graph["source_spans"] = {
        node["id"]: {"start": caption.index(node["source_quote"]),
                     "end": caption.index(node["source_quote"]) + len(node["source_quote"]),
                     "text": node["source_quote"]}
        for node in value.get("nodes", []) if node.get("source_quote") and node["source_quote"] in caption
    }
    candidate_graph = deepcopy(graph)
    graph = protect_source_obligation(caption, graph)
    fields["claim_graph"] = graph
    fields["expected_visual_state"], fields["opposite_visual_state"] = projected_requirements(graph)
    fields["_graph_generation"] = dict(raw_response=raw, diagnostics=diagnostics, seconds=seconds,
        candidate_graph=candidate_graph, admitted_graph_fingerprint=graph["fingerprint"],
        admission_method="source_identity_only_until_decomposer_qualified")
    return seconds


def repair_graph(agent, caption, fields, verification=None):
    """Patch named defective nodes; retain unaffected nodes and every attempt."""
    graph = deepcopy(fields.get("claim_graph") or {})
    if not graph.get("nodes"):
        history = fields.setdefault("_graph_repair_history", [])
        history.append({"before": deepcopy(graph), "reason": "INVALID_GRAPH_SHAPE"})
        seconds = attach_graph(agent, caption, fields, verification)
        history[-1]["after"] = deepcopy(fields["claim_graph"])
        return seconds
    checks = verification or {}
    errors = graph.get("errors", [])
    target_ids = {e.split(":", 1)[0] for e in errors if ":" in e}
    if not target_ids or checks.get("audit_status") in {"FAIL", "UNKNOWN"}:
        # A coverage/composition defect cannot be safely localized to a node.
        # Retain the old graph, but allow one new source-grounded decomposition.
        history = fields.setdefault("_graph_repair_history", [])
        history.append({"before": deepcopy(graph), "reason": "SOURCE_COVERAGE_OR_COMPOSITION", "audit": deepcopy(checks)})
        seconds = attach_graph(agent, caption, fields, checks)
        history[-1]["after"] = deepcopy(fields["claim_graph"])
        return seconds
    seconds = 0.0
    for index, node in enumerate(graph["nodes"]):
        if node["id"] not in target_ids:
            continue
        prompt = ("Repair only this caption proposition. Preserve source meaning, all qualifiers and roles. "
                  "Return a complete assertion, not a phrase. Do not invent observations or causes. "
                  "depends_on_indices may name only earlier necessary prerequisites, not sentence order. "
                  "Original node: " + json.dumps(node) + "\nErrors: " + json.dumps(errors))
        draft, raw, elapsed, diag = agent._generate_section(prompt, caption, 384, schema=DRAFT_NODE)
        seconds += elapsed
        before = deepcopy(node)
        # Compile in its original slot to preserve externally bound IDs.
        if validate_shape(draft, DRAFT_NODE) and all(int(p) == p and p <= index for p in draft["depends_on_indices"]):
            updated = {k: v for k, v in draft.items() if k != "depends_on_indices"}
            updated.update(id=node["id"], depends_on=[f"C{int(p)}" for p in draft["depends_on_indices"]])
            updated["support_condition"], updated["conflict_condition"] = truth_conditions(updated["interpreted_proposition"])
            graph["nodes"][index] = updated
        fields.setdefault("_graph_repair_history", []).append({"node_id": node["id"], "before": before,
            "after": deepcopy(graph["nodes"][index]), "raw_response": raw, "diagnostics": diag})
    graph["condition_origin"] = CONDITION_ORIGIN
    graph["errors"] = validate_graph(caption, {k: graph.get(k) for k in ("composition", "nodes")})
    graph["fingerprint"] = fingerprint(graph)
    graph["source_spans"] = {n["id"]: {"start": caption.index(n["source_quote"]),
        "end": caption.index(n["source_quote"]) + len(n["source_quote"]), "text": n["source_quote"]}
        for n in graph["nodes"] if n["source_quote"] and n["source_quote"] in caption}
    fields.setdefault("_graph_repair_history", []).append({"candidate_graph": deepcopy(graph),
        "admission_method": "source_identity_only_until_decomposer_qualified"})
    graph = protect_source_obligation(caption, graph)
    fields["claim_graph"] = graph
    fields["expected_visual_state"], fields["opposite_visual_state"] = projected_requirements(graph)
    return seconds
