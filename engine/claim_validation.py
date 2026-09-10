"""Shared deterministic content guards; these are not semantic proof."""
import re
from engine.output_contracts import CORE, validate_shape
from engine.relation_semantics import is_missing_evidence_only


def missing_requirement(value):
    if not isinstance(value, str):
        return True
    value = value.strip().casefold().rstrip(".!;")
    return value in {"", "none", "null", "n/a", "unknown", "unavailable", "not applicable"}


def requirement_errors(support, conflict):
    errors = []
    if missing_requirement(support):
        errors.append("MISSING_SUPPORT_REQUIREMENT")
    if missing_requirement(conflict):
        errors.append("MISSING_CONFLICT_REQUIREMENT")
    normalize = lambda value: re.sub(r"\W+", " ", str(value).casefold()).strip()
    if not errors and normalize(support) == normalize(conflict):
        errors.append("IDENTICAL_REQUIREMENTS")
    if not missing_requirement(conflict) and is_missing_evidence_only(conflict):
        errors.append("ABSENCE_IS_NOT_CONFLICT")
    return errors


def core_errors(fields):
    subject = {key: fields.get(key) for key in CORE["properties"]}
    errors = [] if validate_shape(subject, CORE) else ["INVALID_CORE_TYPES"]
    graph = fields.get("claim_graph")
    if graph is not None and not isinstance(graph, dict):
        return errors + ["INVALID_GRAPH_SHAPE"]
    if graph is not None:
        from engine.claim_graph import validate_graph, fingerprint, projected_requirements
        errors.extend(validate_graph(subject["caption_proposition"],
                                     {key: graph.get(key) for key in ("composition", "nodes")}))
        if graph.get("fingerprint") != fingerprint(graph) or graph.get("source_caption") != subject["caption_proposition"]:
            errors.append("STALE_CLAIM_GRAPH")
        if projected_requirements(graph) != (subject["expected_visual_state"], subject["opposite_visual_state"]):
            errors.append("REQUIREMENTS_DIVERGE_FROM_GRAPH")
        if graph.get("condition_origin"):
            from engine.claim_graph import conditions_are_compiled
            if not conditions_are_compiled(graph):
                errors.append("COMPILED_CONDITIONS_MODIFIED")
        if graph.get("representation") == "UNDECOMPOSED_SOURCE":
            nodes = graph.get("nodes", [])
            if (graph.get("composition") != "SINGLE" or len(nodes) != 1
                    or nodes[0].get("source_quote") != subject["caption_proposition"]
                    or nodes[0].get("interpreted_proposition") != subject["caption_proposition"]
                    or nodes[0].get("depends_on") or nodes[0].get("reading") != "EXPRESSED"):
                errors.append("SOURCE_IDENTITY_MODIFIED")
    return errors + requirement_errors(subject["expected_visual_state"], subject["opposite_visual_state"])
