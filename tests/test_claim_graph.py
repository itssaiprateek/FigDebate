from copy import deepcopy
import unittest
from unittest.mock import patch
from engine.claim_graph import validate_graph, projected_requirements, fingerprint, resolve_nodes, attach_graph
from engine.claim_validation import core_errors
from engine.claim_witness import audit_claim_witness
from engine.claim_semantics import CHECKS
from tests.critical_fixture import core


def graph_fixture(composition="ALL"):
    caption = "The red line rises and the blue line falls."
    nodes = [dict(id="C1", source_quote="red line rises", subject="red line", predicate="rises", object=None,
                  qualifiers=[], reading="EXPRESSED", interpreted_proposition="The red line rises.",
                  support_condition="The red line rises.", conflict_condition="The red line falls.", depends_on=[]),
             dict(id="C2", source_quote="blue line falls", subject="blue line", predicate="falls", object=None,
                  qualifiers=[], reading="EXPRESSED", interpreted_proposition="The blue line falls.",
                  support_condition="The blue line falls.", conflict_condition="The blue line rises.", depends_on=[])]
    graph = dict(composition=composition, nodes=nodes, source_caption=caption, errors=[])
    graph["fingerprint"] = fingerprint(graph)
    return graph


class ClaimGraphTests(unittest.TestCase):
    def test_source_graph_and_empty_lists_preserved(self):
        graph = graph_fixture()
        self.assertEqual(validate_graph(graph["source_caption"], {k: graph[k] for k in ("composition", "nodes")}), [])
        self.assertEqual(graph["nodes"][0]["qualifiers"], [])

    def test_changed_interpretation_invalidates_fingerprint(self):
        graph = graph_fixture()
        changed = deepcopy(graph)
        changed["nodes"][0]["interpreted_proposition"] = "The red line is sad."
        self.assertNotEqual(fingerprint(changed), graph["fingerprint"])

    def test_missing_quote_and_cycles_rejected(self):
        graph = graph_fixture()
        graph["nodes"][0].update(source_quote="invented", depends_on=["C2"])
        errors = validate_graph(graph["source_caption"], {k: graph[k] for k in ("composition", "nodes")})
        self.assertIn("C1:INVALID_SOURCE_QUOTE", errors)
        self.assertIn("C1:INVALID_DEPENDENCY", errors)

    def test_all_support_needs_every_atom_but_conflict_needs_one(self):
        graph = graph_fixture()
        def record(node, relation):
            return dict(claim_node_id=node, relation=relation, evidence_ids=["E1"])
        self.assertEqual(resolve_nodes(graph, [record("C1", "SUPPORT")], ["E1"])["relation"], "UNRESOLVED")
        self.assertEqual(resolve_nodes(graph, [record("C1", "CONFLICT")], ["E1"])["relation"], "CONFLICT")
        self.assertEqual(resolve_nodes(graph, [record("C1", "SUPPORT"), record("C2", "SUPPORT")], ["E1"])["relation"], "SUPPORT")

    def test_any_conflict_needs_every_atom(self):
        graph = graph_fixture("ANY")
        records = [dict(claim_node_id="C1", relation="CONFLICT", evidence_ids=["E1"])]
        self.assertEqual(resolve_nodes(graph, records, ["E1"])["relation"], "UNRESOLVED")

    def test_unknown_or_uncited_node_never_proves_relation(self):
        records = [dict(claim_node_id="C9", relation="CONFLICT", evidence_ids=["E1"])]
        self.assertTrue(resolve_nodes(graph_fixture(), records, ["E1"])["errors"])
        records[0].update(claim_node_id="C1", evidence_ids=[])
        self.assertEqual(resolve_nodes(graph_fixture(), records, ["E1"])["relation"], "UNRESOLVED")

    def test_projection_keeps_entity_bindings(self):
        support, conflict = projected_requirements(graph_fixture())
        self.assertTrue(support.startswith("ALL"))
        self.assertIn("[C1] The red line rises", support)
        self.assertIn("[C2] The blue line falls", support)
        self.assertTrue(conflict.startswith("AT LEAST ONE"))

    def test_flat_requirement_cannot_diverge_from_graph(self):
        graph = graph_fixture()
        fields = core(graph["source_caption"])
        fields["claim_graph"] = graph
        self.assertIn("REQUIREMENTS_DIVERGE_FROM_GRAPH", core_errors(fields))

    def test_witness_does_not_generate_competing_requirement_pair(self):
        graph = graph_fixture()
        fields = core(graph["source_caption"])
        fields["claim_graph"] = graph
        fields["expected_visual_state"], fields["opposite_visual_state"] = projected_requirements(graph)
        audit = dict(**{key: True for key in CHECKS}, _format_valid=True)
        with patch("engine.claim_witness.audit_core", return_value=audit):
            result = audit_claim_witness(None, graph["source_caption"], {"claim_fields": fields})
        self.assertTrue(result["requirements_valid"])
        self.assertEqual(result["requirements_source"], "canonical_claim_graph")

    def test_generation_is_connected_to_flat_projection(self):
        graph = graph_fixture()
        class Agent:
            def _generate_section(self, *args, **kwargs):
                nodes = [{k:v for k,v in n.items() if k not in
                          {"id", "support_condition", "conflict_condition", "depends_on"}}
                         | {"depends_on_indices": []} for n in graph["nodes"]]
                return {"composition": graph["composition"], "nodes": nodes}, "fixture", 0, {"schema_valid": True}
        fields = core(graph["source_caption"])
        attach_graph(Agent(), graph["source_caption"], fields)
        self.assertFalse(core_errors(fields))
        self.assertEqual(fields["expected_visual_state"], projected_requirements(fields["claim_graph"])[0])
