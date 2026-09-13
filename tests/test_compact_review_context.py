"""Context pressure must preserve exact claims and discoverable evidence."""
from copy import deepcopy
import json
import unittest

from engine.case_dossier import compact_evidence_packet, fit_judge_packet
from engine.claim_graph import source_identity_graph


class CompactReviewContextTests(unittest.TestCase):
    def test_retrieved_records_fit_without_losing_sources_or_catalog(self):
        caption = "The left person is calmer than the right person, after the announcement."
        graph = source_identity_graph(caption)
        records = [{"id": f"VF{i}", "text": f"Exact observation {i}. " * 6,
                    "type": "visual_fact", "source": "agent1",
                    "diagnostics": {"raw_output": "unused diagnostic " * 200}}
                   for i in range(12)]
        packet = {"source_caption": caption, "claim_agent": {"claim_graph": graph},
                  "evidence_ledger": records, "remaining_evidence_index": [],
                  "evidence_rendering": {"active_count": len(records)},
                  "candidate_cases": [{"context_id": "CTX1", "argument": "fallible " * 300}],
                  "targeted_hearing": {"visual": {"question_answers": [
                      {"question": "Which person looks alarmed?", "answer": "The right person."}],
                      "answer": "duplicate " * 100}}}
        original = deepcopy(packet)
        compact = compact_evidence_packet(packet)
        compact["protocol"] = "evidence-review-4.0"
        fitted, audit = fit_judge_packet(compact, json.dumps, token_counter=len,
                                        max_tokens=3000, protected_ids=["VF11"])
        self.assertEqual(packet, original)
        self.assertLessEqual(audit["text_input_tokens"], 3000)
        self.assertEqual(fitted["source_caption"], caption)
        self.assertEqual(fitted["claim_agent"]["claim_graph"], graph)
        self.assertEqual({r["id"] for r in fitted["evidence_ledger"] + fitted["remaining_evidence_index"]},
                         {r["id"] for r in records})
        self.assertIn("VF11", {r["id"] for r in fitted["evidence_ledger"]})
        for record in fitted["evidence_ledger"]:
            self.assertEqual(record["text"], next(r["text"] for r in records if r["id"] == record["id"]))
        self.assertEqual(fitted["remaining_context_index"][0]["context_id"], "CTX1")
        self.assertEqual(fitted["targeted_hearing"]["visual"]["question_answers"],
                         original["targeted_hearing"]["visual"]["question_answers"])

    def test_unavoidable_context_overflow_is_explicit(self):
        packet = {"source_caption": "Exact source " * 1000,
                  "claim_agent": {"claim_graph": source_identity_graph("Exact source")},
                  "evidence_ledger": [{"id": "VF1", "text": "Visible fact."}]}
        with self.assertRaisesRegex(ValueError, "mandatory caption"):
            fit_judge_packet(compact_evidence_packet(packet), json.dumps,
                             token_counter=len, max_tokens=100)
