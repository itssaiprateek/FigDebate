"""Official-split sampling, gold reference isolation, and frozen feedback tests."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from dataset.selection import select_run, usage_policy, sample_count
from engine.sampling import build_selection_manifest
from engine.batch_runner import StagewiseRunner
from evaluation.reference_bank import build_references
from evaluation.input_coverage import reconcile
from evaluation.evaluate_predictions import evaluate_predictions
from evaluation.build_feedback_memory import build_memory
from evaluation.configuration_pairing import verify_configuration, INVARIANT_KEYS


def rows():
    return [dict(id=f"vflute_train_{i}", caption=f"Claim {i}", label="ENTAILS" if i % 2 else "CONTRADICTS",
                 phenomenon="humor" if i < 10 else "sarcasm", source_dataset="memecap" if i < 10 else "muse",
                 image_bytes=f"image{i}".encode(), explanation=f"Reference {i}", source_lineage_verified=True)
            for i in range(20)]


def saved_run(directory, delivered=1):
    selected, manifest = select_run(rows(), "vflute_train", count=2, selection_seed=4)
    root = Path(directory)
    (root / "sample_manifest.json").write_text(json.dumps(manifest))
    (root / "evaluation_references.json").write_text(json.dumps(build_references(selected, manifest)))
    (root / "run_config.json").write_text(json.dumps({"dataset": "vflute_train", "requested_samples": 2,
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "data_usage": usage_policy("vflute_train", "diagnostic", "disabled")}))
    frame = pd.DataFrame([dict(entry, ground_truth=entry["label"], prediction=entry["label"],
        reference_explanation=row["explanation"], final_reason=row["explanation"], final_decision_valid=True)
        for entry, row in zip(manifest["records"][:delivered], selected[:delivered])])
    path = root / "predictions.csv"
    frame.to_csv(path, index=False)
    return frame, path, manifest


class OfficialSplitWorkflowTests(unittest.TestCase):
    def test_seeded_selection_is_repeatable_and_nested(self):
        a, _ = select_run(rows(), "vflute_train", count=4, selection_seed=7)
        b, _ = select_run(rows(), "vflute_train", count=8, selection_seed=7, inference_seed=999)
        c, _ = select_run(rows(), "vflute_train", count=4, selection_seed=9)
        self.assertEqual(a, b[:4])
        self.assertNotEqual(a, c)

    def test_filters_only_narrow_the_chosen_split(self):
        selected, manifest = select_run(rows(), "vflute_train", count=None, phenomena=["sarcasm"], sources=["muse"])
        self.assertEqual(len(selected), 10)
        self.assertEqual(manifest["dataset_split"], "vflute_train")
        self.assertTrue(all(r["phenomenon"] == "sarcasm" for r in selected))

    def test_explicit_ids_preserve_order_or_can_be_randomized(self):
        ids = [r["id"] for r in rows()[3:12]]
        selected, _ = select_run(rows(), "vflute_train", count=None, sample_ids=ids)
        shuffled, _ = select_run(rows(), "vflute_train", count=None, sample_ids=ids, strategy="random", selection_seed=2)
        self.assertEqual([r["id"] for r in selected], ids)
        self.assertEqual({r["id"] for r in shuffled}, set(ids))
        self.assertNotEqual(shuffled, selected)

    def test_saved_ids_file_defines_randomization_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.json"
            ids = [r["id"] for r in rows()[:8]]
            path.write_text(json.dumps(ids))
            selected, _ = select_run(rows(), "vflute_train", count=3, ids_file=path, strategy="random")
            self.assertTrue({r["id"] for r in selected}.issubset(ids))

    def test_unknown_duplicate_or_cross_split_ids_fail(self):
        for ids in (["vflute_test_1"], ["vflute_train_1", "vflute_train_1"], []):
            with self.assertRaises(ValueError):
                select_run(rows(), "vflute_train", sample_ids=ids)

    def test_empty_or_unknown_filters_fail(self):
        for kwargs in (dict(sources=["unknown"]), dict(phenomena=["humor"], sources=["muse"])):
            with self.assertRaises(ValueError):
                select_run(rows(), "vflute_train", **kwargs)

    def test_oversized_zero_and_negative_requests_do_not_silently_shrink(self):
        for count in (0, -1, 21):
            with self.assertRaises(ValueError):
                select_run(rows(), "vflute_train", count=count)
        self.assertIsNone(sample_count("all"))

    def test_replay_inherits_seed_and_strategy_without_extra_flags(self):
        original, manifest = select_run(rows(), "vflute_train", count=5, selection_seed=999)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            selected, _ = select_run(rows(), "vflute_train", count=5, manifest_path=path)
            self.assertEqual(original, selected)
            with self.assertRaises(ValueError):
                select_run(rows(), "vflute_val", manifest_path=path)
            with self.assertRaises(ValueError):
                select_run(rows(), "vflute_train", manifest_path=path, selection_seed=1)
            with self.assertRaises(ValueError):
                select_run(rows(), "vflute_train", manifest_path=path, phenomena=["humor"])

    def test_legacy_manifest_can_still_replay(self):
        manifest = build_selection_manifest(rows(), "stratified", 42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(len(select_run(rows(), "vflute_train", count=4, manifest_path=path)[0]), 4)

    def test_reference_change_is_detected_during_selection(self):
        _, manifest = select_run(rows(), "vflute_train", count=1)
        changed = deepcopy(rows())
        for r in changed:
            r["explanation"] = "Altered reference"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "reference_explanation"):
                select_run(changed, "vflute_train", manifest_path=path)

    def test_reference_sidecar_covers_missing_outputs_as_zero_explanation_score(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path, _ = saved_run(directory)
            result = evaluate_predictions(path, Path(directory) / "metrics")
            self.assertEqual(result["accuracy"], .5)
            self.assertTrue(result["input_coverage"]["references_bound_to_manifest"])
            self.assertEqual(result["explanations"]["reference_available_count"], 2)
            self.assertEqual(result["explanations"]["mean_token_f1"], .5)
            self.assertFalse(result["explanations"]["automatic_semantic_verification"])

    def test_wrong_csv_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            frame, path, _ = saved_run(directory)
            frame.loc[0, "reference_explanation"] = "A different answer key"
            with self.assertRaisesRegex(ValueError, "different reference"):
                reconcile(frame, path)

    def test_missing_or_modified_reference_bank_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            frame, path, _ = saved_run(directory)
            bank_path = Path(directory) / "evaluation_references.json"
            data = json.loads(bank_path.read_text())
            data["records"][0]["reference_explanation"] = "Changed"
            bank_path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                reconcile(frame, path)
            bank_path.unlink()
            with self.assertRaises(ValueError):
                reconcile(frame, path)

    def test_test_diagnostics_allowed_but_test_feedback_forbidden(self):
        self.assertEqual(usage_policy("vflute_test", "diagnostic", "disabled")["purpose"], "diagnostic")
        for mode in ("collect", "calibrate"):
            with self.assertRaises(ValueError):
                usage_policy("vflute_test", "diagnostic", mode)
            with self.assertRaises(ValueError):
                usage_policy("vflute_val", "evaluation", mode)

    def test_verified_memory_never_reads_gold_or_updates_reliability(self):
        class NoGold(dict):
            def get(self, key, default=None):
                if key in {"label", "explanation"}:
                    raise AssertionError("Gold reached frozen feedback")
                return super().get(key, default)
        runner = object.__new__(StagewiseRunner)
        runner.feedback_mode = "verified"
        runner.feedback_events = []
        runner.debate = object()  # Any attempt to call its old updater fails.
        result = {0: {"decision": {"label": "ENTAILS"}, "pre_feedback_decision": {"label": "ENTAILS"}}}
        for _ in range(3):
            runner._apply_feedback([{"raw": NoGold(id="a"), "index": 0}], result,
                                   {"matched_rule_ids": ["memory"], "batch_index": 0})
        self.assertEqual(result[0]["feedback"]["reliability_updates"], [])

    def test_feedback_builder_requires_development_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "configuration"):
                build_memory(root / "records.jsonl", root / "memory.json")
            (root / "run_config.json").write_text(json.dumps({"dataset": "vflute_test", "data_usage": {"purpose": "diagnostic"}}))
            with self.assertRaisesRegex(ValueError, "never test"):
                build_memory(root / "records.jsonl", root / "memory.json")

    def test_code_revision_comparison_requires_explicit_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root / "before", root / "after"
            left.mkdir()
            right.mkdir()
            config = dict.fromkeys(INVARIANT_KEYS, "same")
            changed = dict(config, pipeline_source_sha256="different")
            (left / "run_config.json").write_text(json.dumps(config))
            (right / "run_config.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "invariant"):
                verify_configuration(left / "predictions.csv", right / "predictions.csv")
            result = verify_configuration(left / "predictions.csv", right / "predictions.csv",
                                          code_change_description="Repair caption role binding")
            self.assertTrue(result["verified"])
            self.assertIsNotNone(result["declared_code_change"])
            changed["model_language_revision"] = "another-model"
            (right / "run_config.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "model_language_revision"):
                verify_configuration(left / "predictions.csv", right / "predictions.csv",
                                     code_change_description="Repair caption role binding")

    def test_feedback_builder_accepts_complete_verified_development_run(self):
        with tempfile.TemporaryDirectory() as directory:
            frame, path, _ = saved_run(directory, delivered=2)
            records_path = Path(directory) / "records.jsonl"
            records_path.write_text("".join(json.dumps(row) + "\n" for row in frame.to_dict("records")))
            with patch("dataset.loaders.load_split", return_value=rows()):
                result = build_memory(records_path, Path(directory) / "memory.json")
            self.assertEqual(result["incorrect_samples"], 0)
            self.assertEqual(json.loads((Path(directory) / "memory.json").read_text()), [])

    def test_feedback_treatment_changes_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root / "before", root / "after"
            left.mkdir()
            right.mkdir()
            config = dict.fromkeys(INVARIANT_KEYS, "same")
            changed = dict(config, feedback_mode="verified", verified_feedback_sha256="new-memory")
            (left / "run_config.json").write_text(json.dumps(config))
            (right / "run_config.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                verify_configuration(left / "predictions.csv", right / "predictions.csv")
            result = verify_configuration(left / "predictions.csv", right / "predictions.csv",
                                          allowed_changes=("feedback_mode", "verified_feedback_sha256"))
            self.assertTrue(result["verified"])


if __name__ == "__main__":
    unittest.main()
