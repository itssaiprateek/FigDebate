"""Adversarial fixtures for dataset lineage and honest evaluation denominators."""
from io import BytesIO
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image
from dataset.prepare_vflute import prepare
from dataset.verify_upstream import compare_row
from dataset.loaders import attach_provenance
from engine.sampling import build_selection_manifest
from evaluation.input_coverage import reconcile
from evaluation.evaluate_predictions import evaluate_predictions, explanation_metrics, confidence_metrics
from evaluation import metrics_core


def fixture():
    return [dict(id=str(i), caption="Claim", label="ENTAILS", phenomenon="humor", image_bytes=b"fixture") for i in range(2)]


def write_case(directory, delivered=1):
    manifest = build_selection_manifest(fixture(), strategy="prefix")
    root = Path(directory)
    (root / "sample_manifest.json").write_text(json.dumps(manifest))
    (root / "run_config.json").write_text(json.dumps({"requested_samples": 2,
        "selection_manifest_sha256": manifest["manifest_sha256"]}))
    rows = [dict(row, ground_truth=row["label"], prediction="ENTAILS", final_decision_valid=True, runtime_seconds=1.0)
            for row in manifest["records"][:delivered]]
    frame = pd.DataFrame(rows)
    path = root / "predictions.csv"
    frame.to_csv(path, index=False)
    return frame, path


class DatasetCorrectionTests(unittest.TestCase):
    def test_missing_output_counts_as_error_and_run_remains_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path = write_case(directory)
            result = evaluate_predictions(path, str(Path(directory) / "report"))
            self.assertEqual(result["samples"], 2)
            self.assertEqual(result["accuracy"], .5)
            self.assertEqual(result["unanswered_count"], 1)
            self.assertEqual(result["run_completion"]["recorded_samples"], 1)
            self.assertFalse(result["run_completion"]["complete"])
            self.assertFalse(result["publication_qualified"])
            json.dumps(result, allow_nan=False)

    def test_modified_gold_or_identity_is_rejected(self):
        for field in ("ground_truth", "image_sha256", "caption_sha256", "phenomenon", "id"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                frame, path = write_case(directory)
                frame.loc[0, field] = "tampered"
                with self.assertRaises(ValueError):
                    reconcile(frame, path)

    def test_duplicate_and_blank_ids_rejected_even_without_manifest(self):
        for ids in (["a", "a"], ["", "b"], [None, "b"], ["  ", "b"]):
            with self.assertRaises(ValueError):
                reconcile(pd.DataFrame({"id": ids}), "unused.csv")

    def test_manifest_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            frame, path = write_case(directory)
            manifest_path = Path(directory) / "sample_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["records"][0]["label"] = "CONTRADICTS"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                reconcile(frame, path)

    def test_absent_reference_not_zero_or_nan_token(self):
        frame = pd.DataFrame([dict(id="a", reference_explanation=float("nan"), final_reason="nan"),
                              dict(id="b", reference_explanation="A cat", final_reason="A cat")])
        result, _ = explanation_metrics(frame)
        self.assertEqual(result["reference_available_count"], 1)
        self.assertEqual(result["reference_missing_count"], 1)
        self.assertEqual(result["mean_token_f1"], 1)
        result, _ = explanation_metrics(frame.iloc[:1])
        self.assertIsNone(result["mean_token_f1"])
        json.dumps(result, allow_nan=False)

    def test_missing_generated_explanation_is_failure_with_reference(self):
        result, _ = explanation_metrics(pd.DataFrame([dict(reference_explanation="A cat", final_reason=float("nan"))]))
        self.assertEqual(result["mean_token_f1"], 0)

    def test_evidence_score_not_reported_as_probability(self):
        result, _ = confidence_metrics(pd.DataFrame([dict(ground_truth="ENTAILS", prediction="ENTAILS", final_confidence=.35)]))
        self.assertIsNone(result["brier_score"])
        self.assertIsNone(result["ece_10_bin"])
        self.assertAlmostEqual(result["score_accuracy_gap_10_bin"], .65)

    def test_out_of_range_confidence_is_not_silently_clipped(self):
        for value in (-1, 2, float("inf")):
            with self.assertRaises(ValueError):
                confidence_metrics(pd.DataFrame([dict(ground_truth="ENTAILS", prediction="ENTAILS", final_confidence=value)]))

    def test_fallback_metrics_reject_truncated_pairs(self):
        for function in (metrics_core.accuracy_score, metrics_core.balanced_accuracy_score):
            with self.assertRaises(ValueError):
                function(["ENTAILS", "CONTRADICTS"], ["ENTAILS"])
        with self.assertRaises(ValueError):
            metrics_core.f1_score(["ENTAILS"], [], labels=["ENTAILS", "CONTRADICTS"])

    def test_no_in_place_force_rebuild(self):
        with self.assertRaises(ValueError):
            prepare(force=True)

    def test_versioned_rebuild_does_not_overwrite_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keep.txt"
            path.write_text("original")
            with self.assertRaises(FileExistsError):
                prepare(output_dir=directory)
            self.assertEqual(path.read_text(), "original")

    def test_source_lineage_requires_metadata_and_exact_image_derivation(self):
        original = BytesIO()
        image = Image.new("RGB", (16, 16), (12, 55, 129))
        image.save(original, format="PNG")
        jpeg = BytesIO()
        image.save(jpeg, format="JPEG", quality=85)
        source = dict(claim="Claim", label="entailment", phenomenon="humor", explanation="Reason",
                      source_dataset="native", image={"bytes": original.getvalue()})
        local = dict(id="vflute_train_0", caption="Claim", label="ENTAILS", phenomenon="humor",
                     explanation="Reason", image_bytes=jpeg.getvalue())
        result = compare_row(local, source)
        self.assertTrue(result["lineage_verified"])
        self.assertEqual(result["image_lineage"], "EXACT_REPRODUCED_JPEG85")
        result = compare_row(dict(local, label="CONTRADICTS"), source)
        self.assertFalse(result["lineage_verified"])
        self.assertEqual(result["field_mismatches"], ["label"])
        self.assertTrue(compare_row(local, dict(source, claim="Claim\r\n", explanation=" Reason "))["lineage_verified"])
        self.assertFalse(compare_row(local, dict(source, claim="Different claim"))["lineage_verified"])
        with self.assertRaises(ValueError):
            compare_row(dict(local, image_bytes=b"changed"), source, cached=compare_row(local, source))
        other = BytesIO()
        Image.new("RGB", (16, 16), "white").save(other, format="PNG")
        self.assertFalse(compare_row(dict(local, image_bytes=other.getvalue()), source)["lineage_verified"])

    def test_source_metadata_is_hash_bound_to_corpus_and_sidecar(self):
        import hashlib
        from dataset.protocol import DATASET_REVISION
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / "data/provenance/vflute_d920d848"
            sidecar.mkdir(parents=True)
            path = root / "corpus.pkl"
            path.write_bytes(pickle.dumps([{"id": "a"}]))
            fields = ("source_dataset", "dataset_revision", "upstream_dataset_id", "upstream_split",
                      "upstream_row_index", "native_label", "upstream_image_sha256", "image_lineage", "text_transformations")
            entry = dict.fromkeys(fields, "fixture")
            entry.update(id="a", lineage_verified=True)
            raw = (json.dumps(entry) + "\n").encode()
            (sidecar / "provenance.jsonl").write_bytes(raw)
            summary = dict(all_source_lineage_verified=True, revision=DATASET_REVISION,
                provenance_sha256=hashlib.sha256(raw).hexdigest(),
                splits=[dict(split="vflute_train", local_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
            (sidecar / "verification.json").write_text(json.dumps(summary))
            with patch("dataset.loaders.BASE", str(root)):
                self.assertTrue(attach_provenance([{"id": "a"}], "vflute_train", path)[0]["source_lineage_verified"])
                (sidecar / "provenance.jsonl").write_bytes(raw + b" ")
                with self.assertRaisesRegex(ValueError, "sidecar checksum"):
                    attach_provenance([{"id": "a"}], "vflute_train", path)
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Dataset bytes changed"):
                    attach_provenance([{"id": "a"}], "vflute_train", path)

    def test_legacy_native_tasks_are_not_scored_as_visual_entailment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            pd.DataFrame([dict(id="mmsd2_test_1", ground_truth="ENTAILS", prediction="ENTAILS", phenomenon="sarcasm")]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "native tasks"):
                evaluate_predictions(path)


if __name__ == "__main__":
    unittest.main()
