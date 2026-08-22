import json
import os
import tempfile
import unittest

from engine.run_integrity import validate_resume_config
from engine.sampling import select_records


class ResumeSafetyTests(unittest.TestCase):
    def _config(self):
        return {
            "dataset": "vflute_val",
            "requested_samples": 50,
            "execution_mode": "stagewise",
            "debate_mode": "enabled",
            "evidence_mode": "enabled",
            "feedback_mode": "disabled",
            "verified_feedback_sha256": None,
            "model_vision": "vision",
            "model_vision_revision": "v1",
            "model_language": "language",
            "model_language_revision": "v1",
            "seed": 42,
            "selection_strategy": "stratified",
            "dataset_selection_sha256": "data",
            "pipeline_source_sha256": "source",
            "evidence_ledger_version": "5.0",
        }

    def test_identical_resume_configuration_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config()
            with open(os.path.join(directory, "run_config.json"), "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            loaded = validate_resume_config(directory, dict(config))
        self.assertEqual(loaded["seed"], 42)

    def test_mismatched_resume_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config()
            with open(os.path.join(directory, "run_config.json"), "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            changed = dict(config)
            changed["debate_mode"] = "disabled"
            with self.assertRaisesRegex(ValueError, "incompatible configuration"):
                validate_resume_config(directory, changed)


class SamplingSafetyTests(unittest.TestCase):
    def test_stratified_sampling_is_deterministic_and_balanced(self):
        rows = [
            {"id": f"{phenomenon}-{label}-{index}", "phenomenon": phenomenon,
             "label": label}
            for phenomenon in ("humor", "metaphor", "sarcasm")
            for label in ("ENTAILS", "CONTRADICTS")
            for index in range(4)
        ]
        first = select_records(rows, 12, "stratified", seed=7)
        second = select_records(rows, 12, "stratified", seed=7)
        self.assertEqual(
            [item["id"] for item in first], [item["id"] for item in second]
        )
        groups = {
            (item["phenomenon"], item["label"]) for item in first
        }
        self.assertEqual(len(groups), 6)


if __name__ == "__main__":
    unittest.main()
