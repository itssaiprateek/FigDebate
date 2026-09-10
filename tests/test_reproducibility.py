import copy
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import run_reproducible
from engine.reproducibility import derived_seed, seed_stage
from engine.sampling import build_selection_manifest, select_from_manifest, select_records


def rows():
    output = []
    for phenomenon in ("humor", "metaphor", "sarcasm"):
        for label in ("ENTAILS", "CONTRADICTS"):
            for index in range(5):
                output.append({
                    "id": f"{phenomenon}-{label}-{index}",
                    "phenomenon": phenomenon,
                    "label": label,
                    "caption": f"caption {phenomenon} {label} {index}",
                    "image_bytes": f"image-{phenomenon}-{label}-{index}".encode(),
                })
    return output


class ReproducibilityTests(unittest.TestCase):
    def test_stratified_subsets_are_nested_including_full_set(self):
        records = rows()
        ten = select_records(records, 10, "stratified", seed=42)
        twenty = select_records(records, 20, "stratified", seed=42)
        full = select_records(records, len(records), "stratified", seed=42)
        self.assertEqual([x["id"] for x in ten], [x["id"] for x in twenty[:10]])
        self.assertEqual([x["id"] for x in twenty], [x["id"] for x in full[:20]])

    def test_manifest_validates_content_and_selects_prefix(self):
        records = rows()
        manifest = build_selection_manifest(records, "stratified", 42)
        self.assertEqual(len(select_from_manifest(records, manifest, 10)), 10)
        tampered = copy.deepcopy(manifest)
        tampered["records"][0]["label"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "checksum"):
            select_from_manifest(records, tampered, 10)

    def test_stage_seed_is_order_independent(self):
        first = derived_seed(42, "sample-a", "judge", 1)
        seed_stage(42, "unrelated", "judge", 1)
        second = derived_seed(42, "sample-a", "judge", 1)
        self.assertEqual(first, second)
        self.assertNotEqual(first, derived_seed(42, "sample-a", "judge", 2))

    def test_windows_safe_launcher_preserves_paths_with_spaces(self):
        completed = SimpleNamespace(returncode=0)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                sys,
                "argv",
                ["run_reproducible.py", "--seed", "7", "--help"],
            ),
            patch.object(
                run_reproducible.subprocess,
                "run",
                return_value=completed,
            ) as mocked_run,
        ):
            with self.assertRaisesRegex(SystemExit, "0"):
                run_reproducible.main()

        command = mocked_run.call_args.args[0]
        child_environment = mocked_run.call_args.kwargs["env"]
        self.assertEqual(command[0], sys.executable)
        self.assertTrue(os.path.isabs(command[1]))
        self.assertEqual(command[2:], ["--seed", "7", "--help"])
        self.assertEqual(child_environment["PYTHONHASHSEED"], "7")


if __name__ == "__main__":
    unittest.main()
