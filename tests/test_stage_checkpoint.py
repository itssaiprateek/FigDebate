import tempfile
import unittest

from engine.stage_checkpoint import StageCheckpointStore


class StageCheckpointTests(unittest.TestCase):
    def test_checkpoint_is_disabled_until_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StageCheckpointStore(directory, enabled=False)
            sample = {"index": 0, "raw": {"id": "case/1"}}
            store.save("visual_grounding", sample, {"facts": ["x"]})
            self.assertIsNone(store.load("visual_grounding", sample))

    def test_round_trip_uses_sample_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = {"index": 0, "raw": {"id": "case/1"}}
            writer = StageCheckpointStore(directory, enabled=False)
            writer.save("visual_grounding", sample, {"facts": ["x"]})
            reader = StageCheckpointStore(directory, enabled=True)
            self.assertEqual(
                reader.load("visual_grounding", sample), {"facts": ["x"]}
            )

    def test_checkpoint_rejects_configuration_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = {"index": 1, "raw": {"id": "sample-1"}}
            StageCheckpointStore(
                directory, enabled=False, fingerprint="first"
            ).save("visual", sample, {"ok": True})
            reader = StageCheckpointStore(
                directory, enabled=True, fingerprint="second"
            )
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                reader.load("visual", sample)


if __name__ == "__main__":
    unittest.main()
