from io import BytesIO
import unittest
from PIL import Image
from dataset.protocol import included, image_identity, validate_subset, grouped_partition


class DatasetProtocolTests(unittest.TestCase):
    def test_only_requested_three_phenomena_are_included(self):
        for name in ("humor", "metaphor", "sarcasm"):
            self.assertTrue(included({"phenomenon": name}))
        for name in ("idiom", "simile", "unknown"):
            self.assertFalse(included({"phenomenon": name}))

    def test_duplicate_ids_fail_without_silent_overwrite(self):
        row = {"id": "A", "label": "ENTAILS", "phenomenon": "humor"}
        with self.assertRaises(ValueError):
            validate_subset([row, row])

    def test_related_caption_pairs_stay_in_one_partition(self):
        rows = [{"id": str(i), "image_group_id": str(i // 2)} for i in range(20)]
        a = grouped_partition(rows, seed=9)
        b = grouped_partition(list(reversed(rows)), seed=9)
        self.assertEqual(set(a["calibration_ids"]), set(b["calibration_ids"]))
        for i in range(0, 20, 2):
            self.assertEqual(str(i) in a["calibration_ids"], str(i + 1) in a["calibration_ids"])

    def test_lossless_encoding_variants_share_pixel_identity(self):
        image = Image.new("RGB", (12, 12), "red")
        first, second = BytesIO(), BytesIO()
        image.save(first, "PNG", compress_level=0)
        image.save(second, "PNG", compress_level=9)
        a, b = image_identity(first.getvalue()), image_identity(second.getvalue())
        self.assertNotEqual(a["image_sha256"], b["image_sha256"])
        self.assertEqual(a["image_group_id"], b["image_group_id"])
