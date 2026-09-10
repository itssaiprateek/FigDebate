import unittest

from dataset.prepare_vflute import normalize_label, validate_records


class _ClassLabel:
    def int2str(self, value):
        return ("ENTAILS", "CONTRADICTS")[value]


class DataPreparationTests(unittest.TestCase):
    def test_untyped_numeric_labels_must_not_be_guessed(self):
        for value in (0, 1, "0", "1"):
            with self.assertRaises(ValueError):
                normalize_label(value)

    def test_label_normalization_uses_dataset_class_label(self):
        self.assertEqual(normalize_label(0, _ClassLabel()), "ENTAILS")
        self.assertEqual(normalize_label(1, _ClassLabel()), "CONTRADICTS")

    def test_record_validation_rejects_duplicate_ids(self):
        records = [
            {"id": "same", "label": "ENTAILS"},
            {"id": "same", "label": "CONTRADICTS"},
        ] * 25
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_records("vflute_train_dev50", records)


if __name__ == "__main__":
    unittest.main()
