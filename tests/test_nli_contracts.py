import unittest

from models.nli_loader import NliVerifier


class NliContractTests(unittest.TestCase):
    def test_binary_resolution_rejects_neutral_winner(self):
        label, confidence = NliVerifier.resolve_binary(
            {"contradiction": 0.2, "entailment": 0.3, "neutral": 0.5}
        )
        self.assertIsNone(label)
        self.assertEqual(confidence, 0.5)

    def test_binary_resolution_normalizes_selected_binary_probability(self):
        label, confidence = NliVerifier.resolve_binary(
            {"contradiction": 0.8, "entailment": 0.1, "neutral": 0.1}
        )
        self.assertEqual(label, "CONTRADICTS")
        self.assertAlmostEqual(confidence, 0.8 / 0.9)


if __name__ == "__main__":
    unittest.main()
