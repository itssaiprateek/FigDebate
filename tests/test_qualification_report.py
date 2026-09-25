import unittest
from evaluation.qualification_report import explanation_status, origin_report, correction_metrics, calibration_eligibility


class QualificationReportTests(unittest.TestCase):
    def test_cap_is_signal_not_semantic_failure(self):
        value = explanation_status('A complete sentence.', [{'hit_token_limit': True}])
        self.assertTrue(value['completion_needs_review'])
        self.assertFalse(value['dangling_clause_signal'])
        self.assertEqual(value['semantic_correctness'], 'NOT_ASSESSED')

    def test_copy_origin_and_semantic_review_are_separate(self):
        row = {'id': 'x', 'initial_decision': {'explanation': 'The object is'}, 'decision': {'explanation': 'The object is'}}
        report = origin_report(row, [{'stage': 'initial_arbiter', 'semantic_sound': False, 'reason': 'wrong object'}])
        self.assertEqual(report['stages'][-1]['exact_copy_of'], ['initial_arbiter'])
        self.assertEqual(report['first_reviewed_semantic_error']['reason'], 'wrong object')
        self.assertFalse(report['causal_origin_established'])

    def test_failures_stay_in_denominator_and_wrong_reason_is_explicit(self):
        rows = [{'id': 'x', 'initial_decision': {'label': 'CONTRADICTS'}, 'decision': {'label': 'ENTAILS'},
            'judge': {'tribunal_reviews': [{'provisional_verdict': 'ENTAILS'}], 'tribunal_resolution': {'accepted': True}}},
            {'id': 'missing', 'decision': {}}]
        report = correction_metrics(rows, {'x': 'ENTAILS'}, {'x': {'proposal_1': {'semantic_sound': False, 'decisive_reason_sound': False}}})
        self.assertEqual(report['correct_fraction_all_cases_lower_bound'], .5)
        self.assertEqual(report['official_accuracy_referenced_cases'], 1.)
        self.assertEqual(report['missing_final'], 1)
        self.assertEqual(report['missing_reference'], 1)
        self.assertEqual(report['helpful_changes'], 1)
        self.assertEqual(report['unsound_accepted'], 1)
        self.assertEqual(report['correct_label_wrong_reason'], 1)

    def test_duplicate_results_cannot_improve_denominator(self):
        with self.assertRaises(ValueError):
            correction_metrics([{'id': 'x'}, {'id': 'x'}], {})

    def test_calibration_requires_disjoint_images_and_qualified_verifier(self):
        manifest = {'semantic_qualification_passed': True, 'development_image_groups': ['same'],
                    'calibration_image_groups': ['same'], 'test_image_groups': ['new'], 'frozen_criterion': 'fixed'}
        report = calibration_eligibility(manifest)
        self.assertFalse(report['eligible_to_evaluate'])
        self.assertIn('image_group_leakage', report['reasons'])
        self.assertFalse(report['threshold_changed'])

    def test_unreviewed_is_not_incorrect(self):
        report = correction_metrics([{'id': 'x', 'judge': {'tribunal_reviews': [{'provisional_verdict': 'ENTAILS'}]}}], {})
        self.assertEqual(report['semantic_unreviewed'], 1)
        self.assertEqual(report['unsound_rejected'], 0)

    def test_no_reason_field_is_not_missing_visual_answer(self):
        report = origin_report({'id': 'x', 'visual_output': {'facts': ['A chair.']}, 'decision': {'explanation': 'A chair is visible.'}})
        self.assertIsNone(report['first_recorded_delivery_signal'])

    def test_original_language_and_visual_outputs_retained_without_certification(self):
        report = origin_report({'id': 'x', 'visual_output': {'visual_facts': ['A person with a pad.']},
            'language_output': {'caption_proposition': 'Maybe defensive.', 'intended_meaning': 'A confident diagnosis.'}})
        self.assertEqual(report['stages'][0]['recorded_content']['visual_facts'], ['A person with a pad.'])
        self.assertEqual(report['stages'][1]['recorded_content']['intended_meaning'], 'A confident diagnosis.')
        self.assertIsNone(report['first_reviewed_semantic_error'])

    def test_confirmation_and_abstention_are_not_rejected_corrections(self):
        rows = [{'id': 'x', 'initial_decision': {'label': 'ENTAILS'}, 'judge': {
            'tribunal_reviews': [{'provisional_verdict': 'ENTAILS'}, {'provisional_verdict': 'ABSTAIN'}]}}]
        report = correction_metrics(rows, {'x': 'ENTAILS'}, {'x': {
            'proposal_1': {'semantic_sound': True}, 'proposal_2': {'semantic_sound': True}}})
        self.assertEqual(report['sound_confirmations'], 1)
        self.assertEqual(report['reviewed_unresolved'], 1)
        self.assertEqual(report['sound_rejected'], 0)

    def test_missing_initial_is_not_a_helpful_correction(self):
        report = correction_metrics([{'id': 'x', 'final_label': 'ENTAILS'}], {'x': 'ENTAILS'})
        self.assertEqual(report['missing_initial'], 1)
        self.assertEqual(report['helpful_changes'], 0)
        self.assertEqual(report['final_correct'], 1)
