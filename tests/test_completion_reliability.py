import unittest
from engine.semantic_checks import semantic_valid
from engine.review_routing import route_question


class CompletionReliabilityTests(unittest.TestCase):
    def test_retired_candidate_profile_cannot_be_enabled(self):
        from engine.runtime_profile import PROFILES, resolve_runtime_profile
        self.assertNotIn('completion-candidate-8gb', PROFILES)
        with self.assertRaises(ValueError):
            resolve_runtime_profile('completion-candidate-8gb')
        self.assertNotIn('judge_structured_interpretation', PROFILES['paper-8gb'].as_dict())

    def test_run_command_rejects_retired_candidate(self):
        import contextlib
        import io
        from unittest.mock import patch
        from run_figdebate import parse_args
        with patch('sys.argv', ['run_figdebate.py', '--hardware-profile', 'completion-candidate-8gb']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                parse_args()
        self.assertEqual(error.exception.code, 2)

    def test_missing_scope_and_negative_evidence_cannot_certify_conflict(self):
        self.assertFalse(semantic_valid({'_semantic_check_version':'1','observations':[],
          'coverage':{'required_scope':'all panels','observed_scope':'one panel','missing_decisive_scope':'other panel'}},set()))
        self.assertFalse(semantic_valid({'_semantic_check_version':'1','relation':'CONFLICT','condition_checks':[],
          'role_check':{'claim_bearer':'author','image_bearer':'author','same_subject_and_scope':True,'missing_evidence_only':True}},set()))

    def test_unknown_evidence_cannot_support_critic(self):
        self.assertFalse(semantic_valid({'_semantic_check_version':'1','decision_errors':[],
            'objection_checks':[{'evidence_ids':['unknown'],'supported':False}]},{'VF1'}))

    def test_metadata_question_never_reaches_pixel_witness(self):
        self.assertEqual(route_question('Read the image metadata to identify its date.')[0],'blocked')
        self.assertEqual(route_question('What does the phrase historical records mean?', 'CAPTION_PREMISE')[0], 'language')

    def test_repeat_question_cannot_justify_another_semantic_hearing(self):
        from engine.review_routing import new_semantic_questions
        self.assertEqual(new_semantic_questions(['Which person is speaking?'], [' which PERSON is speaking ']), [])
        self.assertEqual(new_semantic_questions(['Which outcome actually happened?', 'Who desired it?'], []),
                         ['Which outcome actually happened?'])

    def test_advisor_check_is_part_of_verification_identity(self):
        from engine.independent_review import subject_hash
        proposal = {'source_caption': 'A calm meeting.'}
        original = subject_hash(proposal, [])
        proposal['advisor_check'] = {'distinguishing_check': 'Who looks calm?'}
        self.assertNotEqual(original, subject_hash(proposal, []))

    def test_advisor_metrics_do_not_claim_causal_benefit(self):
        from evaluation.tribunal_quality import summarize_tribunal
        advice = {'_format_valid': False, '_execution_status': 'FAILED', '_generation_seconds': 60}
        result = summarize_tribunal([{'ground_truth':'ENTAILS', 'initial_prediction':'ENTAILS',
            'prediction':'ENTAILS', 'judge_requested':False,
            'trace':{'debate_details':{'semantic_advisor':advice}}}])
        self.assertEqual(result['semantic_advisor']['failed'], 1)
        self.assertEqual(result['semantic_advisor']['seconds'], 60)
        self.assertIn('not_established', result['semantic_advisor_accuracy_contribution'])
