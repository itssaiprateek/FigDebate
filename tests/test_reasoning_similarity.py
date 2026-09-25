import unittest
from evaluation.reasoning_similarity import explanation_pairs, score_pairs, summarize


class ReasoningSimilarityTests(unittest.TestCase):
    def test_missing_answers_remain_in_denominator_but_missing_references_do_not(self):
        pairs=[dict(id='a',stage='final',generated='',reference='Evidence.'),
               dict(id='b',stage='final',generated='Answer.',reference='')]
        results=score_pairs(pairs,lambda *args: self.fail('Nothing should be encoded'))
        summary=summarize(results)['stages']['final']
        self.assertEqual(summary['mean_f1'],0.)
        self.assertEqual(summary['empty_answers'],1)
        self.assertEqual(summary['reference_missing'],1)

    def test_all_stages_use_the_same_case_reference(self):
        pairs=explanation_pairs([dict(id='a',reference_explanation='Human reason.',final_reason='Final.',
            trace={'initial_decision':{'explanation':'Initial.'},'judge':{'tribunal_reviews':[{'reason':'Proposal.'}]}})])
        self.assertEqual([p['stage'] for p in pairs],['initial','final','proposal_1'])
        captured=[]
        def scorer(candidates,references):
            captured.append(references)
            return [.1,.3,.2],[.1,.3,.2],[.1,.3,.2]
        result=score_pairs(pairs,scorer)
        self.assertEqual(captured,[['Human reason.']*3])
        self.assertAlmostEqual(summarize(result)['mean_paired_final_minus_initial_f1'],.2)

    def test_oversized_inputs_are_reported_without_silent_truncation(self):
        pairs=[dict(id='a',stage='final',generated='long text',reference='short')]
        result=score_pairs(pairs,lambda *a:self.fail('Oversized text should not be scored'),len,5)
        self.assertEqual(summarize(result)['stages']['final']['overlength'],1)
        self.assertIsNone(result[0]['f1'])

    def test_duplicate_ids_are_rejected_and_csv_does_not_invent_initial_reason(self):
        row=dict(id='a',reference_explanation='Human.',final_reason='Final.')
        self.assertEqual([p['stage'] for p in explanation_pairs([row])],['final'])
        with self.assertRaises(ValueError):explanation_pairs([row,row])


if __name__=='__main__':unittest.main()
