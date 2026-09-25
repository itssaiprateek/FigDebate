"""Regression tests for prompt-conformant answers and completed OCR retention."""
import unittest
from agents.visual_adapter import AtomicVisualQuestionController as Controller


class VisualAnswerRetentionTests(unittest.TestCase):
    def test_yes_no_enum_does_not_consume_the_five_factual_words(self):
        answer='YES, symbol attached to red bottle.'
        text,status,valid,error=Controller.validate_answer(answer,'Is a symbol visible?','yes_no')
        self.assertTrue(valid,error)
        self.assertEqual(text,answer)
        self.assertFalse(Controller.validate_answer('YES, symbol attached to the red bottle.',
                                                   'Is a symbol visible?','yes_no')[2])

    def test_finished_crop_text_is_preserved_and_capped_prose_is_rejected(self):
        text=' '.join(f'visible{i}' for i in range(230))+'.'
        for hit_cap in (False,True):
            result=Controller.validate_answer(text,'Transcribe the visible text.','ocr',
                                              {'hit_token_limit':hit_cap,'ended_by_eos':True})
            self.assertTrue(result[2],result[3])
            self.assertEqual(result[0],text)
        unfinished=Controller.validate_answer(text,'Transcribe the visible text.','ocr',
                                              {'hit_token_limit':True,'ended_by_eos':False})
        self.assertFalse(unfinished[2])
        self.assertEqual(unfinished[3],'truncated_response')


if __name__=='__main__':unittest.main()
