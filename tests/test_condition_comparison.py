import copy
import unittest
from evaluation.restricted_comparison import evaluate

class ComparisonTests(unittest.TestCase):
    def fixture(self):
        return {'claim_quote':'A took less time than B','operator':'LT',
          'left':{'subject':'A','scope':'elapsed duration','evidence_id':'E1','quote':'A took 2 hours','value':'2','unit':'hour'},
          'right':{'subject':'B','scope':'elapsed duration','evidence_id':'E2','quote':'B took 180 minutes','value':'180','unit':'minute'}}
    def run_check(self, c):
        return evaluate(c,'A took less time than B',[{'id':'E1','text':'A took 2 hours'}, {'id':'E2','text':'B took 180 minutes'}])
    def test_explicit_conversion_and_opposite_claim(self):
        c=self.fixture();self.assertEqual(self.run_check(c)['status'],'MATCH')
        c['operator']='GT';self.assertEqual(self.run_check(c)['status'],'MISMATCH')
    def test_scope_units_quotes_values_and_sources_cannot_be_guessed(self):
        for field,value in [('unit','month'),('value','NaN'),('value','3'),('quote','invented 2 hours'),('scope','another event'),('evidence_id','unknown')]:
            c=self.fixture();c['left'][field]=value
            with self.subTest(field=field): self.assertEqual(self.run_check(c)['status'],'UNRESOLVED')
    def test_swapping_operands_and_operator_preserves_arithmetic(self):
        c=self.fixture();before=self.run_check(c)['status'];c['left'],c['right']=c['right'],c['left'];c['operator']='GT'
        self.assertEqual(self.run_check(c)['status'],before)
    def test_not_a_label_certificate(self):
        self.assertNotIn('relation', self.run_check(self.fixture()))
        self.assertIn('fallible', self.run_check(self.fixture())['authority'])
