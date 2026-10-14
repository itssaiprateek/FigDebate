"""Restricted, source-bound numeric comparison. No label authority or generated code.

The extraction remains a fallible model interpretation. Results are audit context,
never certificates of full-caption truth, and unknown inputs never imply conflict.
"""
from decimal import Decimal, InvalidOperation
import re
from engine.output_contracts import object_schema

VERSION = 'explicit-quantity-comparison-1'
UNITS = {'second': Decimal(1), 'minute': Decimal(60), 'hour': Decimal(3600),
         'day': Decimal(86400), 'week': Decimal(604800), 'count': Decimal(1)}
OPERATORS = {'LT', 'LE', 'EQ', 'NE', 'GE', 'GT'}


def schema(known):
    text = {'type':'string', 'minLength':1}
    operand = object_schema({'subject':text, 'scope':text, 'evidence_id':{'type':'string','enum':sorted(known)},
        'quote':text, 'value':text, 'unit':{'type':'string','enum':list(UNITS)}})
    return {'type':'array','maxItems':1,'items':object_schema({
        'claim_quote':text, 'operator':{'type':'string','enum':sorted(OPERATORS)},
        'left':operand,'right':operand})}


INSTRUCTIONS = (
    'comparisons is [] unless two EXPLICIT numeric actual outcomes can be compared under the same scope. '
    'If applicable, copy each operand quote from a cited observation; value must be its written numeral. '
    'Use count, second, minute, hour, day or week; no guessed values for vague quantities, dates or months. '
    'Preserve each participant and the same measured property/time in scope. claim_quote is an exact '
    'caption span stating the ordering, and operator is its asserted LT/LE/EQ/NE/GE/GT relation. '
    'Do not turn a wish, instruction, promise or prediction into an observed outcome. '
    'This optional arithmetic check cannot establish other caption clauses or a figurative mapping. '
)


def evaluate(check, source, observations):
    """Compute only the translated ordering; preserve unknowns and full provenance."""
    unknown = lambda why: {'protocol':VERSION, 'status':'UNRESOLVED', 'reason':why,
                          'authority':'diagnostic_not_full_claim_verification'}
    if not isinstance(check,dict) or check.get('operator') not in OPERATORS:
        return unknown('Unsupported comparison operator')
    if not isinstance(source,str) or not isinstance(check.get('claim_quote'),str) or not check['claim_quote'] or check['claim_quote'] not in source:
        return unknown('Comparison is not source-bound')
    records = {r.get('id',r.get('evidence_id')):r.get('text',r.get('observed','')) for r in observations}
    values=[];units=[]
    for name in ('left','right'):
        operand=check.get(name,{})
        if not isinstance(operand,dict): return unknown('Invalid operand record')
        quote=operand.get('quote',''); value=operand.get('value',''); unit=operand.get('unit')
        if not isinstance(quote,str) or not isinstance(value,str): return unknown('Invalid operand text')
        if not quote or quote not in records.get(operand.get('evidence_id'),''):
            return unknown('Operand is not bound to a selected observation')
        if not operand.get('subject','').strip() or not operand.get('scope','').strip():
            return unknown('Missing participant or scope')
        if unit not in UNITS or not re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?',value):
            return unknown('Unsupported or non-finite quantity')
        # Do not silently choose one of several numbers from an operand quote.
        if re.findall(r'(?<![\w.])-?(?:[0-9]+)(?:\.[0-9]+)?(?![\w.])',quote) != [value]:
            return unknown('Operand quote does not uniquely establish its numeral')
        if unit != 'count' and not re.search(r'\b'+unit+r's?\b',quote,re.I):
            return unknown('Unit is not explicit in the operand quote')
        try: values.append(Decimal(value)*UNITS[unit])
        except (InvalidOperation,ValueError): return unknown('Invalid numeric operand')
        units.append(unit)
    if (units[0]=='count') != (units[1]=='count'):
        return unknown('Incompatible quantity dimensions')
    if check['left']['scope'] != check['right']['scope']:
        return unknown('Operands have different scope')
    a,b=values;op=check['operator']
    outcome={'LT':a<b,'LE':a<=b,'EQ':a==b,'NE':a!=b,'GE':a>=b,'GT':a>b}[op]
    return {'protocol':VERSION,'status':'MATCH' if outcome else 'MISMATCH',
        'normalized_values':[str(a),str(b)],'operator':op,
        'authority':'arithmetic_over_fallible_translation_not_full_claim_verification'}
