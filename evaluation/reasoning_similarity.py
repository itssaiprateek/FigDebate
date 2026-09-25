"""Offline reference similarity, never imported by the inference pipeline.

Run after inference against records.jsonl (all stages) or predictions.csv
(available exported stages). BERTScore is not semantic truth or image faithfulness.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import mean


def explanation_pairs(rows):
    """Do not silently exclude missing answers or invent absent initial traces."""
    pairs = []
    seen = set()
    for row in rows:
        identifier = row.get('id')
        if not identifier or identifier in seen:
            raise ValueError('Each record must have a unique nonempty id')
        seen.add(identifier)
        reference = str(row.get('reference_explanation') or '').strip()
        trace = row.get('trace') or {}
        stages = [('final', row.get('final_reason', ''))]
        if 'initial_decision' in trace:
            stages.insert(0, ('initial', trace['initial_decision'].get('explanation', '')))
        reviews = (trace.get('judge') or {}).get('tribunal_reviews') or []
        if reviews:
            stages.extend((f'proposal_{i+1}', review.get('reason', ''))
                          for i, review in enumerate(reviews))
        elif 'judge_reason' in row:
            stages.append(('proposal_exported', row.get('judge_reason', '')))
        for stage, text in stages:
            pairs.append({'id': identifier, 'stage': stage, 'reference': reference,
                          'generated': str(text or '').strip()})
    return pairs


def score_pairs(pairs, score_batch, token_count=None, max_tokens=None):
    """Missing references are unavailable; delivered empty answers score zero."""
    results = []
    pending = []
    for pair in pairs:
        result = dict(pair, precision=None, recall=None, f1=None)
        if not pair['reference']:
            result['status'] = 'REFERENCE_MISSING'
        elif not pair['generated']:
            result.update(status='EMPTY_ANSWER', precision=0., recall=0., f1=0.)
        elif token_count and max(token_count(pair['reference']),
                                 token_count(pair['generated'])) > max_tokens:
            result['status'] = 'INPUT_TOO_LONG_NOT_SILENTLY_TRUNCATED'
        else:
            result['status'] = 'SCORED'
            pending.append(len(results))
        results.append(result)
    if pending:
        precision, recall, f1 = score_batch([results[i]['generated'] for i in pending],
                                           [results[i]['reference'] for i in pending])
        if any(len(x) != len(pending) for x in (precision, recall, f1)):
            raise ValueError('Metric backend returned a mismatched result count')
        for j, i in enumerate(pending):
            results[i].update(precision=float(precision[j]), recall=float(recall[j]), f1=float(f1[j]))
    return results


def summarize(results):
    groups = defaultdict(list)
    for result in results:
        groups[result['stage']].append(result)
    summaries = {}
    for stage, rows in groups.items():
        usable = [r for r in rows if r['f1'] is not None]
        summaries[stage] = dict(samples=len(rows), scored_or_empty=len(usable),
            reference_missing=sum(r['status']=='REFERENCE_MISSING' for r in rows),
            empty_answers=sum(r['status']=='EMPTY_ANSWER' for r in rows),
            overlength=sum(r['status']=='INPUT_TOO_LONG_NOT_SILENTLY_TRUNCATED' for r in rows),
            **{f'mean_{k}':mean(r[k] for r in usable) if usable else None
               for k in ('precision','recall','f1')})
    by_case = defaultdict(dict)
    for row in results:
        by_case[row['id']][row['stage']] = row
    deltas = [{'id':identifier, 'final_minus_initial_f1':stages['final']['f1']-stages['initial']['f1']}
              for identifier, stages in by_case.items()
              if all(stage in stages and stages[stage]['f1'] is not None for stage in ('initial','final'))]
    return {'stages':summaries, 'paired_final_minus_initial':deltas,
            'mean_paired_final_minus_initial_f1':mean(d['final_minus_initial_f1'] for d in deltas) if deltas else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True, help='Existing local RoBERTa-large model directory; no downloads.')
    parser.add_argument('--model-revision', required=True, help='Pinned upstream commit SHA for the local weights.')
    parser.add_argument('--output-dir', type=Path)
    args=parser.parse_args()
    model=args.model_path.resolve(strict=True)
    config=json.loads((model/'config.json').read_text())
    if config.get('model_type')!='roberta' or config.get('num_hidden_layers')!=24:
        raise ValueError('This protocol requires RoBERTa-large, with layer 17 embeddings')
    if len(args.model_revision)!=40 or any(x not in '0123456789abcdef' for x in args.model_revision.lower()):
        raise ValueError('Supply the pinned 40-character upstream revision')
    with args.input.open(encoding='utf-8-sig',newline='') as stream:
        rows = list(csv.DictReader(stream)) if args.input.suffix.lower()=='.csv' else [json.loads(line) for line in stream if line.strip()]
    from bert_score import BERTScorer
    from bert_score.utils import sent_encode
    scorer=BERTScorer(model_type=str(model),num_layers=17,device='cpu',nthreads=4,
                     batch_size=8,idf=False,rescale_with_baseline=False,use_fast_tokenizer=False)
    # sent_encode follows this pinned metric version's prefix-space tokenization.
    # Use an unbounded tokenizer view to detect texts the backend would truncate.
    import copy
    tokenizer=copy.deepcopy(scorer._tokenizer)
    limit=tokenizer.model_max_length
    tokenizer.model_max_length=10**9
    results=score_pairs(explanation_pairs(rows),lambda candidates,references:scorer.score(candidates,references,batch_size=8),
                        lambda text:len(sent_encode(tokenizer,text)),limit)
    metadata=dict(metric='BERTScore',metric_role='reference_similarity_not_semantic_truth',
        model_id='FacebookAI/roberta-large',model_revision=args.model_revision,local_model=str(model),
        num_layers=17,idf=False,rescale_with_baseline=False,use_fast_tokenizer=False,device='cpu',
        bert_score_version=version('bert-score'),transformers_version=version('transformers'),
        source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        configuration_sha256=hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),
        tokenizer_max_tokens=limit,automatic_semantic_verification=False,image_faithfulness_verified=False)
    output=args.output_dir or args.input.parent
    output.mkdir(parents=True,exist_ok=True)
    (output/'reasoning_similarity.json').write_text(json.dumps(dict(configuration=metadata,
        summary=summarize(results),results=results),indent=2,ensure_ascii=True),encoding='utf-8')
    with (output/'reasoning_similarity.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','stage','status','precision','recall','f1','generated','reference'])
        writer.writeheader();writer.writerows(results)
    print(json.dumps(summarize(results),indent=2))


if __name__=='__main__':
    main()
