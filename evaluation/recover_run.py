"""Export durable results/checkpoints into a new directory, without model inference."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, target = Path(args.source_run), Path(args.output)
    from engine.result_store import read_records, atomic_json, commit_record, completion_manifest
    from engine.stage_checkpoint import StageCheckpointStore
    from dataset.loaders import load_split
    from engine.batch_runner import StagewiseRunner
    from engine.final_artifact import final_artifact
    from run_figdebate import build_record, write_predictions, write_debate_logs, write_feedback_decision_logs
    from evaluation.evaluate_predictions import evaluate_predictions
    from evaluation.tribunal_quality import summarize_tribunal
    config = json.loads((source/'run_config.json').read_text(encoding='utf-8'))
    ids = json.loads((source/'selected_ids.json').read_text(encoding='utf-8'))
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate selection identities')
    data = {r['id']:r for r in load_split(config['dataset']) if r['id'] in ids}
    if set(data) != set(ids):
        raise ValueError('Dataset selection mismatch')
    existing = read_records(source/'records.jsonl')
    if not set(existing) <= set(ids):
        raise ValueError('Foreign records in source run')
    store = StageCheckpointStore(source/'stage_checkpoints', enabled=True, fingerprint=config['stage_fingerprints'])
    records, provenance = [], []
    for index, key in enumerate(ids):
        raw = data[key]; sample = {'index':index, 'raw':raw, 'caption':raw['caption']}
        state = None; stage = None
        for name in ('final_result', 'tribunal_round_2', 'tribunal_round_1'):
            state = store.load(name, sample)
            if state is not None:
                stage = name
                break
        if key in existing:
            record = existing[key]
            import hashlib
            if record.get('image_sha256') != raw['image_sha256'] or record.get('caption_sha256') != hashlib.sha256(raw['caption'].encode()).hexdigest():
                raise ValueError('Durable result image/caption mismatch')
            if state and (record['initial_prediction'] != state['initial_decision']['label'] or record['final_prediction'] != state['decision']['label']):
                raise ValueError('Durable result and latest checkpoint disagree')
            origin = 'durable_original_record'
        elif state is not None:
            if stage != 'final_result' and config.get('feedback_mode') not in ('precedent','disabled','integrated'):
                raise ValueError('Cannot infer post-tribunal feedback result for this mode')
            state = deepcopy(state)
            if stage == 'tribunal_round_1' and state.get('judge', {}).get('tribunal_session', {}).get('state') == 'FOLLOW_UP_REQUIRED':
                raise ValueError('Pending follow-up hearing is not a completed case: ' + key)
            if stage != 'final_result':
                mode = config.get('feedback_mode', 'disabled')
                state.setdefault('feedback', {}).update(mode=mode, enabled=mode != 'disabled',
                    accounting_status='reconstructed_from_run_config_and_completed_hearings')
            StagewiseRunner._finish_timing(state)
            state['final_artifact'] = final_artifact(raw['caption'], state)
            record = build_record(index, raw, state, state['timing']['sample_inference_seconds'])
            origin = 'reconstructed_' + stage
        else:
            raise ValueError('No complete final result/checkpoint for ' + key)
        records.append(record); provenance.append({'id':key,'origin':origin})
    target.mkdir(parents=True, exist_ok=False)
    for record in records:
        commit_record(target, record)
    for name in ('run_config.json','sample_manifest.json','selected_ids.json','evaluation_references.json'):
        (target/name).write_bytes((source/name).read_bytes())
    atomic_json(target/'recovery_manifest.json', {'source_run':str(source.resolve()),'inference_performed':False,'provenance':provenance,
                'limitation':'Reconstructed tribunal checkpoints omit unsaved post-stage accounting; original failure history remains in source.'})
    progress = {'status':'inference_complete_export_pending','completed_samples':len(records),'requested_samples':len(ids),'completed_ids':ids}
    atomic_json(target/'progress.json', progress)
    try:
        (target/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in records),encoding='utf-8')
        write_predictions(target/'predictions.csv', records)
        write_debate_logs(target, records); write_feedback_decision_logs(target, records)
        metrics = evaluate_predictions(target/'predictions.csv', target)
        atomic_json(target/'tribunal_quality.json', summarize_tribunal(records))
        completion_manifest(target, ids, records)
    except Exception as error:
        progress.update(failure_type=type(error).__name__,failure_reason=str(error))
        atomic_json(target/'progress.json', progress)
        raise
    progress['status']='complete'; atomic_json(target/'progress.json', progress)
    print(json.dumps({'samples':len(records),'accuracy':metrics['accuracy'],'inference_performed':False,'output':str(target)}))


if __name__ == '__main__':
    main()
