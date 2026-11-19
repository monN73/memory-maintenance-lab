"""Replay N9 final requests with two readers, with and without tagged source messages."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from eval_and_report import judge_one
from llm_client import DEFAULT_MODELS, LLMClient
from metrics import estimate_messages_tokens


def load(path):
    return json.loads(path.read_text())


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    batch = args.batch.resolve()
    report = ROOT / 'reports' / batch.parent.name / batch.name
    scores = {x['question_id']: x for x in map(json.loads, (report/'eval_N9.jsonl').open())}
    clients = {p: LLMClient(provider=p) for p in DEFAULT_MODELS}
    summary = load(report/'summary.json')
    assert summary['judge_provider'] == 'ecnu'
    judge_model = summary['judge_model']
    refs = json.loads((ROOT/'data/longmemeval/longmemeval_s_cleaned.json').read_text())
    out = batch/'answer-model-comparison'
    out.mkdir(exist_ok=True)
    def run(job):
        item, provider, condition, original, messages, ids = job
        qid = item['question_id']
        path = out/f'{qid}-{provider}-{condition}.json'
        record = load(path) if path.exists() else {}
        if not record:
            reused = provider == 'ecnu' and condition == 'original'
            response = original['response'] if reused else clients[provider].chat(
                DEFAULT_MODELS[provider], messages, seed=42, thinking='disabled' if provider=='deepseek' else None)
            record = dict(question_id=qid, question=item['question'], expected=item['answer'], provider=provider,
                          model=DEFAULT_MODELS[provider], condition=condition, messages=messages,
                          estimated_tokens=estimate_messages_tokens(messages), source_messages=ids,
                          response=response, reused=reused)
            save(path, record)
        if record.get('judgment', {}).get('judge_status') != 'ok':
            scored = scores[qid] if record['reused'] else judge_one(clients['ecnu'], judge_model, item,
                       dict(question_id=qid,status='completed',hypothesis=record['response']['text']))
            record['judgment'] = {k:scored.get(k) for k in ('judge_status','judge_label','judge_raw','judge_usage')}
            record['judge_model'] = judge_model
            save(path,record)
        if record['judgment']['judge_status'] != 'ok':
            raise RuntimeError(record['judgment']['judge_status'])
        return {k:v for k,v in record.items() if k not in ('messages','response')} | {'answer':record['response']['text']}
    jobs=[]
    for item in refs:
        qid=item['question_id']
        if qid not in scores or item['question_type']!='knowledge-update': continue
        directory=next((batch/'runs').glob(f'*-{qid}-direct-N9-r1'))
        original=load(directory/'requests/final_answer.json')
        msgs=original['messages']
        evidence=[]; ids=[]
        for sid,date,session in zip(item['haystack_session_ids'], item['haystack_dates'],item['haystack_sessions']):
            for i,m in enumerate(session):
                if m.get('has_answer') is True:
                    evidence.append(dict(role=m['role'],content=f"[{date}] {m['content']}"))
                    ids.append([sid,i])
        assert evidence
        for condition,add in [('original',[]),('restored',evidence)]:
            messages=msgs[:-1]+add+msgs[-1:]
            assert estimate_messages_tokens(messages)<=load(directory/'manifest.json')['effective_budget']['input_tokens']
            for provider in clients:
                jobs.append((item,provider,condition,original,messages,ids if add else []))
    assert len(jobs)==16
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(run,jobs))
    save(report/'answer-model-comparison.json',dict(source_batch=batch.name,strategy='N9',rows=rows))


if __name__=='__main__':
    main()
