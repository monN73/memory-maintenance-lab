"""Replay N7 archive writes with bounded policies. No model clients or network."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from budget_executor import BudgetConfig, Exchange, bm25_retrieve, build_messages, serialize_exchanges
from dataset_adapters import align_longmemeval_turns
from metrics import estimate_messages_tokens, estimate_text_tokens

POLICIES = ('n7_unbounded', 'fifo', 'dedup_fifo', 'session_balanced')
MEMORY_MARKER = '\n\n[显式记忆 M]\n'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def payload(exchange):
    def normalize(text):
        return ' '.join(text.split()) if text is not None else None
    return normalize(exchange.user_text), normalize(exchange.assistant_text)


def maintain(archive, incoming, policy, cap, cost, exchanges):
    if policy == 'n7_unbounded':
        return archive + incoming
    kept = archive.copy()
    for tid in incoming:
        if cost[tid] > cap:
            continue
        if policy == 'dedup_fifo':
            key = payload(exchanges[tid])
            kept = [old for old in kept if payload(exchanges[old]) != key]
        kept.append(tid)
        total = sum(cost[i] for i in kept)
        while total > cap:
            if policy == 'session_balanced':
                by_session = Counter()
                for i in kept:
                    by_session[exchanges[i].session_id] += cost[i]
                victim = min(kept, key=lambda i: (-by_session[exchanges[i].session_id], i))
            else:
                victim = kept[0]
            kept.remove(victim)
            total -= cost[victim]
    return kept


def replay_item(run, raw, cap, checkpoints):
    manifest = load(run / 'manifest.json')
    qid = manifest['dataset']['question_id']
    require(manifest['status'] == 'completed', f'{qid}: N7 run did not complete')
    turns = align_longmemeval_turns(raw)
    exchanges = {
        t.turn_id: Exchange(t.turn_id, t.session_id, t.user_text, t.assistant_text, t.timestamp)
        for t in turns
    }
    evidence_msg = {i for i, m in enumerate(m for session in raw['haystack_sessions'] for m in session)
                    if m.get('has_answer') is True}
    evidence = {t.turn_id for t in turns if evidence_msg.intersection(t.source_message_indices)}
    cost = {i: estimate_text_tokens(serialize_exchanges([e])) for i, e in exchanges.items()}
    request = load(run / 'requests' / 'final_answer.json')
    final_tid = request['turn_id']
    require(final_tid == len(exchanges)+1, f'{qid}: inconsistent final turn')
    events = jsonl(run / 'events.jsonl') if (run / 'events.jsonl').exists() else []
    query = request['messages'][-1]['content']
    system, _, original_memory = request['messages'][0]['content'].partition(MEMORY_MARKER)
    cfg = BudgetConfig(**manifest['effective_budget'])
    # Treat the saved N7 eviction schedule as fixed; archive variants never change the recent window.
    discarded = set()
    stores = {p: [] for p in POLICIES}

    def checkpoint(tid, phase, writes, evicted):
        require(not discarded.intersection(evicted), f'{qid}: duplicate eviction')
        discarded.update(evicted)
        seen_evidence = {i for i in evidence if i <= tid}
        recent_evidence = seen_evidence - discarded
        for policy, archive in stores.items():
            archive = stores[policy] = maintain(archive, writes, policy, cap, cost, exchanges)
            stored = sum(cost[i] for i in archive)
            require(policy == 'n7_unbounded' or stored <= cap, f'{qid}: {policy} overflow')
            archived_evidence = seen_evidence.intersection(archive)
            checkpoints.write(json.dumps({
                'question_id': qid, 'turn_id': tid, 'phase': phase, 'policy': policy,
                'seen_evidence_turns': len(seen_evidence),
                'archive_evidence_turns': len(archived_evidence),
                'retained_evidence_turns': len(recent_evidence | archived_evidence),
                'archive_tokens_est': stored,
            }) + '\n')

    expected_ids = []
    has_query_readout = False
    for event in events:
        tid, mechanism = event['turn_id'], event['mechanism']
        evicted = event.get('evicted_turn_ids', [])
        if mechanism == 'bm25_index_eviction':
            require(tid < final_tid, f'{qid}: archive write at unexpected turn {tid}')
            checkpoint(tid, 'ingestion', evicted, evicted)
        elif mechanism in ('target_unreachable_eviction', 'zero_memory_budget_eviction'):
            checkpoint(tid, 'final' if tid == final_tid else 'ingestion', [], evicted)
        elif mechanism == 'bm25_retrieval':
            require(tid == final_tid and not has_query_readout, f'{qid}: unexpected query event')
            # N7 can drop more recent turns after running BM25. These are not in the read pool.
            baseline_archive = stores['n7_unbounded']
            prefix = next((n for n in range(len(evicted), -1, -1)
                           if serialize_exchanges([exchanges[i] for i in baseline_archive + evicted[:n]])
                           == event['before_text']), None)
            require(prefix is not None, f'{qid}: could not reconstruct logged N7 pool')
            checkpoint(tid, 'final', evicted[:prefix], evicted)
            has_query_readout = True
            expected_ids = event['retrieved_turn_ids']
        else:
            raise ValueError(f'{qid}: unsupported event {mechanism}')

    recent_ids = set(exchanges) - discarded
    recent = [e for i, e in exchanges.items() if i in recent_ids]
    fixed = estimate_messages_tokens(build_messages(system, '', [], query))
    read_cap = max(0, min(cfg.memory_tokens, cfg.target_tokens - fixed)) if has_query_readout else 0
    results = []
    for policy, archive in stores.items():
        text, retrieved = bm25_retrieve(query, [exchanges[i] for i in archive], read_cap)
        messages = build_messages(system, text, recent, query)
        tokens = estimate_messages_tokens(messages)
        require(tokens <= cfg.input_tokens, f'{qid}: {policy} reader exceeded input budget')
        if policy == 'n7_unbounded':
            require(text == original_memory and retrieved == expected_ids and messages == request['messages'],
                    f'{qid}: offline N7 reader differs from logged final request')
        stored = len(evidence.intersection(recent_ids.union(archive)))
        visible = len(evidence.intersection(recent_ids.union(retrieved)))
        results.append(dict(question_id=qid, policy=policy, evidence_turns=len(evidence),
                            stored_evidence_turns=stored, visible_evidence_turns=visible,
                            archive_tokens_est=sum(cost[i] for i in archive), read_cap_tokens=read_cap,
                            final_input_tokens_est=tokens))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path, help='Existing final-only N7 batch directory')
    parser.add_argument('--archive-tokens', type=int, default=8192)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.archive_tokens > 0, 'Archive budget must be positive')
    runs = sorted((args.batch / 'runs').glob('*-direct-N7-r1'))
    require(bool(runs), 'No N7 runs in batch')
    source = [load(run / 'manifest.json') for run in runs]
    first = source[0]
    require(all(m['probe_schedule'] == 'final_only' and m['effective_budget'] == first['effective_budget']
                and m['dataset']['subset_manifest'] == first['dataset']['subset_manifest'] for m in source),
            'Mixed budgets, mixed subsets or non-final probes')
    subset = load(ROOT / first['dataset']['subset_manifest'])
    qids = {m['dataset']['question_id'] for m in source}
    require(qids == set(subset['question_ids']) and len(runs) == len(qids), 'Incomplete or duplicate source cohort')
    with (ROOT / subset['source_file']).open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
        require(all(m['dataset']['file_sha256'] == digest for m in source), 'Dataset hash differs from N7 batch')
        f.seek(0)
        raw = {item['question_id']: item for item in json.load(f) if item['question_id'] in qids}
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with (args.output / 'checkpoints.jsonl').open('w', encoding='utf-8') as checkpoints:
        for run, manifest in zip(runs, source):
            results.extend(replay_item(run, raw[manifest['dataset']['question_id']], args.archive_tokens, checkpoints))
    with (args.output / 'question-results.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = dict(source_batch=args.batch.name, subset=subset['manifest_id'], dataset_sha256=digest,
                   effective_budget=first['effective_budget'], bounded_archive_tokens=args.archive_tokens,
                   archive_token_accounting='sum of estimated separately serialized exchanges',
                   n7_final_requests_matched=len(runs), api_calls=0, policies={})
    for policy in POLICIES:
        rows = [r for r in results if r['policy'] == policy]
        eligible = [r for r in rows if r['evidence_turns']]
        def macro(key):
            return round(mean(r[key] / r['evidence_turns'] for r in eligible), 6) if eligible else None
        summary['policies'][policy] = dict(questions=len(rows), evidence_questions=len(eligible),
                                           archive_tokens_mean=round(mean(r['archive_tokens_est'] for r in rows), 2),
                                           stored_evidence_recall_macro=macro('stored_evidence_turns'),
                                           visible_evidence_recall_macro=macro('visible_evidence_turns'))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary['policies'], indent=2))


if __name__ == '__main__':
    main()
