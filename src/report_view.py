"""Offline report data preparation and safe HTML packaging; never calls a model."""
from __future__ import annotations
from html import escape
import json
import re
from collections import defaultdict
from pathlib import Path
from hashlib import sha256


def read_jsonl(path: Path, warnings: list[str]) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').split('\n'), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            warnings.append(f'{path.name} 第 {number} 行损坏，未能读取；原文件保留。')
    return rows


def source_turns(item: dict) -> list[dict]:
    if 'turns' in item:
        turns = [
            {
                'turn_id': int(t.get('turn_id', idx + 1)),
                'session_id': str(t.get('session_id', 's1')),
                'timestamp': str(t.get('timestamp', '')),
                'input': str(t.get('user_text', t.get('user', ''))),
                'output': t.get('assistant_text', t.get('assistant')),
                'source_message_indices': list(t.get('source_message_indices', [])),
                'kind': 'replay',
            }
            for idx, t in enumerate(item['turns'])
        ]
        turns.append({'turn_id': len(turns) + 1, 'session_id': 'final_question',
                      'timestamp': item.get('question_date'), 'input': item.get('question', ''),
                      'output': None, 'kind': 'final'})
        return turns
    turns = []
    index = 0
    for sid, date, messages in zip(item['haystack_session_ids'], item['haystack_dates'], item['haystack_sessions']):
        i = 0
        while i < len(messages):
            message = messages[i]
            indices = [index]
            index += 1
            user, assistant = '', None
            if message['role'] == 'user':
                user = message['content']
                if i + 1 < len(messages) and messages[i+1]['role'] == 'assistant':
                    assistant = messages[i+1]['content']
                    indices.append(index)
                    index += 1
                    i += 1
            else:
                assistant = message['content']
            turns.append({'turn_id': len(turns)+1, 'session_id': sid, 'timestamp': date,
                          'input': user, 'output': assistant, 'source_message_indices': indices,
                          'kind': 'replay'})
            i += 1
    turns.append({'turn_id': len(turns)+1, 'session_id': 'final_question', 'timestamp': item['question_date'],
                  'input': item['question'], 'output': None, 'kind': 'final'})
    return turns


from budget_executor import ARCHIVE_ONLY_STRATEGIES, BOUNDED_ARCHIVE_STRATEGIES, FACT_STRATEGIES

DETERMINISTIC_AUDIT_STRATEGIES = ("N1", "N2", "N3", "N5", *ARCHIVE_ONLY_STRATEGIES, *FACT_STRATEGIES)


def _norm_sent(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _split_sents(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[。！？.!?\n])", text or "") if p and p.strip()]


def _extract_tagged_sentences(text: str) -> list[tuple[int | None, str]]:
    """Parse [T<id> ...] lines into (turn_id, normalized_sentence) pairs."""
    items: list[tuple[int | None, str]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^\[T(\d+)(?:\s+\(.*?\))?\]\s*(?:User:|Assistant:)?\s*(.*)$", line)
        if m:
            tid = int(m.group(1))
            body = m.group(2)
        else:
            tid = None
            body = line
        m_fact = re.match(r"^([^|]+?)\s*\|\s*([^=]+?)\s*=\s*(.+)$", body)
        if m_fact:
            ent, attr, val = m_fact.group(1).strip(), m_fact.group(2).strip(), m_fact.group(3).strip()
            body = val if (ent == "context" and attr.startswith("s_")) else f"{ent} {attr} {val}"
        for s in _split_sents(body):
            ns = _norm_sent(s)
            if ns:
                items.append((tid, ns))
    return items


def _count_fact_false_additions(before_text: str, after_text: str) -> int:
    """Count false additions for N9/N10 fact memory without only matching the RHS value."""
    before_raw_lines = [_norm_sent(ln) for ln in (before_text or "").splitlines() if ln.strip()]
    before_sents = [s for _, s in _extract_tagged_sentences(before_text)]
    searchable_before = before_raw_lines + before_sents
    false_adds = 0
    for raw_line in (after_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^\[T(\d+)\]\s*([^|]+?)\s*\|\s*([^=]+?)\s*=\s*(.+)$", line)
        if not m:
            ns = _norm_sent(re.sub(r"^\[T\d+\]\s*", "", line))
            if ns and ns not in searchable_before:
                false_adds += 1
            continue
        ent, attr, val = m.group(2).strip(), m.group(3).strip(), m.group(4).strip()
        ent_n, val_n = _norm_sent(ent), _norm_sent(val)
        if ent == "context" and attr.startswith("s_"):
            if val_n and not any(val_n == bs or val_n in bs for bs in searchable_before):
                false_adds += 1
        elif val == "[RETRACTED]":
            if ent_n and ent_n != "user" and not any(ent_n in bs for bs in searchable_before):
                false_adds += 1
        else:
            # Must match BOTH entity (when not generic 'user') and value in the same source line/sentence
            matched = False
            for bs in searchable_before:
                if val_n in bs and (ent_n == "user" or ent_n in bs):
                    matched = True
                    break
            if not matched:
                false_adds += 1
    return false_adds


def audit_n1_n3_maintenance(
    strategy: str,
    status: str,
    events: list[dict],
    source_turns_list: list[dict] | None,
    answer_session_ids: list[str] | None,
    question_type: str | None,
) -> dict:
    """Compute deterministic (zero-API) maintenance quality metrics for N1, N2, and N3.

    Common checkpoint: final readout state (after all maintenance events, right before answering).
    Interrupted runs (overflow / maintenance_*_error / api_error) are marked non-evaluable (None)
    rather than 0.0 so execution failures are separated from evidence loss.
    """
    if strategy not in DETERMINISTIC_AUDIT_STRATEGIES:
        return {}
    answer_sids = set(answer_session_ids or [])
    replay_turns = [t for t in (source_turns_list or []) if t.get("kind") == "replay"]
    all_replay_tids = {int(t["turn_id"]) for t in replay_turns}
    turn_sents: dict[int, set[str]] = {}
    evidence_tids: set[int] = set()
    for t in replay_turns:
        tid = int(t["turn_id"])
        sents: set[str] = set()
        for field in (t.get("input") or "", t.get("output") or ""):
            for s in _split_sents(field):
                ns = _norm_sent(s)
                if ns:
                    sents.add(ns)
        turn_sents[tid] = sents
        if t.get("session_id") in answer_sids:
            evidence_tids.add(tid)

    if strategy in ("N1", "N5"):
        if status != "completed":
            return {
                "evidence_eval_status": "not_evaluable_interrupted",
                "final_evidence_retention": None,
                "all_evidence_retained": None,
                "total_fact_omissions": 0,
                "total_false_additions": 0,
                "final_stale_residues": 0,
            }
        if not evidence_tids:
            return {
                "evidence_eval_status": "not_applicable_no_evidence",
                "final_evidence_retention": None,
                "all_evidence_retained": None,
                "total_fact_omissions": 0,
                "total_false_additions": 0,
                "final_stale_residues": 0,
            }
        return {
            "evidence_eval_status": "evaluable",
            "final_evidence_retention": 1.0,
            "all_evidence_retained": True,
            "total_fact_omissions": 0,
            "total_false_additions": 0,
            "final_stale_residues": 0,
        }

    cum_evicted: set[int] = set()
    seen_lost_ev_tids: set[int] = set()
    seen_lost_ev_sents: set[tuple[int, str]] = set()
    last_stale = 0
    total_evidence_omissions = 0
    total_false_additions = 0
    last_after_by_tid: dict[int, set[str]] = {}
    last_retrieved_tids: set[int] = set()

    for ev in events:
        mech = ev.get("mechanism")
        if mech not in (
            "sliding_window_eviction",
            "extractive_compaction",
            "bm25_index_eviction",
            "bm25_bounded_eviction",
            "bm25_dedup_eviction",
            "bm25_session_balanced_eviction",
            "bm25_retrieval",
            "fact_append_compaction",
            "fact_update_compaction",
            "fact_hybrid_compaction",
            "fact_enhanced_compaction",
        ):
            continue
        tid_now = int(ev.get("turn_id", 0))
        prev_evicted = set(cum_evicted)
        for etid in ev.get("evicted_turn_ids") or []:
            cum_evicted.add(int(etid))
        if mech == "bm25_retrieval" or ev.get("trigger") == "query_readout":
            last_retrieved_tids = {int(x) for x in (ev.get("retrieved_turn_ids") or [])}
        # Replay turns up to current turn (final_answer turn is > max(all_replay_tids))
        ingested_replay = {t for t in all_replay_tids if t < tid_now}
        kept_tids = ingested_replay - cum_evicted

        before_items = _extract_tagged_sentences(ev.get("before_text") or "")
        after_items = _extract_tagged_sentences(ev.get("after_text") or "")
        before_set = {s for _, s in before_items}
        after_set = {s for _, s in after_items}

        if strategy in FACT_STRATEGIES:
            false_add_count = _count_fact_false_additions(ev.get("before_text") or "", ev.get("after_text") or "")
        else:
            false_add_count = len(after_set - before_set)
        total_false_additions += false_add_count

        after_by_tid: dict[int, set[str]] = defaultdict(set)
        for ptid, ps in after_items:
            if ptid is not None:
                after_by_tid[ptid].add(ps)
        last_after_by_tid = after_by_tid

        seen_ev_tids = {etid for etid in evidence_tids if etid < tid_now}
        ev_ret: float | None = None
        ev_turns_kept = 0
        newly_omitted_ev = 0

        if seen_ev_tids:
            if strategy in ("N2", *ARCHIVE_ONLY_STRATEGIES):
                retained = seen_ev_tids & (kept_tids | (last_retrieved_tids if mech == "bm25_retrieval" else set()))
                if strategy in BOUNDED_ARCHIVE_STRATEGIES and mech in ("bm25_bounded_eviction", "bm25_dedup_eviction", "bm25_session_balanced_eviction"):
                    arch_dropped = {int(x) for x in (ev.get("archive_evicted_turn_ids") or [])}
                    retained = retained - arch_dropped
                ev_turns_kept = len(retained)
                ev_ret = round(len(retained) / len(seen_ev_tids), 4)
                if mech == "bm25_retrieval":
                    newly_omitted_ev = 0
                else:
                    # Evidence turns evicted in this event that were not already marked lost
                    newly_lost = (seen_ev_tids - prev_evicted - retained) - seen_lost_ev_tids
                    newly_omitted_ev = len(newly_lost)
                    seen_lost_ev_tids.update(newly_lost)
            elif strategy in FACT_STRATEGIES:
                # Turn-level retention in kept_tids, structured fact lines [T<id>], or retrieved hybrid turn_ids
                retained_turns = {
                    etid for etid in seen_ev_tids
                    if etid in kept_tids or bool(after_by_tid.get(etid)) or (ev.get("trigger") == "query_readout" and etid in last_retrieved_tids)
                }
                ev_turns_kept = len(retained_turns)
                ev_ret = round(len(retained_turns) / len(seen_ev_tids), 4)
                newly_lost = (seen_ev_tids - prev_evicted - retained_turns) - seen_lost_ev_tids
                newly_omitted_ev = len(newly_lost)
                seen_lost_ev_tids.update(newly_lost)
            else:
                total_ev_sents = 0
                kept_ev_sents = 0
                for etid in seen_ev_tids:
                    tsents = turn_sents.get(etid, set())
                    total_ev_sents += len(tsents)
                    if etid in kept_tids:
                        kept_ev_sents += len(tsents)
                        if tsents:
                            ev_turns_kept += 1
                    else:
                        matched = tsents & after_by_tid.get(etid, set())
                        kept_ev_sents += len(matched)
                        if matched:
                            ev_turns_kept += 1
                        missing = tsents - matched
                        for ms in missing:
                            key = (etid, ms)
                            if key not in seen_lost_ev_sents:
                                seen_lost_ev_sents.add(key)
                                newly_omitted_ev += 1
                ev_ret = round(kept_ev_sents / total_ev_sents, 4) if total_ev_sents else 1.0

        total_evidence_omissions += newly_omitted_ev

        stale_count = 0
        if question_type == "knowledge-update" and len(seen_ev_tids) >= 2:
            latest_ev_tid = max(seen_ev_tids)
            for etid in seen_ev_tids:
                if etid < latest_ev_tid and (etid in kept_tids or bool(after_by_tid.get(etid))):
                    stale_count += 1

        ev["audit"] = {
            "evidence_retention": ev_ret,
            "evidence_turns_retained": ev_turns_kept if seen_ev_tids else None,
            "evidence_turns_total": len(seen_ev_tids) if seen_ev_tids else None,
            "fact_omission_count": newly_omitted_ev,
            "false_addition_count": false_add_count,
            "stale_fact_residue_count": stale_count,
        }
        last_stale = stale_count

    # Common checkpoint evaluation: right before final_answer (only for completed runs with evidence)
    if status != "completed":
        return {
            "evidence_eval_status": "not_evaluable_interrupted",
            "final_evidence_retention": None,
            "all_evidence_retained": None,
            "total_fact_omissions": total_evidence_omissions,
            "total_false_additions": total_false_additions,
            "final_stale_residues": last_stale,
        }
    if not evidence_tids:
        return {
            "evidence_eval_status": "not_applicable_no_evidence",
            "final_evidence_retention": None,
            "all_evidence_retained": None,
            "total_fact_omissions": total_evidence_omissions,
            "total_false_additions": total_false_additions,
            "final_stale_residues": last_stale,
        }

    final_kept_tids = all_replay_tids - cum_evicted
    if strategy == "N2":
        retained = evidence_tids & final_kept_tids
        final_ev_ret = round(len(retained) / len(evidence_tids), 4)
        all_retained = evidence_tids.issubset(final_kept_tids)
    elif strategy in ARCHIVE_ONLY_STRATEGIES:
        final_visible = final_kept_tids | last_retrieved_tids
        retained = evidence_tids & final_visible
        final_ev_ret = round(len(retained) / len(evidence_tids), 4)
        all_retained = evidence_tids.issubset(final_visible)
    elif strategy in FACT_STRATEGIES:
        retained = {etid for etid in evidence_tids if etid in final_kept_tids or etid in last_retrieved_tids or bool(last_after_by_tid.get(etid))}
        final_ev_ret = round(len(retained) / len(evidence_tids), 4)
        all_retained = evidence_tids.issubset(retained)
    else:
        total_ev_sents = 0
        kept_ev_sents = 0
        for etid in evidence_tids:
            tsents = turn_sents.get(etid, set())
            total_ev_sents += len(tsents)
            if etid in final_kept_tids:
                kept_ev_sents += len(tsents)
            else:
                kept_ev_sents += len(tsents & last_after_by_tid.get(etid, set()))
        final_ev_ret = round(kept_ev_sents / total_ev_sents, 4) if total_ev_sents else 1.0
        all_retained = kept_ev_sents == total_ev_sents

    return {
        "evidence_eval_status": "evaluable",
        "final_evidence_retention": final_ev_ret,
        "all_evidence_retained": all_retained,
        "total_fact_omissions": total_evidence_omissions,
        "total_false_additions": total_false_additions,
        "final_stale_residues": last_stale,
    }


def apply_n4_audit_results(
    status: str,
    events: list[dict],
    audit_rows: list[dict],
    source_turns_list: list[dict] | None,
    answer_session_ids: list[str] | None,
) -> dict:
    """Merge cached LLM semantic audit rows (audit_N4.jsonl) into N4 events and summary."""
    answer_sids = set(answer_session_ids or [])
    replay_turns = [t for t in (source_turns_list or []) if t.get("kind") == "replay"]
    evidence_tids = {int(t["turn_id"]) for t in replay_turns if t.get("session_id") in answer_sids}

    by_key = {(r.get("event_id") or r.get("checkpoint")): r for r in audit_rows}
    total_omissions = 0
    total_false_additions = 0
    last_stale = 0

    for ev in events:
        if ev.get("mechanism") != "llm_compaction" or ev.get("status") != "ok":
            continue
        row = by_key.get(ev.get("event_id"))
        if not row:
            continue
        om = int(row.get("fact_omission_count") or 0)
        fa = int(row.get("false_addition_count") or 0)
        st = int(row.get("stale_fact_residue_count") or 0)
        total_omissions += om
        total_false_additions += fa
        last_stale = st
        ev["audit"] = {
            "evidence_retention": row.get("evidence_retention"),
            "evidence_turns_retained": row.get("evidence_turns_retained"),
            "evidence_turns_total": row.get("evidence_turns_total"),
            "fact_omission_count": om,
            "false_addition_count": fa,
            "stale_fact_residue_count": st,
            "audit_mode": "llm_judge",
        }

    if status != "completed":
        return {
            "evidence_eval_status": "not_evaluable_interrupted",
            "final_evidence_retention": None,
            "all_evidence_retained": None,
            "total_fact_omissions": total_omissions if audit_rows else None,
            "total_false_additions": total_false_additions if audit_rows else None,
            "final_stale_residues": last_stale if audit_rows else None,
        }
    if not evidence_tids:
        return {
            "evidence_eval_status": "not_applicable_no_evidence",
            "final_evidence_retention": None,
            "all_evidence_retained": None,
            "total_fact_omissions": total_omissions if audit_rows else None,
            "total_false_additions": total_false_additions if audit_rows else None,
            "final_stale_residues": last_stale if audit_rows else None,
        }

    final_row = by_key.get("final")
    if not final_row:
        return {}
    return {
        "evidence_eval_status": "evaluable",
        "final_evidence_retention": final_row.get("evidence_retention"),
        "all_evidence_retained": final_row.get("all_evidence_retained"),
        "total_fact_omissions": total_omissions,
        "total_false_additions": total_false_additions,
        "final_stale_residues": last_stale,
    }


def enrich_strategy(
    run_dir: Path,
    info: dict,
    strategy: str | None = None,
    source_turns_list: list[dict] | None = None,
    answer_session_ids: list[str] | None = None,
    question_type: str | None = None,
    n4_audit_rows: list[dict] | None = None,
) -> dict:
    warnings = []
    turns = read_jsonl(run_dir / 'turns.jsonl', warnings)
    events = read_jsonl(run_dir / 'events.jsonl', warnings)
    probes = read_jsonl(run_dir / 'probes.jsonl', warnings)
    audit_summary = {}
    if strategy in DETERMINISTIC_AUDIT_STRATEGIES:
        audit_summary = audit_n1_n3_maintenance(
            strategy=strategy,
            status=info.get("status", ""),
            events=events,
            source_turns_list=source_turns_list,
            answer_session_ids=answer_session_ids,
            question_type=question_type,
        )
    elif strategy == "N4" and n4_audit_rows:
        audit_summary = apply_n4_audit_results(
            status=info.get("status", ""),
            events=events,
            audit_rows=n4_audit_rows,
            source_turns_list=source_turns_list,
            answer_session_ids=answer_session_ids,
        )
    request = None
    path = run_dir / 'requests/final_answer.json'
    if path.exists():
        try:
            request = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            warnings.append('final_answer.json 损坏，实际请求未能读取。')
    return {**info, **audit_summary, 'turns': turns, 'events': events, 'probes': probes, 'final_request': request, 'warnings': warnings}


def safe_json(data) -> str:
    # Prevent data from ending a script element or injecting markup.
    return json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def build_html_report(batch_id, dataset_meta, overview, comparisons, rows, timelines, judge_provider, judge_model, detail_dir: Path | None = None) -> str:
    questions = []
    inline = {}
    if detail_dir:
        detail_dir.mkdir(parents=True, exist_ok=True)
    for qid, detail in timelines.items():
        filename = sha256(qid.encode()).hexdigest()[:24] + '.js'
        if detail_dir:
            (detail_dir / filename).write_text(f'window.REPORT_DETAILS[{safe_json(qid)}]={safe_json(detail)};\n', encoding='utf-8')
        else:
            inline[qid] = detail
        questions.append({'question_id': qid, 'question_type': detail['question_type'],
                          'is_abstention': detail['is_abstention'], 'question': detail['question'],
                          'file': f'details/{filename}' if detail_dir else None,
                          'strategies': {s: {k: v for k, v in info.items() if k in ('status', 'judge_label', 'maintenance_events', 'generation_provider', 'generation_model')}
                                         for s, info in detail['strategies'].items()}})
    payload = {'batch_id': batch_id, 'dataset': dataset_meta, 'overview': overview,
               'comparisons': comparisons, 'questions': questions,
               'judge_provider': judge_provider, 'judge_model': judge_model}
    template = (Path(__file__).parent / 'report_template.html').read_text(encoding='utf-8')
    comparison_path = detail_dir.parent / 'answer-model-comparison.json' if detail_dir else None
    if comparison_path and comparison_path.exists():
        comparison = json.loads(comparison_path.read_text())
        if comparison['source_batch'] == detail_dir.parent.name:
            groups = defaultdict(list)
            for row in comparison['rows']:
                groups[(row['provider'], row['condition'])].append(row)
            parts = ['<section class="panel"><h2>更新题 · 回答模型对照</h2>',
                     '<p>固定 N9 原始请求，或在最终问题前按时间补回带标注的原始消息。'
                     '每个条件均使用 ECNU 评分。四题来自开发子集，补入证据的位置也可能影响结果。</p>']
            for (provider, condition), rows in groups.items():
                correct = sum(r['judgment']['judge_label'] is True for r in rows)
                label = '原始请求' if condition == 'original' else '补回证据'
                parts.append(f'<details><summary>{escape(provider)} · {label} · {correct}/{len(rows)}</summary>')
                for r in rows:
                    verdict = '正确' if r['judgment']['judge_label'] else '错误'
                    parts.append(f"<p><strong>{escape(r['question_id'])} · {verdict}</strong> "
                                 f"{escape(r['question'])}<br>参考答案：{escape(str(r['expected']))}"
                                 f"<br>回答：{escape(r['answer'])}</p>")
                parts.append('</details>')
            parts.append('</section>')
            template = template.replace('</main>', ''.join(parts) + '</main>')
    evidence_links = ""
    if detail_dir and (detail_dir.parent / "evidence_trajectory.md").is_file():
        evidence_links = ' <a href="evidence_trajectory.md">evidence_trajectory.md</a> · <a href="evidence_trajectory.json">evidence_trajectory.json</a>'
    template = template.replace("__EVIDENCE_TRAJECTORY_LINK__", evidence_links)
    return template.replace('__REPORT_PAYLOAD__', safe_json(payload)).replace('__REPORT_DETAILS__', safe_json(inline))
