"""Deterministic budget executor and N0-N6 maintenance strategies."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any

from llm_client import LLMClient
from metrics import (
    duplicate_ngram_rate,
    estimate_messages_tokens,
    estimate_text_tokens,
    lexical_jaccard,
    lexical_retention,
    normalize_text,
    sha256_text,
)

EXECUTOR_VERSION = "budget-executor-v5-n0-n14"
LLM_STRATEGIES = ("N4", "N6")
ARCHIVE_ONLY_STRATEGIES = ("N7", "N8", "N11", "N12")
ARCHIVE_STRATEGIES = ("N7", "N8", "N11", "N12", "N13")
BOUNDED_ARCHIVE_STRATEGIES = ("N8", "N11", "N12", "N13")
ARCHIVE_POLICY_BY_STRATEGY = {
    "N8": "fifo",
    "N11": "dedup_fifo",
    "N12": "session_balanced",
    "N13": "fifo",
}
FACT_STRATEGIES = ("N9", "N10", "N13", "N14")
MEMORY_STRATEGIES = ("N3", *FACT_STRATEGIES, *LLM_STRATEGIES)
FULL_HISTORY_STRATEGIES = ("N1", "N5")
STRATEGIES = ("N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11", "N12", "N13", "N14")
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "compaction-v1.md"


def load_compaction_prompt_template() -> tuple[str, str]:
    raw = PROMPT_PATH.read_text(encoding="utf-8")
    # Use the prompt section between the header and "## runner 约定"
    lines = raw.splitlines()
    body_lines: list[str] = []
    started = False
    for line in lines:
        if line.startswith("## runner"):
            break
        if line.startswith("你负责维护对话记忆"):
            started = True
        if started:
            body_lines.append(line)
    template = "\n".join(body_lines).strip()
    return template, sha256_text(template)


@dataclass
class Exchange:
    turn_id: int
    session_id: str
    user_text: str
    assistant_text: str | None = None
    timestamp: str | None = None
    source_message_indices: list[int] = field(default_factory=list)


@dataclass
class BudgetConfig:
    input_tokens: int = 8192
    max_output_tokens: int = 512
    memory_tokens: int = 2048
    recent_history_tokens: int = 4096
    n1_context_tokens: int | None = None
    n4_output_tokens: int = 4096
    n4_target_ratio: float = 0.65
    n4_retry_ratio: float = 0.40
    chunk_tokens: int = 2048
    n8_archive_tokens: int = 16384
    trigger_ratio: float = 0.8
    target_ratio: float = 0.6

    @property
    def trigger_tokens(self) -> int:
        return math.floor(self.trigger_ratio * self.input_tokens)

    @property
    def target_tokens(self) -> int:
        return math.floor(self.target_ratio * self.input_tokens)


def build_messages(
    system_prompt: str,
    memory_text: str,
    recent_exchanges: list[Exchange],
    current_user_text: str,
) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    sys_content = system_prompt
    if memory_text.strip():
        sys_content = f"{system_prompt}\n\n[显式记忆 M]\n{memory_text.strip()}"
    msgs.append({"role": "system", "content": sys_content})
    for ex in recent_exchanges:
        prefix = f"[{ex.timestamp}] " if ex.timestamp else ""
        msgs.append({"role": "user", "content": f"{prefix}{ex.user_text}"})
        if ex.assistant_text is not None:
            msgs.append({"role": "assistant", "content": ex.assistant_text})
    msgs.append({"role": "user", "content": current_user_text})
    return msgs


def serialize_exchanges(exchanges: list[Exchange]) -> str:
    parts: list[str] = []
    for ex in exchanges:
        ts = f" ({ex.timestamp})" if ex.timestamp else ""
        parts.append(f"[T{ex.turn_id}{ts}] User: {ex.user_text}")
        if ex.assistant_text is not None:
            parts.append(f"[T{ex.turn_id}{ts}] Assistant: {ex.assistant_text}")
    return "\n".join(parts)


def chunk_exchanges(exchanges: list[Exchange], limit: int) -> list[list[Exchange]]:
    """Group contiguous exchanges by serialized size. Keep oversized exchanges intact."""
    chunks: list[list[Exchange]] = []
    current: list[Exchange] = []
    # Count the newline between exchanges and preserve role and source markers.
    for exchange in exchanges:
        candidate = current + [exchange]
        if current and estimate_text_tokens(serialize_exchanges(candidate)) > limit:
            chunks.append(current)
            candidate = [exchange]
        current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_sentences(text: str) -> list[str]:
    raw_parts = re.split(r"(?<=[。！？.!?\n])", text)
    return [p.strip() for p in raw_parts if p and p.strip()]


def extractive_compact(old_memory: str, evicted: list[Exchange], max_tokens: int) -> str:
    """N3: deterministic sentence splitting, exact-normalized dedup (keep latest), newest-first fit, chronological output."""
    if max_tokens <= 0:
        return ""

    entries: list[tuple[int, int, str]] = []
    seq = 0
    # Parse existing M lines; preserve [T<id>] tag if present
    for line in old_memory.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\[T(\d+)\]\s*(.+)$", line)
        tid = int(m.group(1)) if m else 0
        body = m.group(2) if m else line
        for s in split_sentences(body):
            entries.append((tid, seq, s))
            seq += 1

    for ex in evicted:
        for s in split_sentences(ex.user_text):
            entries.append((ex.turn_id, seq, f"User: {s}"))
            seq += 1
        if ex.assistant_text:
            for s in split_sentences(ex.assistant_text):
                entries.append((ex.turn_id, seq, f"Assistant: {s}"))
                seq += 1

    # Deduplicate by normalized text, keeping latest (highest turn_id, seq)
    dedup: dict[str, tuple[int, int, str]] = {}
    for tid, s_idx, sent in entries:
        norm = normalize_text(sent)
        if not norm:
            continue
        dedup[norm] = (tid, s_idx, sent)

    # Select from newest to oldest within max_tokens
    newest_first = sorted(dedup.values(), key=lambda x: (x[0], x[1]), reverse=True)
    selected: list[tuple[int, int, str]] = []
    used = 0
    for item in newest_first:
        line = f"[T{item[0]}] {item[2]}"
        t = estimate_text_tokens(line)
        if used + t <= max_tokens:
            selected.append(item)
            used += t

    # Output in chronological order
    selected.sort(key=lambda x: (x[0], x[1]))
    return "\n".join(f"[T{tid}] {sent}" for tid, _, sent in selected)


def _bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", (text or "").lower())


def bm25_retrieve(
    query: str,
    archived: list[Exchange],
    max_tokens: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[str, list[int]]:
    """N7: Okapi BM25 retrieval over evicted exchanges within max_tokens, output chronologically."""
    if max_tokens <= 0 or not archived:
        return "", []
    q_tokens = _bm25_tokenize(query)
    docs = [_bm25_tokenize(f"{ex.user_text}\n{ex.assistant_text or ''}") for ex in archived]
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, n_docs)
    df: Counter[str] = Counter()
    for d in docs:
        for tok in set(d):
            df[tok] += 1

    scored: list[tuple[float, int, Exchange]] = []
    for ex, d in zip(archived, docs):
        if not d:
            scored.append((0.0, ex.turn_id, ex))
            continue
        tf = Counter(d)
        dl = len(d)
        score = 0.0
        for qt in q_tokens:
            if qt not in tf:
                continue
            idf = math.log((n_docs - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)
            freq = tf[qt]
            denom = freq + k1 * (1.0 - b + b * dl / max(1.0, avgdl))
            score += idf * (freq * (k1 + 1.0)) / denom
        scored.append((score, ex.turn_id, ex))

    # Rank highest BM25 score first (tie-break newer turns first)
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected: list[Exchange] = []
    used = 0
    for score, _, ex in scored:
        if score <= 0 and selected:
            break
        block = serialize_exchanges([ex])
        tok = estimate_text_tokens(block)
        if used + tok <= max_tokens:
            selected.append(ex)
            used += tok
    selected.sort(key=lambda ex: ex.turn_id)
    return serialize_exchanges(selected), [ex.turn_id for ex in selected]


def _archive_dedup_key(ex: Exchange) -> str:
    return normalize_text(f"{ex.user_text}\n{ex.assistant_text or ''}")


def bound_archive_fifo(archived: list[Exchange], max_tokens: int) -> tuple[list[Exchange], list[Exchange]]:
    """N8/N13 FIFO: keep newest exchanges within archive token limit, evicting oldest first."""
    return bound_archive(archived, max_tokens, policy="fifo")


def bound_archive(
    archived: list[Exchange],
    max_tokens: int,
    policy: str = "fifo",
) -> tuple[list[Exchange], list[Exchange]]:
    """Bounded archive eviction supporting fifo (N8/N13), dedup_fifo (N11), and session_balanced (N12)."""
    if max_tokens <= 0:
        return [], list(archived)
    cost = {ex.turn_id: estimate_text_tokens(serialize_exchanges([ex])) for ex in archived}
    kept: list[Exchange] = []
    dropped: list[Exchange] = []
    for ex in archived:
        if cost[ex.turn_id] > max_tokens:
            dropped.append(ex)
            continue
        if policy == "dedup_fifo":
            key = _archive_dedup_key(ex)
            if key:
                new_kept: list[Exchange] = []
                for old in kept:
                    if _archive_dedup_key(old) == key:
                        dropped.append(old)
                    else:
                        new_kept.append(old)
                kept = new_kept
        kept.append(ex)
        total = sum(cost[x.turn_id] for x in kept)
        while total > max_tokens and kept:
            if policy == "session_balanced":
                by_session: Counter[str] = Counter()
                for x in kept:
                    by_session[x.session_id] += cost[x.turn_id]
                victim = min(kept, key=lambda x: (-by_session[x.session_id], x.turn_id))
            else:
                victim = kept[0]
            kept.remove(victim)
            dropped.append(victim)
            total -= cost[victim.turn_id]
    return kept, dropped


def _extract_fact_ops(ex: Exchange) -> list[tuple[str, str, str, str]]:
    """Extract (op, entity, attr, value) tuples from user turns using deterministic rules."""
    facts: list[tuple[str, str, str, str]] = []
    text = ex.user_text or ""
    idx = 0
    for sent in split_sentences(text):
        s_raw = sent.strip()
        if not s_raw:
            continue
        # Detect optional sentence-level entity scope, e.g. "更新杉木项目：..." or "请记住：..."
        current_entity: str | None = None
        s_clean = re.sub(r"^(?:请记住|请更新|请记录|记住|补充|注意)[：:\s]*", "", s_raw).strip()
        m_scope = re.match(r"^(?:更新|修改|关于)?([^\s：:，,。]{1,20})[：:](.+)$", s_clean)
        if m_scope and not re.search(r"(?:是|为|=)", m_scope.group(1)):
            current_entity = m_scope.group(1).strip()
            s_clean = m_scope.group(2).strip()

        clauses = [c.strip() for c in re.split(r"[，,；;]+", s_clean) if c and c.strip()]
        for clause in clauses:
            idx += 1
            m_ret = re.match(r"^(?:撤销|取消|删除|作废)\s*([^\s的]{1,20})的([^\s，。,.]{1,20})", clause)
            if m_ret:
                current_entity = m_ret.group(1).strip()
                facts.append(("retract", current_entity, m_ret.group(2).strip(), clause))
                continue
            if current_entity:
                m_ret_scoped = re.match(r"^(?:撤销|取消|删除|作废)\s*([^\s，。,.]{1,20})$", clause)
                if m_ret_scoped:
                    facts.append(("retract", current_entity, m_ret_scoped.group(1).strip(), clause))
                    continue
            m_upd = re.match(r"^([^\s的]{1,20})的([^\s，。,.]{1,20})(?:改为|更新为|变更为|调整为|改成了|换成了)(.+)$", clause)
            if not m_upd:
                m_upd = re.match(r"^([^\s的]{1,12}?(?:项目|系统|模块|计划|任务|团队|产品|方案|服务|部门|公司))([^\s，。,.]{1,15})(?:改为|更新为|变更为|调整为|改成了|换成了)(.+)$", clause)
            if m_upd:
                current_entity = m_upd.group(1).strip()
                facts.append(("update", current_entity, m_upd.group(2).strip(), m_upd.group(3).strip("。.")))
                continue
            if current_entity:
                m_upd_scoped = re.match(r"^([^\s，。,.]{1,20})(?:改为|更新为|变更为|调整为|改成了|换成了)(.+)$", clause)
                if m_upd_scoped:
                    facts.append(("update", current_entity, m_upd_scoped.group(1).strip(), m_upd_scoped.group(2).strip("。.")))
                    continue
            m_ass = re.match(r"^([^\s的]{1,20})的([^\s是：:=]{1,20})(?:是|为|：|:|=)(.+)$", clause)
            if not m_ass:
                m_ass = re.match(r"^([^\s的]{1,12}?(?:项目|系统|模块|计划|任务|团队|产品|方案|服务|部门|公司))([^\s是：:=]{1,15})(?:是|为|：|:|=)(.+)$", clause)
            if m_ass:
                current_entity = m_ass.group(1).strip()
                facts.append(("assert", current_entity, m_ass.group(2).strip(), m_ass.group(3).strip("。.")))
                continue
            if current_entity:
                m_ass_scoped = re.match(r"^([^\s是为：:=]{1,20}?)(?:是|为|：|:|=)(.+)$", clause)
                if m_ass_scoped and not re.search(r"(?:不再|独立|不变|介绍)", clause):
                    facts.append(("assert", current_entity, m_ass_scoped.group(1).strip(), m_ass_scoped.group(2).strip("。.")))
                    continue
            m_en_ret = re.search(r"\b(?:no longer|cancelled|canceled|retract|stopped|quit)\b\s+(?:my\s+)?([a-z0-9_ ]{2,30})", clause, re.I)
            if m_en_ret:
                attr = "_".join(m_en_ret.group(1).lower().split()[:3])
                facts.append(("retract", "user", attr, clause))
                continue
            m_en_upd = re.search(r"\bmy\s+([a-z0-9_ ]{2,25}?)\s+(?:changed to|updated to|is now|are now|became)\s+(.+)", clause, re.I)
            if m_en_upd:
                attr = "_".join(m_en_upd.group(1).lower().split())
                facts.append(("update", "user", attr, m_en_upd.group(2).strip(".")))
                continue
            m_en_ass = re.search(r"\bmy\s+([a-z0-9_ ]{2,25}?)\s+(?:is|are)\s+(.+)", clause, re.I)
            if m_en_ass:
                attr = "_".join(m_en_ass.group(1).lower().split())
                facts.append(("assert", "user", attr, m_en_ass.group(2).strip(".")))
                continue
            facts.append(("assert", "context", f"s_{ex.turn_id}_{idx}", clause))
    return facts


def _extract_fact_ops_enhanced(ex: Exchange) -> list[tuple[str, str, str, str]]:
    """N14: Enhanced deterministic rule extractor adding natural-language English/Chinese state updates & retractions."""
    base_ops = _extract_fact_ops(ex)
    enhanced: list[tuple[str, str, str, str]] = []
    for op, ent, attr, val in base_ops:
        if not (ent == "context" and attr.startswith("s_")):
            enhanced.append((op, ent, attr, val))
            continue
        clause = val.strip()
        # 1. Explicit English first-person retraction / stopping / selling / giving up
        m_ret = re.search(
            r"\bi\s+(?:am\s+)?(?:no longer|stopped|quit|gave up|dropped|cancelled|canceled|sold|unsubscribed from)\s+([a-z0-9_' -]{2,35})",
            clause,
            re.I,
        )
        if m_ret:
            key = "_".join(re.findall(r"[a-z0-9]+", m_ret.group(1).lower())[:3])
            if key:
                enhanced.append(("retract", "user", key, clause))
                continue
        # 2. Explicit English first-person switch / move / upgrade / change
        m_sw = re.search(
            r"\bi\s+(?:just\s+|recently\s+)?(?:switched|moved|upgraded|changed|transitioned)\s+(?:(?:my\s+)?([a-z0-9_ ]{2,20}?)\s+)?(?:from\s+[^,.;]+?\s+)?to\s+([a-z0-9_' -]{2,40})",
            clause,
            re.I,
        )
        if m_sw:
            subj = m_sw.group(1) or "preference"
            key = "_".join(re.findall(r"[a-z0-9]+", subj.lower())[:3]) or "state"
            enhanced.append(("update", "user", key, m_sw.group(2).strip(".,;!? ")))
            continue
        # 3. Quantity / count / progress updates ("I've added 25 new postcards", "I've tried out 3 of Emma's recipes", "I own 4 bikes")
        m_cnt = re.search(
            r"\bi(?:'ve| have)?\s+(?:just\s+|now\s+|currently\s+|already\s+|recently\s+)?(?:added|tried(?:\s+out)?|completed|finished|visited|read|watched|bought|collected|own|have)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:of\s+)?(?:my\s+|new\s+)?([a-z0-9_' -]{2,35})",
            clause,
            re.I,
        )
        if m_cnt:
            nouns = [w for w in re.findall(r"[a-z0-9]+", m_cnt.group(2).lower()) if w not in ("the", "a", "an", "to", "in", "on", "of", "for", "so", "far", "them")]
            key = "_".join(nouns[:3]) or "count"
            enhanced.append(("update", "user", key, f"{m_cnt.group(1)} ({clause})"))
            continue
        # 4. Personal record / goal / metric updates
        m_rec = re.search(
            r"\b(?:my\s+)?(?:new\s+)?(personal best|pb|record|goal|target|rank|score|weight|budget|salary|commute|rent|schedule|favorite|preferred)\b[^.!?]{0,40}?\b(?:is now|was|is|dropped to|improved to|reached|hit|at)\s+(.+)",
            clause,
            re.I,
        )
        if m_rec:
            key = "_".join(re.findall(r"[a-z0-9]+", m_rec.group(1).lower())[:3])
            enhanced.append(("update", "user", key, m_rec.group(2).strip(".,;!? ")))
            continue
        enhanced.append((op, ent, attr, val))
    return enhanced


def fact_memory_compact_with_stats(
    old_memory: str,
    evicted: list[Exchange],
    max_tokens: int,
    allow_update: bool = False,
    store_fallback: bool = True,
    enhanced_extractor: bool = False,
) -> tuple[str, dict[str, Any]]:
    """N9 (append-only) / N10 (update + [RETRACTED]) / N13 (fact-only + archive) / N14 (enhanced extractor) compaction."""
    if max_tokens <= 0:
        return "", {"effective_m_budget": max_tokens, "used_tokens": 0}
    entries: list[tuple[int, int, str, str, str]] = []
    seq = 0
    for line in (old_memory or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\[T(\d+)\]\s*([^|]+?)\s*\|\s*([^=]+?)\s*=\s*(.+)$", line)
        if m:
            entries.append((int(m.group(1)), seq, m.group(2).strip(), m.group(3).strip(), m.group(4).strip()))
        else:
            m2 = re.match(r"^\[T(\d+)\]\s*(.+)$", line)
            tid = int(m2.group(1)) if m2 else 0
            body = m2.group(2) if m2 else line
            entries.append((tid, seq, "context", f"s_{tid}_{seq}", body))
        seq += 1

    ops_counts = {"assert": 0, "update": 0, "retract": 0, "fallback": 0}
    key_hits_existing = 0
    value_changes = 0
    retract_hits_active = 0

    extractor = _extract_fact_ops_enhanced if enhanced_extractor else _extract_fact_ops
    for ex in evicted:
        for op, ent, attr, val in extractor(ex):
            is_fallback = attr.startswith("s_") and ent == "context"
            if is_fallback:
                ops_counts["fallback"] += 1
                if not store_fallback:
                    continue
            else:
                ops_counts[op] += 1
                ent_l, attr_l = ent.lower(), attr.lower()
                matching = [e for e in entries if e[2].lower() == ent_l and e[3].lower() == attr_l]
                if matching:
                    key_hits_existing += 1
                    latest_val = matching[-1][4]
                    if latest_val != val:
                        value_changes += 1
                    if op == "retract" and latest_val != "[RETRACTED]":
                        retract_hits_active += 1
                if allow_update:
                    entries = [e for e in entries if not (e[2].lower() == ent_l and e[3].lower() == attr_l)]
            store_val = "[RETRACTED]" if (allow_update and op == "retract") else val
            entries.append((ex.turn_id, seq, ent, attr, store_val))
            seq += 1

    newest_first = sorted(entries, key=lambda x: (x[0], x[1]), reverse=True)
    selected: list[tuple[int, int, str, str, str]] = []
    used = 0
    for item in newest_first:
        line = f"[T{item[0]}] {item[2]} | {item[3]} = {item[4]}"
        t = estimate_text_tokens(line)
        if used + t <= max_tokens:
            selected.append(item)
            used += t
    selected.sort(key=lambda x: (x[0], x[1]))

    # Compute M-slot occupancy breakdown (duplicate keys, retracted markers, fallback context)
    latest_seq_by_key: dict[tuple[str, str], int] = {}
    for _, s_idx, ent, attr, _ in selected:
        if not (ent == "context" and attr.startswith("s_")):
            latest_seq_by_key[(ent.lower(), attr.lower())] = s_idx

    dup_entries = dup_tokens = ret_entries = ret_tokens = fb_entries = fb_tokens = 0
    for tid, s_idx, ent, attr, val in selected:
        tok = estimate_text_tokens(f"[T{tid}] {ent} | {attr} = {val}")
        if ent == "context" and attr.startswith("s_"):
            fb_entries += 1
            fb_tokens += tok
        else:
            if val == "[RETRACTED]":
                ret_entries += 1
                ret_tokens += tok
            if latest_seq_by_key.get((ent.lower(), attr.lower())) != s_idx:
                dup_entries += 1
                dup_tokens += tok

    stats = {
        "effective_m_budget": max_tokens,
        "used_tokens": used,
        "total_entries": len(selected),
        "duplicate_key_entries": dup_entries,
        "duplicate_key_tokens": dup_tokens,
        "retracted_entries": ret_entries,
        "retracted_tokens": ret_tokens,
        "context_fallback_entries": fb_entries,
        "context_fallback_tokens": fb_tokens,
        "ops_counts": ops_counts,
        "key_hits_existing": key_hits_existing,
        "value_changes": value_changes,
        "retract_hits_active": retract_hits_active,
    }
    text = "\n".join(f"[T{tid}] {ent} | {attr} = {val}" for tid, _, ent, attr, val in selected)
    return text, stats


def fact_memory_compact(
    old_memory: str,
    evicted: list[Exchange],
    max_tokens: int,
    allow_update: bool = False,
) -> str:
    """N9 (append-only) / N10 (update + explicit [RETRACTED]) deterministic fact memory compaction."""
    text, _ = fact_memory_compact_with_stats(old_memory, evicted, max_tokens, allow_update=allow_update)
    return text


def fact_memory_retrieve(query: str, memory_text: str, max_tokens: int) -> tuple[str, list[int]]:
    """Shared N9/N10 query readout: rank stored fact lines by query overlap, output chronologically."""
    if max_tokens <= 0 or not (memory_text or "").strip():
        return "", []
    q_tokens = set(_bm25_tokenize(query))
    lines = [l.strip() for l in memory_text.splitlines() if l.strip()]
    scored: list[tuple[int, int, str, int]] = []
    for idx, line in enumerate(lines):
        m = re.match(r"^\[T(\d+)\]", line)
        tid = int(m.group(1)) if m else 0
        ltoks = set(_bm25_tokenize(line))
        overlap = len(q_tokens & ltoks)
        scored.append((overlap, tid, line, idx))
    scored.sort(key=lambda x: (x[0], x[1], x[3]), reverse=True)
    selected: list[tuple[int, int, str]] = []
    used = 0
    for _, tid, line, idx in scored:
        t = estimate_text_tokens(line)
        if used + t <= max_tokens:
            selected.append((tid, idx, line))
            used += t
    selected.sort(key=lambda x: (x[0], x[1]))
    ret_tids = sorted({tid for tid, _, _ in selected if tid > 0})
    return "\n".join(line for _, _, line in selected), ret_tids


def build_readout_memory(
    strategy: str,
    query: str,
    memory_text: str,
    archived_exchanges: list[Exchange] | None,
    max_tokens: int,
) -> tuple[str, list[int]]:
    """Shared readout function used by both online runner and offline snapshot replay."""
    if max_tokens <= 0:
        return "", []
    if strategy in ARCHIVE_ONLY_STRATEGIES:
        return bm25_retrieve(query, archived_exchanges or [], max_tokens)
    if strategy in ("N9", "N10", "N14"):
        return fact_memory_retrieve(query, memory_text, max_tokens)
    if strategy == "N13":
        # Frozen N13 allocation: structured facts take priority up to max_tokens; remaining budget uses BM25 archive
        # excluding turn_ids already covered by selected structured facts to prevent duplicate consumption.
        fact_text, fact_tids = fact_memory_retrieve(query, memory_text, max_tokens) if (memory_text or "").strip() else ("", [])
        used_fact = estimate_text_tokens(fact_text) if fact_text else 0
        rem = max_tokens - used_fact - (2 if fact_text else 0)
        bm25_text, bm25_tids = "", []
        if rem > 0 and archived_exchanges:
            fact_tid_set = set(fact_tids)
            filtered_pool = [ex for ex in archived_exchanges if ex.turn_id not in fact_tid_set]
            bm25_text, bm25_tids = bm25_retrieve(query, filtered_pool, rem)
        parts = [p for p in (fact_text, bm25_text) if p]
        return "\n\n".join(parts), sorted(set(fact_tids) | set(bm25_tids))
    return memory_text, []


def select_recent_exchanges(
    exchanges: list[Exchange],
    budget_limit: int,
    system_prompt: str,
    memory_text: str,
    current_user_text: str,
    total_cap: int,
    current_assistant_text: str | None = None,
) -> tuple[list[Exchange], list[Exchange]]:
    """Keep newest complete exchanges fitting within both recent_cap and total_cap."""
    if not exchanges or budget_limit <= 0:
        return [], list(exchanges)

    kept_rev: list[Exchange] = []
    used_recent = 0
    for ex in reversed(exchanges):
        ex_msgs = [{"role": "user", "content": ex.user_text}]
        if ex.assistant_text is not None:
            ex_msgs.append({"role": "assistant", "content": ex.assistant_text})
        ex_tok = estimate_messages_tokens(ex_msgs) - 3
        if used_recent + ex_tok > budget_limit:
            break
        candidate_kept = list(reversed(kept_rev + [ex]))
        total_tok = estimate_messages_tokens(
            build_messages(system_prompt, memory_text, candidate_kept, current_user_text)
        )
        if current_assistant_text is not None:
            total_tok += 4 + estimate_text_tokens("assistant") + estimate_text_tokens(current_assistant_text)
        if total_tok > total_cap:
            break
        kept_rev.append(ex)
        used_recent += ex_tok

    kept = list(reversed(kept_rev))
    evicted_count = len(exchanges) - len(kept)
    return kept, exchanges[:evicted_count]


def apply_maintenance(
    strategy: str,
    turn_id: int,
    run_id: str,
    system_prompt: str,
    memory_text: str,
    recent_exchanges: list[Exchange],
    current_user_text: str,
    budget: BudgetConfig,
    llm_client: LLMClient | None = None,
    compaction_model: str | None = None,
    compaction_prompt_template: str = "",
    current_assistant_text: str | None = None,
    progress_callback=None,
    archived_exchanges: list[Exchange] | None = None,
    is_query: bool = False,
) -> dict[str, Any]:
    """Execute strategy maintenance for a single turn and return updated state, messages, and events."""
    def current_messages(system_prompt, memory_text, recent_exchanges, current_user_text):
        messages = build_messages(system_prompt, memory_text, recent_exchanges, current_user_text)
        if current_assistant_text is not None:
            messages.append({"role": "assistant", "content": current_assistant_text})
        return messages

    events: list[dict[str, Any]] = []
    maint_input_tokens_actual = 0
    maint_output_tokens_actual = 0
    maint_latency_ms = 0

    if strategy == "N0":
        msgs = current_messages(system_prompt, "", [], current_user_text)
        tok = estimate_messages_tokens(msgs)
        if tok > budget.input_tokens:
            return {
                "status": "overflow",
                "memory_text": "",
                "recent_exchanges": [],
                "messages": msgs,
                "estimated_tokens": tok,
                "events": events,
                "maint_usage": (0, 0, 0),
            }
        return {
            "status": "ok",
            "memory_text": "",
            "recent_exchanges": [],
            "messages": msgs,
            "estimated_tokens": tok,
            "events": events,
            "maint_usage": (0, 0, 0),
        }

    history = recent_exchanges
    if strategy == "N5":
        history = list(chain.from_iterable(chunk_exchanges(history, budget.chunk_tokens)))

    candidate_msgs = current_messages(
        system_prompt,
        memory_text if strategy in MEMORY_STRATEGIES else "",
        history,
        current_user_text,
    )
    candidate_tokens = estimate_messages_tokens(candidate_msgs)

    if strategy in FULL_HISTORY_STRATEGIES:
        if budget.n1_context_tokens is None:
            raise ValueError("N1/N5 require an explicit model context capacity (n1_context_tokens)")
        input_capacity = budget.n1_context_tokens - budget.max_output_tokens
        if candidate_tokens > input_capacity:
            return {
                "status": "overflow",
                "memory_text": "",
                "recent_exchanges": recent_exchanges,
                "messages": candidate_msgs,
                "estimated_tokens": candidate_tokens,
                "events": events,
                "maint_usage": (0, 0, 0),
            }
        return {
            "status": "ok",
            "memory_text": "",
            "recent_exchanges": recent_exchanges,
            "messages": candidate_msgs,
            "estimated_tokens": candidate_tokens,
            "events": events,
            "maint_usage": (0, 0, 0),
        }

    # N2-N4, N6-N14: trigger maintenance if candidate_tokens > 0.8 * B (or retrieval query)
    if candidate_tokens <= budget.trigger_tokens and not (
        is_query
        and (
            (strategy in ARCHIVE_STRATEGIES and archived_exchanges)
            or (strategy in FACT_STRATEGIES and memory_text.strip())
        )
    ):
        return {
            "status": "ok",
            "memory_text": memory_text if strategy in MEMORY_STRATEGIES else "",
            "recent_exchanges": recent_exchanges,
            "messages": candidate_msgs,
            "estimated_tokens": candidate_tokens,
            "events": events,
            "maint_usage": (0, 0, 0),
        }

    T = budget.target_tokens
    F = estimate_messages_tokens(current_messages(system_prompt, "", [], current_user_text))

    if F > T:
        if F > budget.input_tokens:
            return {
                "status": "overflow",
                "memory_text": memory_text,
                "recent_exchanges": recent_exchanges,
                "messages": current_messages(system_prompt, "", [], current_user_text),
                "estimated_tokens": F,
                "events": events,
                "maint_usage": (0, 0, 0),
            }
        # target_unreachable but <= B: evict memory and recent exchanges
        ev_text_before = serialize_exchanges(recent_exchanges)
        events.append({
            "event_id": f"{run_id}-t{turn_id}-e1",
            "run_id": run_id,
            "turn_id": turn_id,
            "owner": "external",
            "mechanism": "target_unreachable_eviction",
            "phase": "before_answer",
            "trigger": "candidate_gt_0.8B_and_F_gt_0.6B",
            "parent_event_ids": [],
            "input_tokens": candidate_tokens,
            "output_tokens": F,
            "status": "target_unreachable",
            "metrics": {"lexical_jaccard": None, "lexical_retention": None, "compression_ratio": None},
            "metric_null_reasons": {"all": "budget_eviction"},
            "evicted_turn_ids": [ex.turn_id for ex in recent_exchanges],
            "before_text": ev_text_before,
            "after_text": "",
        })
        msgs = current_messages(system_prompt, "", [], current_user_text)
        return {
            "status": "target_unreachable",
            "memory_text": "",
            "recent_exchanges": [],
            "messages": msgs,
            "estimated_tokens": estimate_messages_tokens(msgs),
            "events": events,
            "maint_usage": (0, 0, 0),
        }

    if strategy == "N2":
        kept, evicted = select_recent_exchanges(
            recent_exchanges,
            min(budget.recent_history_tokens, T - F),
            system_prompt,
            "",
            current_user_text,
            T,
            current_assistant_text,
        )
        before_str = serialize_exchanges(recent_exchanges)
        after_str = serialize_exchanges(kept)
        jac, jac_r = lexical_jaccard(before_str, after_str)
        ret, ret_r = lexical_retention(before_str, after_str)
        in_tok = estimate_text_tokens(before_str)
        out_tok = estimate_text_tokens(after_str)
        comp_ratio = round(out_tok / in_tok, 6) if in_tok > 0 else None
        events.append({
            "event_id": f"{run_id}-t{turn_id}-e1",
            "run_id": run_id,
            "turn_id": turn_id,
            "owner": "external",
            "mechanism": "sliding_window_eviction",
            "phase": "before_answer",
            "trigger": f"candidate_tokens({candidate_tokens})>trigger({budget.trigger_tokens})",
            "parent_event_ids": [],
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "status": "ok",
            "metrics": {
                "lexical_jaccard": jac,
                "lexical_retention": ret,
                "compression_ratio": comp_ratio,
            },
            "metric_null_reasons": {
                k: v for k, v in [("lexical_jaccard", jac_r), ("lexical_retention", ret_r)] if v
            },
            "evicted_turn_ids": [ex.turn_id for ex in evicted],
            "before_text": before_str,
            "after_text": after_str,
        })
        msgs = current_messages(system_prompt, "", kept, current_user_text)
        return {
            "status": "ok",
            "memory_text": "",
            "recent_exchanges": kept,
            "messages": msgs,
            "estimated_tokens": estimate_messages_tokens(msgs),
            "events": events,
            "maint_usage": (0, 0, 0),
        }

    # N3 / N4 / N6 / N7: reserve m = min(2048, T - F), then recent <= min(4096, T - F - m)
    m = min(budget.memory_tokens, T - F)
    if m == 0:
        kept, evicted = select_recent_exchanges(
            recent_exchanges,
            min(budget.recent_history_tokens, T - F),
            system_prompt,
            "",
            current_user_text,
            T,
            current_assistant_text,
        )
        events.append({
            "event_id": f"{run_id}-t{turn_id}-e1",
            "run_id": run_id,
            "turn_id": turn_id,
            "owner": "external",
            "mechanism": "zero_memory_budget_eviction",
            "phase": "before_answer",
            "trigger": f"candidate_tokens({candidate_tokens})>trigger({budget.trigger_tokens})",
            "parent_event_ids": [],
            "input_tokens": estimate_text_tokens(memory_text),
            "output_tokens": 0,
            "status": "budget_eviction",
            "metrics": {"lexical_jaccard": None, "lexical_retention": None, "compression_ratio": None},
            "metric_null_reasons": {"all": "zero_memory_budget"},
            "evicted_turn_ids": [ex.turn_id for ex in evicted],
            "before_text": memory_text,
            "after_text": "",
        })
        msgs = current_messages(system_prompt, "", kept, current_user_text)
        return {
            "status": "ok",
            "memory_text": "",
            "recent_exchanges": kept,
            "messages": msgs,
            "estimated_tokens": estimate_messages_tokens(msgs),
            "events": events,
            "maint_usage": (0, 0, 0),
        }

    recent_budget = min(budget.recent_history_tokens, max(0, T - F - m))
    # Reserve m tokens in total_cap check when selecting recent exchanges
    kept, evicted = select_recent_exchanges(
        recent_exchanges,
        recent_budget,
        system_prompt,
        "x" * math.ceil(m * 3.8),
        current_user_text,
        T,
        current_assistant_text,
    )
    evicted_text = serialize_exchanges(evicted)
    maint_input_text = f"{memory_text}\n{evicted_text}".strip()

    if strategy in ARCHIVE_ONLY_STRATEGIES:
        policy = ARCHIVE_POLICY_BY_STRATEGY.get(strategy)
        if not is_query:
            before_str = serialize_exchanges(recent_exchanges)
            after_str = serialize_exchanges(kept)
            in_tok = estimate_text_tokens(before_str)
            out_tok = estimate_text_tokens(after_str)
            arch_evicted_ids: list[int] = []
            if policy:
                full_pool = list(archived_exchanges or []) + list(evicted)
                _, dropped_arch = bound_archive(full_pool, budget.n8_archive_tokens, policy=policy)
                arch_evicted_ids = [ex.turn_id for ex in dropped_arch]
            mech_map = {
                "N7": "bm25_index_eviction",
                "N8": "bm25_bounded_eviction",
                "N11": "bm25_dedup_eviction",
                "N12": "bm25_session_balanced_eviction",
            }
            events.append({
                "event_id": f"{run_id}-t{turn_id}-e1",
                "run_id": run_id,
                "turn_id": turn_id,
                "owner": "external",
                "mechanism": mech_map[strategy],
                "phase": "before_answer",
                "trigger": f"candidate_tokens({candidate_tokens})>trigger({budget.trigger_tokens})",
                "parent_event_ids": [],
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "status": "ok",
                "metrics": {"lexical_jaccard": None, "lexical_retention": None, "compression_ratio": round(out_tok / in_tok, 6) if in_tok > 0 else None},
                "evicted_turn_ids": [ex.turn_id for ex in evicted],
                "archive_evicted_turn_ids": arch_evicted_ids,
                "before_text": before_str,
                "after_text": after_str,
            })
            msgs = current_messages(system_prompt, "", kept, current_user_text)
            return {
                "status": "ok",
                "memory_text": "",
                "recent_exchanges": kept,
                "evicted_exchanges": evicted,
                "messages": msgs,
                "estimated_tokens": estimate_messages_tokens(msgs),
                "events": events,
                "maint_usage": (0, 0, 0),
                "search_latency_ms": 0.0,
            }
        pool = list(archived_exchanges or []) + list(evicted)
        if policy:
            pool, _ = bound_archive(pool, budget.n8_archive_tokens, policy=policy)
        t0 = time.perf_counter()
        retrieved_memory, retrieved_ids = build_readout_memory(strategy, current_user_text, "", pool, m)
        search_latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        final_msgs = current_messages(system_prompt, retrieved_memory, kept, current_user_text)
        while estimate_messages_tokens(final_msgs) > T and kept:
            evicted.append(kept[0])
            kept = kept[1:]
            final_msgs = current_messages(system_prompt, retrieved_memory, kept, current_user_text)
        pool_text = serialize_exchanges(pool)
        in_tok = estimate_text_tokens(pool_text)
        out_tok = estimate_text_tokens(retrieved_memory)
        events.append({
            "event_id": f"{run_id}-t{turn_id}-e1",
            "run_id": run_id,
            "turn_id": turn_id,
            "owner": "external",
            "mechanism": "bm25_retrieval",
            "phase": "before_answer",
            "trigger": "query_readout",
            "parent_event_ids": [],
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_ms": search_latency_ms,
            "status": "ok",
            "metrics": {"lexical_jaccard": None, "lexical_retention": None, "compression_ratio": round(out_tok / in_tok, 6) if in_tok > 0 else None},
            "evicted_turn_ids": [ex.turn_id for ex in evicted],
            "retrieved_turn_ids": retrieved_ids,
            "before_text": pool_text,
            "after_text": retrieved_memory,
        })
        return {
            "status": "ok",
            "memory_text": retrieved_memory,
            "recent_exchanges": kept,
            "evicted_exchanges": evicted,
            "messages": final_msgs,
            "estimated_tokens": estimate_messages_tokens(final_msgs),
            "events": events,
            "maint_usage": (0, 0, 0),
            "search_latency_ms": search_latency_ms,
        }

    mechanism = {
        "N3": "extractive_compaction",
        "N4": "llm_compaction",
        "N6": "chunked_llm_compaction",
        "N9": "fact_append_compaction",
        "N10": "fact_update_compaction",
        "N13": "fact_hybrid_compaction",
        "N14": "fact_enhanced_compaction",
    }[strategy]
    fact_stats: dict[str, Any] | None = None
    arch_evicted_ids_fact: list[int] = []
    if strategy in ("N3", *FACT_STRATEGIES):
        if strategy == "N3":
            new_memory = extractive_compact(memory_text, evicted, m)
        else:
            new_memory, fact_stats = fact_memory_compact_with_stats(
                memory_text,
                evicted,
                m,
                allow_update=(strategy in ("N10", "N13", "N14")),
                store_fallback=(strategy != "N13"),
                enhanced_extractor=(strategy == "N14"),
            )
            pool_for_n13: list[Exchange] | None = None
            if strategy == "N13":
                full_pool = list(archived_exchanges or []) + list(evicted)
                pool_for_n13, dropped_n13 = bound_archive(full_pool, budget.n8_archive_tokens, policy="fifo")
                arch_evicted_ids_fact = [ex.turn_id for ex in dropped_n13]
            if is_query:
                t0 = time.perf_counter()
                retrieved_mem, retrieved_ids = build_readout_memory(strategy, current_user_text, new_memory, pool_for_n13, m)
                search_latency_ms = round((time.perf_counter() - t0) * 1000, 3)
                final_msgs = current_messages(system_prompt, retrieved_mem, kept, current_user_text)
                while estimate_messages_tokens(final_msgs) > T and kept:
                    evicted.append(kept[0])
                    kept = kept[1:]
                    final_msgs = current_messages(system_prompt, retrieved_mem, kept, current_user_text)
                events.append({
                    "event_id": f"{run_id}-t{turn_id}-e1",
                    "run_id": run_id,
                    "turn_id": turn_id,
                    "owner": "external",
                    "mechanism": mechanism,
                    "phase": "before_answer",
                    "trigger": "query_readout",
                    "parent_event_ids": [],
                    "input_tokens": estimate_text_tokens(maint_input_text),
                    "output_tokens": estimate_text_tokens(retrieved_mem),
                    "latency_ms": search_latency_ms,
                    "status": "ok",
                    "metrics": {
                        "lexical_jaccard": None,
                        "lexical_retention": None,
                        "compression_ratio": None,
                        "fact_memory_stats": fact_stats,
                    },
                    "evicted_turn_ids": [ex.turn_id for ex in evicted],
                    "archive_evicted_turn_ids": arch_evicted_ids_fact,
                    "retrieved_turn_ids": retrieved_ids,
                    "before_text": maint_input_text,
                    "after_text": new_memory,
                })
                return {
                    "status": "ok",
                    "memory_text": new_memory,
                    "recent_exchanges": kept,
                    "evicted_exchanges": evicted,
                    "messages": final_msgs,
                    "estimated_tokens": estimate_messages_tokens(final_msgs),
                    "events": events,
                    "maint_usage": (0, 0, 0),
                    "search_latency_ms": search_latency_ms,
                }
        status = "ok"
    else:
        # Preserve every attempt, including empty/truncated/over-budget responses.
        if llm_client is None or not compaction_model:
            raise RuntimeError(f"{strategy} requires llm_client and compaction_model")
        attempt_records = []

        def compact(previous_memory, source_text, stage):
            nonlocal maint_input_tokens_actual, maint_output_tokens_actual, maint_latency_ms
            status = "maintenance_empty_error"
            new_memory = ""
            prev_est = 0
            for attempt in range(2):
                if progress_callback:
                    progress_callback(stage | {"phase": "maintenance_attempt_start", "turn_id": turn_id, "attempt": attempt + 1})
                target = max(1, int(m * (budget.n4_target_ratio if attempt == 0 else budget.n4_retry_ratio)))
                word_budget = max(1, int(target * 0.5))
                char_budget = max(1, int(target * 1.1))
                prompt_text = (
                    compaction_prompt_template.replace("{{memory_token_budget}}", str(target))
                    .replace("{{memory_word_budget}}", str(word_budget))
                    .replace("{{memory_char_budget}}", str(char_budget))
                    .replace("{{previous_memory}}", previous_memory or "(空)")
                    .replace("{{evicted_turns_with_ids}}", source_text or "(无)")
                )
                if attempt:
                    prompt_text += (
                        f"\n\n注意：上一次压缩输出约 {prev_est} tokens，未能满足 {m} tokens 硬上限（或被截断）。"
                        f"请大幅删减寒暄与次要细节、合并同类条目，直接输出完整且极简的记忆正文，严格控制在 {target} tokens"
                        f"（约 {word_budget} 英文词 / {char_budget} 汉字）以内。"
                    )
                if attempt and new_memory:
                    prompt_text += "\n<previous_attempt>\n" + new_memory + "\n</previous_attempt>\n请缩短上次草稿，并以原始输入核实事实；草稿可能不完整或有误。"
                messages = [{"role": "user", "content": prompt_text}]
                options = {"thinking": "disabled"} if llm_client.provider == "deepseek" else {}
                try:
                    res = llm_client.chat(model=compaction_model, messages=messages,
                                          temperature=0.0, max_tokens=budget.n4_output_tokens, **options)
                except Exception as exc:
                    attempt_records.append(stage | {"messages": messages, "status": "api_error", "error_type": type(exc).__name__})
                    status = "maintenance_api_error"
                    if progress_callback:
                        progress_callback(stage | {"phase": "maintenance_attempt_end", "turn_id": turn_id, "attempt": attempt + 1, "status": status, "error_type": type(exc).__name__})
                    break
                new_memory = res["text"].strip()
                maint_input_tokens_actual += res["usage"].get("prompt_tokens") or 0
                maint_output_tokens_actual += res["usage"].get("completion_tokens") or 0
                maint_latency_ms += res["latency_ms"]
                prev_est = estimate_text_tokens(new_memory)
                if not new_memory:
                    status = "maintenance_empty_error"
                elif res.get("finish_reason") == "length":
                    status = "maintenance_truncated_error"
                elif prev_est > m:
                    status = "maintenance_budget_error"
                else:
                    status = "ok"
                attempt_records.append(stage | {"messages": messages, "response": res, "status": status,
                                        "memory_tokens_est": prev_est,
                                        "memory_budget": m, "target_tokens_est": target,
                                        "max_output_tokens": budget.n4_output_tokens, "thinking": options.get("thinking")})
                if progress_callback:
                    progress_callback(stage | {"phase": "maintenance_attempt_end", "turn_id": turn_id, "attempt": attempt + 1, "status": status, "latency_ms": res["latency_ms"], "usage": res["usage"]})
                if status == "ok":
                    break
            return status, new_memory

        if strategy == "N6":
            chunks = chunk_exchanges(evicted, budget.chunk_tokens)
            partials = []
            status = "ok"
            for index, chunk in enumerate(chunks, 1):
                status, new_memory = compact("", serialize_exchanges(chunk), {
                    "compaction_stage": "chunk",
                    "chunk_index": index,
                    "source_turn_ids": [ex.turn_id for ex in chunk],
                })
                if status != "ok":
                    break
                partials.append(new_memory)
            if status == "ok":
                status, new_memory = compact(memory_text, "\n\n".join(partials), {
                    "compaction_stage": "merge",
                })
        else:
            status, new_memory = compact(memory_text, evicted_text, {})
        if status != "ok":
            events.append({"event_id": f"{run_id}-t{turn_id}-e1", "run_id": run_id,
                           "turn_id": turn_id, "owner": "external", "mechanism": mechanism,
                           "phase": "before_answer", "status": status, "trigger": "candidate_gt_threshold",
                           "evicted_turn_ids": [ex.turn_id for ex in evicted],
                           "before_text": maint_input_text, "after_text": new_memory,
                           "input_tokens": estimate_text_tokens(maint_input_text),
                           "output_tokens": estimate_text_tokens(new_memory),
                           "latency_ms": maint_latency_ms, "attempts": attempt_records,
                           "metrics": {"lexical_jaccard": None, "lexical_retention": None, "compression_ratio": None}})
            return {"status": status, "memory_text": memory_text, "recent_exchanges": recent_exchanges,
                    "messages": candidate_msgs, "estimated_tokens": candidate_tokens,
                    "events": events, "maint_usage": (maint_input_tokens_actual, maint_output_tokens_actual, maint_latency_ms)}

    # Final serialized request verification
    final_msgs = current_messages(system_prompt, new_memory, kept, current_user_text)
    while estimate_messages_tokens(final_msgs) > T and kept:
        evicted.append(kept[0])
        kept = kept[1:]
        final_msgs = current_messages(system_prompt, new_memory, kept, current_user_text)

    in_tok = estimate_text_tokens(maint_input_text)
    out_tok = estimate_text_tokens(new_memory)
    jac, jac_r = lexical_jaccard(maint_input_text, new_memory)
    ret, ret_r = lexical_retention(maint_input_text, new_memory)
    comp_ratio = round(out_tok / in_tok, 6) if in_tok > 0 else None

    events.append({
        "event_id": f"{run_id}-t{turn_id}-e1",
        "run_id": run_id,
        "turn_id": turn_id,
        "owner": "external",
        "mechanism": mechanism,
        "phase": "before_answer",
        "trigger": f"candidate_tokens({candidate_tokens})>trigger({budget.trigger_tokens})",
        "parent_event_ids": [],
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_ms": maint_latency_ms,
        "status": status,
        "metrics": {
            "lexical_jaccard": jac,
            "lexical_retention": ret,
            "compression_ratio": comp_ratio,
            **({"fact_memory_stats": fact_stats} if fact_stats is not None else {}),
        },
        "metric_null_reasons": {
            k: v for k, v in [("lexical_jaccard", jac_r), ("lexical_retention", ret_r)] if v
        },
        "evicted_turn_ids": [ex.turn_id for ex in evicted],
        **({"archive_evicted_turn_ids": arch_evicted_ids_fact} if strategy == "N13" else {}),
        "before_text": maint_input_text,
        "after_text": new_memory,
        "attempts": attempt_records if strategy in LLM_STRATEGIES else [],
    })

    return {
        "status": status,
        "memory_text": new_memory,
        "recent_exchanges": kept,
        "evicted_exchanges": evicted,
        "messages": final_msgs,
        "estimated_tokens": estimate_messages_tokens(final_msgs),
        "events": events,
        "maint_usage": (maint_input_tokens_actual, maint_output_tokens_actual, maint_latency_ms),
    }
