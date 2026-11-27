"""Build static Showcase data bundles from verified probe plans and benchmark batches.

Generates:
  - showcase/data/overview.js
  - showcase/data/cases/<plan_key>__<qid>.js

Both use plain JS global assignment so `showcase/index.html` works seamlessly
over both `file://` and `http://` with zero CORS issues.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_adapters import adapt_longmemeval_item  # noqa: E402
from run_probe_plan import _find_saved_record, _resolve_source_batch, _resolve_turns  # noqa: E402

TZ_CN = timezone(timedelta(hours=8))


def _has_answer_turn_ids(raw_item: dict[str, Any]) -> set[int]:
    """Return 1-based turn_ids whose user or assistant message carries has_answer=True."""
    out: set[int] = set()
    global_turn = 0
    for sess in raw_item.get("haystack_sessions", []):
        i = 0
        while i < len(sess):
            m = sess[i]
            if m.get("role") == "user":
                global_turn += 1
                if m.get("has_answer"):
                    out.add(global_turn)
                if i + 1 < len(sess) and sess[i + 1].get("role") == "assistant":
                    if sess[i + 1].get("has_answer"):
                        out.add(global_turn)
                    i += 2
                    continue
            i += 1
    return out

def _resolve_run_dir(plan: dict[str, Any], qid: str, strategy: str) -> Path:
    batch_id = _resolve_source_batch(plan, qid, strategy)
    batch_dir = ROOT / "runs" / plan["model"] / batch_id
    matches = list((batch_dir / "runs").glob(f"*-{qid}-direct-{strategy}-r1"))
    if not matches:
        matches = list(batch_dir.glob(f"*-{qid}-direct-{strategy}-r1"))
    if not matches:
        raise FileNotFoundError(f"Run dir not found for {qid}/{strategy} in {batch_dir}")
    return matches[0]

DEV_EXPOSED_QIDS = {"01493427", "45dc21b6"}

STRATEGY_SPECS: list[dict[str, Any]] = [
    {
        "id": "N0",
        "name": "Current only",
        "zh_name": "无记忆基线（仅当前轮）",
        "family": "baseline",
        "family_zh": "基线对照",
        "status": "executed",
        "visible_history": "仅当前用户输入",
        "explicit_memory": "无",
        "budget_rule": "B=8192（不保留历史）",
        "archive_policy": "—",
        "maintenance_mechanism": "每轮清空历史，用作“无历史对照”（非数学下界）。",
        "control_note": "基线对照",
    },
    {
        "id": "N1",
        "name": "Long context",
        "zh_name": "全量长上下文基线",
        "family": "baseline",
        "family_zh": "基线对照",
        "status": "executed",
        "visible_history": "全部已摄入历史 + 当前输入",
        "explicit_memory": "无",
        "budget_rule": "模型上限 131072（预留 512 输出，超限记 overflow）",
        "archive_policy": "—",
        "maintenance_mechanism": "不做任何压缩或淘汰；超过模型上下文上限后立即终止。",
        "control_note": "上界参考（受真实长历史长度制约）",
    },
    {
        "id": "N2",
        "name": "Sliding window",
        "zh_name": "纯滑窗淘汰",
        "family": "window_compact",
        "family_zh": "滑窗 / 压缩",
        "status": "executed",
        "visible_history": "最近完整轮次",
        "explicit_memory": "无",
        "budget_rule": "B=8192，trigger>6553，target≤4915",
        "archive_policy": "—",
        "maintenance_mechanism": "超阈值时从最旧轮次开始直接丢弃，不生成任何外部记忆。",
        "control_note": "8K 受控滑窗下界",
    },
    {
        "id": "N3",
        "name": "Extractive compaction",
        "zh_name": "确定性抽取压缩",
        "family": "window_compact",
        "family_zh": "滑窗 / 压缩",
        "status": "executed",
        "visible_history": "最近完整轮次",
        "explicit_memory": "M≤2048（句级去重摘录）",
        "budget_rule": "B=8192，M≤2048，recent≤min(4096, T-F-m)",
        "archive_policy": "—",
        "maintenance_mechanism": "无 LLM；对移出窗口对话做标准化句级去重，按新到旧优先装入 M。",
        "control_note": "N3/N4 属于 method_bundle 对照",
    },
    {
        "id": "N4",
        "name": "LLM compaction",
        "zh_name": "LLM 增量摘要压缩",
        "family": "window_compact",
        "family_zh": "滑窗 / 压缩",
        "status": "executed",
        "visible_history": "最近完整轮次",
        "explicit_memory": "M≤2048（增量 LLM 摘要）",
        "budget_rule": "B=8192，M≤2048；压缩输出超限或截断失败记 error",
        "archive_policy": "—",
        "maintenance_mechanism": "触发维护时调用辅助 LLM 对“旧 M + 移出窗口轮次”做增量重写摘要。",
        "control_note": "无静默回退；展示 LLM 重写保真度与预算稳定性权衡",
    },
    {
        "id": "N5",
        "name": "Chunked full replay",
        "zh_name": "分块全量重放检查",
        "family": "baseline",
        "family_zh": "基线对照",
        "status": "executed",
        "visible_history": "分块后按原序全量重组",
        "explicit_memory": "无",
        "budget_rule": "模型上限 131072（与 N1 相同）",
        "archive_policy": "—",
        "maintenance_mechanism": "将历史按 chunk_tokens 分块后按原序拼回，用于检验分块边界是否改变结果。",
        "control_note": "实现一致性检查（对照 N1）",
    },
    {
        "id": "N6",
        "name": "Chunked LLM compaction",
        "zh_name": "分块 LLM 摘要",
        "family": "window_compact",
        "family_zh": "滑窗 / 压缩",
        "status": "executed",
        "visible_history": "最近完整轮次",
        "explicit_memory": "M≤2048（分块再合并摘要）",
        "budget_rule": "B=8192，M≤2048",
        "archive_policy": "—",
        "maintenance_mechanism": "每轮对超限历史分块摘要后再合并（chunk + merge）；受控预算 B=8192，M≤2048。",
        "control_note": "分块再合并 LLM 摘要基线",
    },
    {
        "id": "N7",
        "name": "Unbounded BM25 archive",
        "zh_name": "无界 BM25 归档检索",
        "family": "archive_bm25",
        "family_zh": "外部归档 BM25",
        "status": "executed",
        "visible_history": "最近完整轮次",
        "explicit_memory": "M≤2048（BM25 召回原文）",
        "budget_rule": "B=8192，归档容量无上限",
        "archive_policy": "unbounded（全历史归档）",
        "maintenance_mechanism": "滑出窗口的完整轮次进入本地无界归档库；读出/终答时按 BM25 召回 top-k 填入 M。",
        "control_note": "存储+检索无界对照基线",
    },
    {
        "id": "N8",
        "name": "Bounded BM25 (FIFO)",
        "zh_name": "16K 有界 BM25（先进先出）",
        "family": "archive_bm25",
        "family_zh": "外部归档 BM25",
        "status": "executed",
        "visible_history": "最近完整轮次",
        "explicit_memory": "M≤2048（BM25 召回原文）",
        "budget_rule": "B=8192，归档硬上限 16384 tokens",
        "archive_policy": "fifo（超 16K 淘汰最旧轮次）",
        "maintenance_mechanism": "沿用 N7 检索读出，但归档池受 16K tokens 约束，超限时按 FIFO 永久淘汰最旧轮次。",
        "control_note": "有界归档基线（N8 vs N11 vs N12）",
    },
    {
        "id": "N9",
        "name": "Append-only fact memory",
        "zh_name": "只追加规则事实记忆",
        "family": "fact_hybrid",
        "family_zh": "事实表 / 混合",
        "status": "executed",
        "visible_history": "最近完整轮次",
        "explicit_memory": "M≤2048（结构化事实 + 原句兜底）",
        "budget_rule": "B=8192，M≤2048",
        "archive_policy": "—",
        "maintenance_mechanism": "仅从 user_text 提取规则事实或以 context|s_... 兜底，新事实只追加不覆写旧值。",
        "control_note": "N9 vs N10 单因子对照（是否更新/撤回）",
    },
    {
        "id": "N10",
        "name": "Updatable fact memory",
        "zh_name": "可更新/撤回规则事实记忆",
        "family": "fact_hybrid",
        "family_zh": "事实表 / 混合",
        "status": "executed",
        "visible_history": "最近完整轮次",
        "explicit_memory": "M≤2048（可更新 + [RETRACTED]）",
        "budget_rule": "B=8192，M≤2048",
        "archive_policy": "—",
        "maintenance_mechanism": "与 N9 共用提取器与预算；同实体属性更新时就地替换，撤回写入显式 [RETRACTED]。",
        "control_note": "后端可更新对照（对比 N9 / N14）",
    },
    {
        "id": "N11",
        "name": "Bounded BM25 (Dedup FIFO)",
        "zh_name": "16K 有界 BM25（去重 FIFO）",
        "family": "archive_bm25",
        "family_zh": "外部归档 BM25",
        "status": "executed",
        "visible_history": "最近完整轮次",
        "explicit_memory": "M≤2048（BM25 召回原文）",
        "budget_rule": "B=8192，归档硬上限 16384 tokens",
        "archive_policy": "dedup_fifo（先文本去重保留最新再 FIFO）",
        "maintenance_mechanism": "与 N8 共用容量与 BM25 读出；归档入池先按文本精确去重（保留最新来源），满载再 FIFO。",
        "control_note": "归档淘汰策略对照（N8 vs N11）",
    },
    {
        "id": "N12",
        "name": "Bounded BM25 (Session-balanced)",
        "zh_name": "16K 有界 BM25（会话均衡）",
        "family": "archive_bm25",
        "family_zh": "外部归档 BM25",
        "status": "executed",
        "visible_history": "最近完整轮次",
        "explicit_memory": "M≤2048（BM25 召回原文）",
        "budget_rule": "B=8192，归档硬上限 16384 tokens",
        "archive_policy": "session_balanced（优先削减最大占用会话）",
        "maintenance_mechanism": "满载时优先淘汰当前归档占用最大 session 中的最旧轮次，防止单个长对话刷爆归档。",
        "control_note": "归档淘汰策略对照（N8 vs N12）",
    },
    {
        "id": "N13",
        "name": "Fact + BM25 hybrid memory",
        "zh_name": "结构化事实优先 + BM25 兜底混合",
        "family": "fact_hybrid",
        "family_zh": "事实表 / 混合",
        "status": "executed",
        "visible_history": "最近完整轮次",
        "explicit_memory": "M≤2048（结构化事实优先 + 剩余 BM25）",
        "budget_rule": "B=8192，归档上限 16384（FIFO），M≤2048",
        "archive_policy": "fifo（事实未命中句不再挤占 2K 事实表）",
        "maintenance_mechanism": "事实表仅保留命中规则的结构化槽位；读出时事实优先占 M，余下预算由 BM25 从归档补召回。",
        "control_note": "与 N10 严格标注为 method_bundle 对照",
    },
    {
        "id": "N14",
        "name": "Enhanced-extractor updatable fact",
        "zh_name": "增强提取 + 可更新/撤回事实记忆",
        "family": "fact_hybrid",
        "family_zh": "事实表 / 混合",
        "status": "executed",
        "visible_history": "最近完整轮次",
        "explicit_memory": "M≤2048（增强提取 + 可更新槽位）",
        "budget_rule": "B=8192，M≤2048",
        "archive_policy": "—",
        "maintenance_mechanism": "与 N10 共用同一后端状态槽位与更新/撤回规则，仅升级前端自然语言进度/状态变更提取器。",
        "control_note": "N10 vs N14 单因子消融（前端提取召回率）",
    },
]

EVENT_FIELD_LABELS: dict[str, tuple[str, str]] = {
    "sliding_window_eviction": ("维护前近期窗口全文", "淘汰旧轮次后保留的近期窗口全文"),
    "extractive_compaction": ("原摘要记忆 + 本轮淘汰原文", "句级去重抽取压缩后的记忆 M"),
    "llm_compaction": ("原摘要记忆 + 本轮淘汰原文", "LLM 增量摘要重写后的记忆 M"),
    "bm25_index_eviction": ("维护前近期窗口（或检索候选）", "维护后近期窗口（或 BM25 命中片段）"),
    "bm25_bounded_eviction": ("维护前近期窗口（或检索候选）", "维护后近期窗口（或 BM25 命中片段）"),
    "bm25_dedup_eviction": ("维护前近期窗口（或检索候选）", "维护后近期窗口（或 BM25 命中片段）"),
    "bm25_session_balanced_eviction": ("维护前近期窗口（或检索候选）", "维护后近期窗口（或 BM25 命中片段）"),
    "fact_append_compaction": ("原事实记忆 + 本轮淘汰原文", "追加更新后的事实记忆 M"),
    "fact_updatable_compaction": ("原事实记忆 + 本轮淘汰原文", "更新/撤回处理后的事实记忆 M"),
    "fact_hybrid_compaction": ("原结构化事实表 + 本轮淘汰原文", "过滤后结构化事实表 M（未命中入归档）"),
    "fact_enhanced_compaction": ("原事实记忆 + 本轮淘汰原文", "增强提取更新/撤回后的事实记忆 M"),
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _clip(text: str | None, limit: int = 4500) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _load_question_csv(csv_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["question_id"], row["strategy"])] = {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "is_abstention": row["is_abstention"].lower() == "true",
                "strategy": row["strategy"],
                "status": row["status"],
                "judge_status": row["judge_status"],
                "judge_label": (
                    None
                    if row["judge_label"] in ("", "None", "null")
                    else row["judge_label"].lower() in ("true", "1", "yes", "correct")
                ),
                "ingested_tokens_est": int(float(row["ingested_tokens_est"] or 0)),
                "final_ctx_tokens_est": int(float(row["final_ctx_tokens_est"] or 0)),
                "maintenance_events": int(float(row["maintenance_events"] or 0)),
                "total_tokens": int(float(row["total_tokens"] or 0)),
                "outcome_category": row["outcome_category"],
                "final_evidence_retention": (
                    None
                    if row["final_evidence_retention"] in ("", "None", "null")
                    else round(float(row["final_evidence_retention"]), 4)
                ),
            }
    return out


def _compute_sdev30_online_probes() -> dict[str, Any]:
    existing = _read_json(ROOT / "reports/ecnu-plus/s-dev-30-existing-probe-analysis/summary.json")
    base = ROOT / "runs/ecnu-plus/longmemeval-s-dev-30-n8n9n10--ecnu--20261118T165716294114+0800"
    for strat in ("N8", "N9", "N10"):
        ge2 = 0
        for pfile in base.rglob(f"*-direct-{strat}-r1/probes.jsonl"):
            valid = sum(1 for row in _read_jsonl(pfile) if isinstance(row.get("correct"), bool))
            if valid >= 2:
                ge2 += 1
        existing[strat]["questions_ge2_valid"] = ge2
        existing[strat]["total_questions"] = 30
    return existing


def _load_oracle5_summary() -> dict[str, Any]:
    report_dirs = [
        ROOT / "reports/ecnu-plus/longmemeval-oracle-5--ecnu--20261117T231654246753+0800",
        ROOT / "reports/ecnu-plus/longmemeval-oracle-5-n5n7--ecnu--20261118T151925052109+0800",
        ROOT / "reports/ecnu-plus/longmemeval-oracle-5-n8n9n10--ecnu--20261118T165630973039+0800",
        ROOT / "reports/ecnu-plus/longmemeval-oracle-5-n6-n11-n14--ecnu--20261125T103742269543+0800",
    ]
    by_strategy: dict[str, Any] = {}
    results_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for rdir in report_dirs:
        data = _read_json(rdir / "summary.json")
        for row in data.get("overview", []):
            by_strategy[row["strategy"]] = {
                "strategy": row["strategy"],
                "planned": row["planned"],
                "judged": row["judged"],
                "correct": row["correct"],
                "end_to_end_success": row["end_to_end_success"],
                "overflow": row.get("overflow", 0),
                "errors": row.get("errors", 0),
                "total_tokens": row.get("total_tokens", 0),
            }
        csv_path = rdir / "question-results.csv"
        if csv_path.exists():
            results_by_key.update(_load_question_csv(csv_path))

    oracle_items = _read_json(ROOT / "data/longmemeval/longmemeval_oracle.json")
    oracle_by_qid = {item["question_id"]: item for item in oracle_items}
    o5_manifest = _read_json(ROOT / "data/manifests/oracle-5.json")
    questions_meta: dict[str, Any] = {}
    for qid in o5_manifest["question_ids"]:
        adapted = adapt_longmemeval_item(oracle_by_qid[qid])
        questions_meta[qid] = {
            "question_id": qid,
            "question": adapted.question,
            "expected": adapted.answer,
            "question_type": adapted.question_type,
            "is_abstention": bool(adapted.is_abstention),
        }

    n7_probe = _read_json(
        ROOT / "reports/ecnu-plus/longmemeval-oracle-5-n7-probe--ecnu--20261118T153500890902+0800/summary.json"
    )
    return {
        "question_count": 5,
        "by_strategy": by_strategy,
        "questions": questions_meta,
        "results": list(results_by_key.values()),
        "n7_every_turn_note": {
            "batch_id": n7_probe["batch_id"],
            "correct": n7_probe["overview"][0]["correct"],
            "planned": n7_probe["overview"][0]["planned"],
            "probe_count": 58,
        },
    }


def _load_snapshot(run_dir: Path, turn_id: int) -> dict[str, Any] | None:
    snap_path = run_dir / "snapshots" / f"turn_{turn_id:03d}.json"
    if not snap_path.exists():
        return None
    return _read_json(snap_path)


def _snapshot_delta(prev_snap: dict[str, Any] | None, curr_snap: dict[str, Any] | None, record: dict[str, Any] | None) -> dict[str, Any] | None:
    if curr_snap is None:
        return None
    prev_kept = [int(k["turn_id"]) for k in (prev_snap or {}).get("kept_exchanges", [])]
    curr_kept = [int(k["turn_id"]) for k in curr_snap.get("kept_exchanges", [])]
    prev_kept_set = set(prev_kept)
    curr_kept_set = set(curr_kept)

    prev_arch = [int(x) for x in (prev_snap or {}).get("archived_turn_ids", [])]
    curr_arch = [int(x) for x in curr_snap.get("archived_turn_ids", [])]
    if not curr_arch and record and record.get("archived_turn_ids"):
        curr_arch = [int(x) for x in record["archived_turn_ids"]]
    prev_arch_set = set(prev_arch)
    curr_arch_set = set(curr_arch)

    prev_mem = (prev_snap or {}).get("memory_text") or ""
    curr_mem = curr_snap.get("memory_text") or ""
    return {
        "prev_kept_ids": prev_kept,
        "curr_kept_ids": curr_kept,
        "added_kept_ids": [t for t in curr_kept if t not in prev_kept_set],
        "dropped_kept_ids": [t for t in prev_kept if t not in curr_kept_set],
        "prev_archived_count": len(prev_arch),
        "curr_archived_count": len(curr_arch),
        "added_archive_ids": [t for t in curr_arch if t not in prev_arch_set],
        "dropped_archive_ids": [t for t in prev_arch if t not in curr_arch_set],
        "prev_mem_len": len(prev_mem),
        "curr_mem_len": len(curr_mem),
        "curr_mem_preview": _clip(curr_mem, 2400),
    }


def _format_event(ev: dict[str, Any]) -> dict[str, Any]:
    mech = ev.get("mechanism") or "unknown"
    before_lbl, after_lbl = EVENT_FIELD_LABELS.get(mech, ("操作前片段", "操作后片段"))
    return {
        "event_id": ev.get("event_id"),
        "turn_id": ev.get("turn_id"),
        "phase": ev.get("phase"),
        "mechanism": mech,
        "trigger": ev.get("trigger"),
        "input_tokens": ev.get("input_tokens"),
        "output_tokens": ev.get("output_tokens"),
        "latency_ms": ev.get("latency_ms"),
        "evicted_turn_ids": ev.get("evicted_turn_ids") or [],
        "archive_evicted_turn_ids": ev.get("archive_evicted_turn_ids") or [],
        "before_label": before_lbl,
        "after_label": after_lbl,
        "before_text": _clip(ev.get("before_text"), 3500),
        "after_text": _clip(ev.get("after_text"), 3500),
        "metrics": ev.get("metrics") or {},
    }


def _enrich_audit_row(row: dict[str, Any], plan_dir: Path) -> dict[str, Any]:
    qid = row["question_id"]
    strat = row["strategy"]
    recheck_files = sorted((plan_dir / "rechecks" / qid / strat).glob("*.jsonl")) if (plan_dir / "rechecks" / qid / strat).exists() else []
    rechecks_summary: list[dict[str, Any]] = []
    instability_turns: list[int] = []
    for rf in recheck_files:
        for rec in _read_jsonl(rf):
            cls = rec.get("recheck_classification")
            rechecks_summary.append({
                "turn_id": int(rec["turn_id"]),
                "classification": cls,
                "primary_correct": rec.get("primary_correct"),
                "recheck_correct": rec.get("correct"),
            })
            if cls and cls != "consistent":
                instability_turns.append(int(rec["turn_id"]))

    cf_files = sorted((plan_dir / "counterfactuals" / qid / strat).glob("*.json")) if (plan_dir / "counterfactuals" / qid / strat).exists() else []
    cf_summary: list[dict[str, Any]] = []
    for cf_path in cf_files:
        cf = _read_json(cf_path)
        cf_summary.append({
            "turn_id": int(cf["turn_id"]),
            "fits_budget": bool(cf.get("fits_budget")),
            "correct": cf.get("correct"),
            "estimated_tokens": cf.get("estimated_tokens"),
            "budget_cap_tokens": cf.get("budget_cap_tokens"),
        })

    row_copy = dict(row)
    row_copy["recheck_summary"] = rechecks_summary
    row_copy["instability_turns"] = sorted(set(instability_turns))
    row_copy["cf_summary"] = cf_summary
    row_copy["has_inbudget_cf_win"] = any(c["fits_budget"] and c["correct"] is True for c in cf_summary)
    return row_copy


def _build_case_file(
    plan_key: str,
    plan: dict[str, Any],
    plan_dir: Path,
    qid: str,
    raw_item: dict[str, Any],
    audit_by_key: dict[tuple[str, str], dict[str, Any]],
    csv_by_key: dict[tuple[str, str], dict[str, Any]],
    ev_traj_by_qid: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    qmeta = plan["questions"][qid]
    adapted = adapt_longmemeval_item(raw_item)
    exchanges = adapted.turns
    ex_by_tid = {ex.turn_id: ex for ex in exchanges}
    raw_has_ans = _has_answer_turn_ids(raw_item)

    probed_turns = _resolve_turns(qmeta, len(exchanges))
    needed_turns: set[int] = set(probed_turns) | (raw_has_ans & set(ex_by_tid.keys()))

    et_item = ev_traj_by_qid.get(qid)
    evidence_turn_ids: list[int] = sorted(raw_has_ans)
    if et_item:
        for u in et_item.get("units", []):
            if u.get("turn_id") is not None:
                evidence_turn_ids.append(int(u["turn_id"]))
                needed_turns.add(int(u["turn_id"]))
    needed_turns = {t for t in needed_turns if t in ex_by_tid}

    strategies_payload: dict[str, Any] = {}
    for strat in plan["strategies"]:
        batch_id = _resolve_source_batch(plan, qid, strat)
        run_dir = _resolve_run_dir(plan, qid, strat)
        run_summary = _read_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
        events_by_turn: dict[int, list[dict[str, Any]]] = {}
        for ev in _read_jsonl(run_dir / "events.jsonl"):
            tid = int(ev["turn_id"])
            events_by_turn.setdefault(tid, []).append(_format_event(ev))

        rechecks_by_turn: dict[int, list[dict[str, Any]]] = {}
        rc_dir = plan_dir / "rechecks" / qid / strat
        if rc_dir.exists():
            for rf in sorted(rc_dir.glob("*.jsonl")):
                for r in _read_jsonl(rf):
                    rechecks_by_turn.setdefault(int(r["turn_id"]), []).append({
                        "turn_id": int(r["turn_id"]),
                        "classification": r.get("recheck_classification"),
                        "primary_correct": r.get("primary_correct"),
                        "correct": r.get("correct"),
                        "judge_raw": r.get("judge_raw"),
                        "hypothesis": r.get("hypothesis"),
                    })

        cfs_by_turn: dict[int, dict[str, Any]] = {}
        cf_dir = plan_dir / "counterfactuals" / qid / strat
        if cf_dir.exists():
            for cf_path in sorted(cf_dir.glob("*.json")):
                cf = _read_json(cf_path)
                tid = int(cf["turn_id"])
                cfs_by_turn[tid] = {
                    "turn_id": tid,
                    "fits_budget": bool(cf.get("fits_budget")),
                    "estimated_tokens": cf.get("estimated_tokens"),
                    "budget_cap_tokens": cf.get("budget_cap_tokens"),
                    "correct": cf.get("correct"),
                    "judge_raw": cf.get("judge_raw"),
                    "hypothesis": cf.get("hypothesis"),
                    "retrieved_turn_ids": cf.get("retrieved_turn_ids") or [],
                    "memory_text_injected": _clip(cf.get("memory_text_injected"), 4000),
                    "request_sha256": cf.get("request_sha256"),
                }
                needed_turns.update(int(x) for x in cfs_by_turn[tid]["retrieved_turn_ids"] if int(x) in ex_by_tid)

        # Build turn-by-turn records + diffs
        turns_data: dict[str, Any] = {}
        last_snap_turn = None
        snaps_dir = run_dir / "snapshots"
        if snaps_dir.exists():
            snap_files = sorted(snaps_dir.glob("turn_*.json"))
            if snap_files:
                last_snap_turn = int(snap_files[-1].stem.split("_")[1])

        for tid in probed_turns:
            rec_path = plan_dir / "records" / qid / strat / f"{tid:03d}.json"
            rec: dict[str, Any] | None = _read_json(rec_path) if rec_path.exists() else None
            if rec is None:
                _, rec = _find_saved_record(plan_dir, plan, qid, strat, tid)
            curr_snap = _load_snapshot(run_dir, tid)
            prev_snap = _load_snapshot(run_dir, tid - 1) if tid > 1 else None
            t_evs = events_by_turn.get(tid, [])

            if rec is None or rec.get("status") != "ok":
                gap_reason = (rec or {}).get("status") or "api_error_gap"
                if gap_reason not in ("termination_turn_no_snapshot", "post_termination") and curr_snap is None:
                    if last_snap_turn is not None and tid == last_snap_turn + 1:
                        gap_reason = "termination_turn_no_snapshot"
                    else:
                        gap_reason = "post_termination"
                turns_data[str(tid)] = {
                    "turn_id": tid,
                    "status": gap_reason,
                    "termination_reason": (rec or {}).get("termination_reason") or run_summary.get("termination_reason"),
                    "events": t_evs,
                }
                continue

            ret_ids = [int(x) for x in (rec.get("retrieved_turn_ids") or [])]
            arch_ids = [int(x) for x in (rec.get("archived_turn_ids") or [])]
            delta = _snapshot_delta(prev_snap, curr_snap, rec)
            if delta:
                needed_turns.update(t for t in delta["added_kept_ids"] + delta["dropped_kept_ids"] + delta["curr_kept_ids"][:3] if t in ex_by_tid)
            needed_turns.update(t for t in ret_ids if t in ex_by_tid)
            for ev in t_evs:
                needed_turns.update(int(x) for x in ev["evicted_turn_ids"] + ev["archive_evicted_turn_ids"] if int(x) in ex_by_tid)

            turns_data[str(tid)] = {
                "turn_id": tid,
                "status": "ok",
                "correct": rec.get("correct"),
                "judge_raw": rec.get("judge_raw"),
                "hypothesis": rec.get("hypothesis"),
                "retrieved_turn_ids": ret_ids,
                "archived_turn_ids": arch_ids,
                "memory_text_injected": _clip(rec.get("memory_text_injected"), 4500),
                "probe_tokens_in": rec.get("probe_tokens_in"),
                "request_sha256": rec.get("request_sha256"),
                "source_batch": rec.get("source_batch"),
                "snapshot_diff": delta,
                "events": t_evs,
                "rechecks": rechecks_by_turn.get(tid, []),
                "counterfactual": cfs_by_turn.get(tid),
            }

        audit_row = audit_by_key.get((qid, strat), {})
        csv_row = csv_by_key.get((qid, strat), {})
        strategies_payload[strat] = {
            "strategy": strat,
            "source_batch": batch_id,
            "run_id": run_summary.get("run_id") or audit_row.get("run_id"),
            "run_status": run_summary.get("status") or audit_row.get("run_status"),
            "termination_reason": run_summary.get("termination_reason"),
            "direct_final": {
                "hypothesis": run_summary.get("hypothesis"),
                "judge_label": csv_row.get("judge_label"),
                "status": csv_row.get("status") or run_summary.get("status"),
                "outcome_category": csv_row.get("outcome_category"),
                "final_ctx_tokens_est": csv_row.get("final_ctx_tokens_est") or run_summary.get("final_ctx_tokens_est"),
                "maintenance_events": csv_row.get("maintenance_events") or run_summary.get("maintenance_events"),
                "total_tokens": csv_row.get("total_tokens"),
                "final_evidence_retention": csv_row.get("final_evidence_retention"),
            },
            "audit": {
                "valid_observations": audit_row.get("valid_observations"),
                "first_observed_error_turn": audit_row.get("first_observed_error_turn"),
                "first_observation_incorrect": audit_row.get("first_observation_incorrect"),
                "first_correct_turn": audit_row.get("first_correct_turn"),
                "last_ok_turn_before_regression": audit_row.get("last_ok_turn_before_regression"),
                "first_regression_turn": audit_row.get("first_regression_turn"),
                "regression_interval": audit_row.get("regression_interval"),
                "bridged_api_error_gaps": audit_row.get("bridged_api_error_gaps") or [],
                "recovered_turns": audit_row.get("recovered_turns") or [],
                "termination_gaps": audit_row.get("termination_gaps") or 0,
                "causal_framing": audit_row.get("causal_framing"),
                "interval_events_count": len(audit_row.get("interval_events") or []),
                "recheck_summary": audit_row.get("recheck_summary") or [],
                "instability_turns": audit_row.get("instability_turns") or [],
                "cf_summary": audit_row.get("cf_summary") or [],
            },
            "turns": turns_data,
        }

    exchanges_payload: dict[str, Any] = {}
    for tid in sorted(needed_turns):
        ex = ex_by_tid[tid]
        exchanges_payload[str(tid)] = {
            "turn_id": ex.turn_id,
            "session_id": ex.session_id,
            "timestamp": ex.timestamp,
            "user_text": _clip(ex.user_text, 2200),
            "assistant_text": _clip(ex.assistant_text, 2200),
            "has_answer": tid in raw_has_ans,
        }

    case_obj = {
        "plan_key": plan_key,
        "question_id": qid,
        "question": adapted.question,
        "question_date": adapted.question_date,
        "expected": adapted.answer,
        "question_type": adapted.question_type,
        "is_abstention": bool(adapted.is_abstention),
        "total_turns": len(exchanges),
        "probed_turns": probed_turns,
        "validity_note": qmeta.get("validity_note"),
        "dev_exposed": qid in DEV_EXPOSED_QIDS,
        "evidence_turn_ids": sorted(set(evidence_turn_ids)),
        "evidence_meta": (
            {
                "has_answer_status": et_item.get("has_answer_status"),
                "message_level_denominator": et_item.get("message_level_denominator"),
                "session_level_denominator": et_item.get("session_level_denominator"),
            }
            if et_item
            else None
        ),
        "exchanges": exchanges_payload,
        "strategies": strategies_payload,
    }

    case_key = f"{plan_key}__{qid}"
    out_file = out_dir / "cases" / f"{case_key}.js"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    js_body = f'window.__SHOWCASE_CASES__ = window.__SHOWCASE_CASES__ || {{}};\nwindow.__SHOWCASE_CASES__[{json.dumps(case_key)}] = {json.dumps(case_obj, ensure_ascii=False)};\n'
    out_file.write_text(js_body, encoding="utf-8")


def _build_plan_checklist(plan_dir: Path) -> dict[str, int]:
    vc = _read_json(plan_dir / "verification_checklist.json")
    recs = []
    for rf in (plan_dir / "rechecks").glob("*/*/*.jsonl"):
        for line in rf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                recs.append(json.loads(line))
    cfs = [json.loads(cf.read_text(encoding="utf-8")) for cf in (plan_dir / "counterfactuals").glob("*/*/*.json")]
    rec_consistent = sum(1 for r in recs if r.get("recheck_classification") == "consistent")
    rec_inst = sum(1 for r in recs if r.get("recheck_classification") == "readout_instability")
    cf_inbudget_ok = sum(1 for c in cfs if c.get("fits_budget") and c.get("correct"))
    cf_over = sum(1 for c in cfs if c.get("fits_budget") is False)
    return {
        "total": int(vc["total_cells"]),
        "reusable_ok": int(vc["reusable_ok"]),
        "pending_new": int(vc["pending_new"]),
        "termination_gaps": int(vc["termination_gaps_total"]),
        "termination_turn_no_snapshot": int(vc["termination_turn_no_snapshot"]),
        "post_termination": int(vc["post_termination"]),
        "unexpected_missing": int(vc["unexpected_missing"]),
        "mismatch_error": int(vc["mismatch_error"]),
        "rechecks_total": len(recs),
        "rechecks_consistent": rec_consistent,
        "rechecks_instability": rec_inst,
        "cf_total": len(cfs),
        "cf_inbudget_ok": cf_inbudget_ok,
        "cf_over_budget": cf_over,
    }


def build_showcase(out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir or (ROOT / "showcase")
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_items = _read_json(ROOT / "data/longmemeval/longmemeval_s_cleaned.json")
    raw_by_qid = {item["question_id"]: item for item in dataset_items}

    sdev_report_dir = ROOT / "reports/ecnu-plus/longmemeval-s-dev-30-merged15--ecnu--20261124T150000000000+0800"
    sku_report_dir = ROOT / "reports/ecnu-plus/longmemeval-s-ku-10-v1-merged15--ecnu--20261124T080000000000+0800"

    sdev_summary = _read_json(sdev_report_dir / "summary.json")
    sku_summary = _read_json(sku_report_dir / "summary.json")
    sdev_csv = _load_question_csv(sdev_report_dir / "question-results.csv")
    sku_csv = _load_question_csv(sku_report_dir / "question-results.csv")
    et_list = _read_json(sku_report_dir / "evidence_trajectory.json")
    ev_traj_by_qid = {item["question_id"]: item for item in et_list}

    sdev_online = _compute_sdev30_online_probes()
    oracle5 = _load_oracle5_summary()

    plan_a_dir = ROOT / "reports/ecnu-plus/s-ku-10-v1-probe-plan"
    plan_b_dir = ROOT / "reports/ecnu-plus/s-dev-30-reversal-probe-plan"
    plan_a = _read_json(plan_a_dir / "plan.json")
    plan_b = _read_json(plan_b_dir / "plan.json")
    audit_a = _read_json(plan_a_dir / "audit_review.json")
    audit_b = _read_json(plan_b_dir / "audit_review.json")

    audit_a_rows = [_enrich_audit_row(r, plan_a_dir) for r in audit_a["rows"]]
    audit_b_rows = [_enrich_audit_row(r, plan_b_dir) for r in audit_b["rows"]]
    audit_a_by_key = {(r["question_id"], r["strategy"]): r for r in audit_a_rows}
    audit_b_by_key = {(r["question_id"], r["strategy"]): r for r in audit_b_rows}

    sdev_ov_map = {r["strategy"]: r for r in sdev_summary.get("overview", [])}
    sku_ov_map = {r["strategy"]: r for r in sku_summary.get("overview", [])}

    strategies_enriched = []
    for spec in STRATEGY_SPECS:
        sid = spec["id"]
        entry = dict(spec)
        entry["s_dev_30"] = sdev_ov_map.get(sid)
        entry["s_ku_10"] = sku_ov_map.get(sid)
        entry["oracle_5"] = oracle5["by_strategy"].get(sid)
        entry["s_dev_30_online"] = sdev_online.get(sid)
        strategies_enriched.append(entry)

    # Build question lists for Plan A and Plan B
    plan_a_questions: dict[str, Any] = {}
    for qid, qmeta in plan_a["questions"].items():
        adapted = adapt_longmemeval_item(raw_by_qid[qid])
        plan_a_questions[qid] = {
            "question_id": qid,
            "question": adapted.question,
            "expected": adapted.answer,
            "question_type": adapted.question_type,
            "first_turn": qmeta["first_turn"],
            "last_turn": qmeta["last_turn"],
            "turn_count": qmeta["last_turn"] - qmeta["first_turn"] + 1,
            "validity_note": qmeta.get("validity_note"),
            "dev_exposed": qid in DEV_EXPOSED_QIDS,
        }

    plan_b_questions: dict[str, Any] = {}
    for qid, qmeta in plan_b["questions"].items():
        adapted = adapt_longmemeval_item(raw_by_qid[qid])
        plan_b_questions[qid] = {
            "question_id": qid,
            "question": adapted.question,
            "expected": adapted.answer,
            "question_type": adapted.question_type,
            "is_abstention": bool(adapted.is_abstention),
            "turns": qmeta["turns"],
            "turn_count": len(qmeta["turns"]),
            "validity_note": qmeta.get("validity_note"),
        }

    # Build s-dev-30 30-question metadata + results matrix
    sdev_qids_order = sorted({qid for (qid, _) in sdev_csv.keys()})
    sdev_questions_meta: dict[str, Any] = {}
    for qid in sdev_qids_order:
        adapted = adapt_longmemeval_item(raw_by_qid[qid])
        sdev_questions_meta[qid] = {
            "question_id": qid,
            "question": adapted.question,
            "expected": adapted.answer,
            "question_type": adapted.question_type,
            "is_abstention": bool(adapted.is_abstention),
            "in_plan_b": qid in plan_b_questions,
        }

    featured_cases = [
        {
            "tag": "跳过维护直接因果配对",
            "title": "89941a93 · N9 · T185（自行车数量：4 → 2）",
            "plan_key": "plan_a",
            "question_id": "89941a93",
            "strategy": "N9",
            "turn": 185,
            "summary": "跳过本轮维护（7304≤8192）仍包含 T155 四辆车清单答对 4；执行维护淘汰 T154–T156 后仅剩两辆车答错为 2。",
        },
        {
            "tag": "跳过维护直接因果配对",
            "title": "4fd1909e · N9 · T130（演唱会地点：Xfinity Center → 未知）",
            "plan_key": "plan_b",
            "question_id": "4fd1909e",
            "strategy": "N9",
            "turn": 130,
            "summary": "s-dev-30 Plan B 典型反转：跳过维护（7320≤8192）答对 Xfinity Center；执行维护后检索 [120..124] 答未知。",
        },
        {
            "tag": "跳过维护直接因果配对",
            "title": "07b6f563 · N9 · T216（手机配件：iPhone 13 Pro 匹配丢失）",
            "plan_key": "plan_b",
            "question_id": "07b6f563",
            "strategy": "N9",
            "turn": 216,
            "summary": "跳过维护（6753≤8192）命中 T204 的 iPhone 13 Pro（yes）；维护后 T204 滑出，仅泛泛推荐配件（no）。",
        },
        {
            "tag": "BM25 排序漂移导致历史依据失落",
            "title": "e66b632c · N7 · T251→T252（5K 上次个人最佳 27:45 → 错算 28:00）",
            "plan_key": "plan_a",
            "question_id": "e66b632c",
            "strategy": "N7",
            "turn": 252,
            "summary": "T251 检索 [17,19,20] 找回 27:45；T252 因 T246 入池挤出 T17，模型反算 26:30+1:30 误答 28:00。",
        },
        {
            "tag": "边界读出不稳定实证",
            "title": "4d6b87c8 · N9/N10 · T232→T234（待看清单 25 vs 计划新增 27）",
            "plan_key": "plan_a",
            "question_id": "4d6b87c8",
            "strategy": "N9",
            "turn": 233,
            "summary": "--recheck 检出 3 例 readout_instability：同一冻结请求在“当前 25”与“拟新增后 27”间摇摆。",
        },
        {
            "tag": "拒答样本对遗忘不敏感对照",
            "title": "bc8a6e93_abs · N6, N8–N14（有效检查点全对）",
            "plan_key": "plan_b",
            "question_id": "bc8a6e93_abs",
            "strategy": "N9",
            "turn": 248,
            "summary": "拒答题标准答案为“未提供”，信息被淘汰后模型依然拒答，解释为何该题在回放中无退化。",
        },
    ]

    overview_payload = {
        "generated_at": datetime.now(TZ_CN).isoformat(),
        "model": "ecnu-plus",
        "judge_model": "ecnu-plus",
        "budget": {
            "total_budget": 8192,
            "trigger_tokens": 6553,
            "target_tokens": 4915,
            "memory_cap": 2048,
            "archive_cap": 16384,
            "n1_context_tokens": 131072,
        },
        "strategies": strategies_enriched,
        "benchmarks": {
            "s_dev_30": {
                "batch_id": sdev_summary["batch_id"],
                "question_count": 30,
                "strategies_executed": len(sdev_summary["overview"]),
                "total_cells": sum(int(r["planned"]) for r in sdev_summary["overview"]),
                "scored_cells": sum(int(r["judged"]) for r in sdev_summary["overview"]),
                "overflow_cells": sum(int(r.get("overflow") or 0) for r in sdev_summary["overview"]),
                "error_cells": sum(int(r.get("errors") or 0) for r in sdev_summary["overview"]),
                "scoring_pending": sum(int(r.get("scoring_pending") or 0) for r in sdev_summary["overview"]),
                "overview": sdev_summary["overview"],
                "comparisons": sdev_summary["comparisons"],
                "online_probes": sdev_online,
                "questions": sdev_questions_meta,
                "results": list(sdev_csv.values()),
            },
            "s_ku_10": {
                "batch_id": sku_summary["batch_id"],
                "question_count": 10,
                "strategies_executed": len(sku_summary["overview"]),
                "overview": sku_summary["overview"],
                "comparisons": sku_summary["comparisons"],
                "evidence_trajectories": [
                    {
                        "question_id": item["question_id"],
                        "question_type": item["question_type"],
                        "message_level_denominator": item["message_level_denominator"],
                        "session_level_denominator": item["session_level_denominator"],
                        "has_answer_status": item["has_answer_status"],
                    }
                    for item in et_list
                ],
            },
            "oracle_5": oracle5,
        },
        "plans": {
            "plan_a": {
                "plan_id": plan_a["plan_id"],
                "title": f"Plan A · s-ku-10-v1 全量尾段冻结回放（10 题 × {len(plan_a['strategies'])} 策略）",
                "selection_rule": plan_a["selection_rule"],
                "checklist": _build_plan_checklist(plan_a_dir),
                "by_strategy": audit_a["by_strategy"],
                "questions": plan_a_questions,
                "rows": audit_a_rows,
            },
            "plan_b": {
                "plan_id": plan_b["plan_id"],
                "title": f"Plan B · s-dev-30 典型反转共同检查点回放（7 题 × N6, N8–N14）",
                "selection_rule": plan_b["selection_rule"],
                "checklist": _build_plan_checklist(plan_b_dir),
                "by_strategy": audit_b["by_strategy"],
                "questions": plan_b_questions,
                "rows": audit_b_rows,
            },
        },
        "featured_cases": featured_cases,
    }

    overview_file = data_dir / "overview.js"
    overview_file.write_text(
        f"window.__SHOWCASE_OVERVIEW__ = {json.dumps(overview_payload, ensure_ascii=False)};\n",
        encoding="utf-8",
    )

    built_cases: list[str] = []
    for plan_key, plan, plan_dir, audit_map, csv_map in [
        ("plan_a", plan_a, plan_a_dir, audit_a_by_key, sku_csv),
        ("plan_b", plan_b, plan_b_dir, audit_b_by_key, sdev_csv),
    ]:
        for qid in plan["questions"]:
            _build_case_file(
                plan_key=plan_key,
                plan=plan,
                plan_dir=plan_dir,
                qid=qid,
                raw_item=raw_by_qid[qid],
                audit_by_key=audit_map,
                csv_by_key=csv_map,
                ev_traj_by_qid=ev_traj_by_qid if plan_key == "plan_a" else {},
                out_dir=data_dir,
            )
            built_cases.append(f"{plan_key}__{qid}")

    return {
        "overview_file": str(overview_file),
        "cases_count": len(built_cases),
        "cases": built_cases,
    }


def main() -> None:
    res = build_showcase()
    print(f"Built showcase overview: {res['overview_file']}")
    print(f"Built {res['cases_count']} case packs: {', '.join(res['cases'])}")


if __name__ == "__main__":
    main()
