"""Data-driven LongMemEval evidence trajectory exporter across strategies.

Design guarantees:
1. Strictly separates message-level `has_answer` evidence (preserving original source_message_indices)
   from coarse session-level `answer_session_ids` proxy, with explicit separate denominators.
   Missing message-level annotations are reported as missing (never silently replaced by whole sessions).
2. Separately tracks three storage layers at every observed checkpoint:
   - recent_window (`kept_exchanges`)
   - external_storage (`archived_turn_ids` / `memory_text`; marked N/A for N0/N1/N2/N5)
   - readout (`retrieved_turn_ids` or `memory_text_injected` when a probe exists at that turn)
3. For N11 (dedup_fifo), reports both `source_turn_retained` and `content_retained` (preserving the
   newest surviving turn_id and timestamp that holds identical normalized content).
4. Strictly reports observed intervals `(last_observed_retained_turn, first_observed_missing_turn]`
   using actual existing snapshot turns (never fabricating `first_missing_turn - 1`).
5. Only attributes a loss to an event when the event actually explains the corresponding storage
   layer transitioning from present to absent; preserves uncertainty when multiple events or
   LLM summary paraphrasing (N4) prevent deterministic attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from budget_executor import ARCHIVE_ONLY_STRATEGIES, ARCHIVE_STRATEGIES, FACT_STRATEGIES  # noqa: E402
from dataset_adapters import align_longmemeval_turns  # noqa: E402
from metrics import normalize_text  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def extract_longmemeval_evidence_units(raw_item: dict[str, Any]) -> dict[str, Any]:
    """Separate message-level `has_answer` evidence turns from session-level `answer_session_ids`."""
    turns = align_longmemeval_turns(raw_item)
    msg_has_answer: set[int] = set()
    idx = 0
    for sess in raw_item.get("haystack_sessions") or []:
        for msg in sess:
            if msg.get("has_answer") is True:
                msg_has_answer.add(idx)
            idx += 1

    msg_level_units: list[dict[str, Any]] = []
    for t in turns:
        hit_indices = sorted(msg_has_answer.intersection(t.source_message_indices))
        if hit_indices:
            msg_level_units.append({
                "turn_id": t.turn_id,
                "session_id": t.session_id,
                "timestamp": t.timestamp,
                "source_message_indices": hit_indices,
                "norm_payload": normalize_text(f"{t.user_text}\n{t.assistant_text or ''}"),
                "user_preview": (t.user_text or "")[:100],
            })

    ans_sids = set(raw_item.get("answer_session_ids") or [])
    sess_level_turns = [t.turn_id for t in turns if t.session_id in ans_sids]

    return {
        "total_turns": len(turns),
        "turns_by_id": {t.turn_id: t for t in turns},
        "has_answer_status": "annotated" if msg_level_units else "missing",
        "message_level_units": msg_level_units,
        "message_level_denominator": len(msg_has_answer),
        "evidence_exchange_count": len(msg_level_units),
        "session_level_turn_ids": sess_level_turns,
        "session_level_denominator": len(sess_level_turns),
    }


def _inspect_unit_at_snapshot(
    strategy: str,
    snap: dict[str, Any],
    unit: dict[str, Any],
    turns_by_id: dict[int, Any],
    probe: dict[str, Any] | None,
) -> dict[str, Any]:
    ev_tid = unit["turn_id"]
    kept_ids = {int(k["turn_id"]) for k in snap.get("kept_exchanges", [])}
    in_recent = ev_tid in kept_ids

    mem_text = snap.get("memory_text") or ""
    arch_ids = [int(x) for x in (snap.get("archived_turn_ids") or [])]
    arch_set = set(arch_ids)

    ext_applicable = strategy not in ("N0", "N1", "N2", "N5")
    in_external: bool | str | None
    content_retained: dict[str, Any] | None = None

    if not ext_applicable:
        in_external = "N/A"
    elif strategy in ARCHIVE_ONLY_STRATEGIES:
        in_external = ev_tid in arch_set
        if strategy == "N11":
            # Distinguish source_turn_id retention vs deduplicated content retention with timestamp/source
            if ev_tid in arch_set:
                content_retained = {"retained": True, "via_turn_id": ev_tid, "timestamp": unit["timestamp"]}
            else:
                match_tid = None
                for atid in reversed(arch_ids):
                    at = turns_by_id.get(atid)
                    if at and normalize_text(f"{at.user_text}\n{at.assistant_text or ''}") == unit["norm_payload"]:
                        match_tid = atid
                        break
                if match_tid is not None:
                    mt = turns_by_id[match_tid]
                    content_retained = {"retained": True, "via_turn_id": match_tid, "timestamp": mt.timestamp}
                else:
                    content_retained = {"retained": False, "via_turn_id": None, "timestamp": None}
    elif strategy == "N13":
        # Hybrid: external storage includes both structured fact table [T<id>] and bounded archive whitelist
        in_fact = bool(re.search(rf"\[T{ev_tid}\b", mem_text))
        in_arch = ev_tid in arch_set
        in_external = in_fact or in_arch
        content_retained = {"in_fact_table": in_fact, "in_archive": in_arch}
    elif strategy in ("N3", "N9", "N10", "N14"):
        in_external = bool(re.search(rf"\[T{ev_tid}\b", mem_text))
    elif strategy in ("N4", "N6"):
        # LLM summaries paraphrase text; cannot assert semantic loss purely from missing [T<id>] tag
        tag_hit = bool(re.search(rf"\bT{ev_tid}\b", mem_text))
        in_external = "tag_present" if tag_hit else "uncertain_llm_paraphrase"
    else:
        in_external = None

    # Readout layer (only observed if a probe exists at this checkpoint)
    if probe is None:
        in_readout: bool | str = "unobserved"
    else:
        ret_ids = set(int(x) for x in (probe.get("retrieved_turn_ids") or []))
        if strategy in ARCHIVE_STRATEGIES or strategy in FACT_STRATEGIES:
            in_readout = in_recent or (ev_tid in ret_ids)
        else:
            in_readout = in_recent or (in_external is True) or (in_external == "tag_present")

    return {
        "in_recent_window": in_recent,
        "in_external_storage": in_external,
        "in_probe_readout": in_readout,
        "probe_correct": probe.get("correct") if probe is not None else None,
        "probe_origin": probe.get("origin", "online") if probe is not None else None,
        "content_detail": content_retained,
    }


def _trace_unit_loss_interval(
    strategy: str,
    run_dir: Path,
    unit: dict[str, Any],
    turns_by_id: dict[int, Any],
) -> dict[str, Any]:
    """Scan all actually saved snapshots to find observed retention/loss intervals and verify attributing events."""
    snap_files = sorted((run_dir / "snapshots").glob("turn_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    evs = _load_jsonl(run_dir / "events.jsonl")
    ev_tid = unit["turn_id"]

    seen_retained = False
    last_observed_retained: int | None = None
    first_observed_missing: int | None = None
    loss_layer: str | None = None

    for sf in snap_files:
        m = re.match(r"turn_(\d+)\.json$", sf.name)
        if not m:
            continue
        tid = int(m.group(1))
        if tid < ev_tid:
            continue
        snap = _load_json(sf)
        st = _inspect_unit_at_snapshot(strategy, snap, unit, turns_by_id, None)
        ext = st["in_external_storage"]
        present = st["in_recent_window"] or (ext is True) or (ext == "tag_present")
        uncertain = (ext == "uncertain_llm_paraphrase") and not st["in_recent_window"]
        if present:
            seen_retained = True
            last_observed_retained = tid
        elif uncertain:
            # Do not claim deterministic loss on LLM summary paraphrasing
            if seen_retained and first_observed_missing is None:
                loss_layer = "uncertain_llm_paraphrase"
        else:
            if seen_retained and first_observed_missing is None:
                first_observed_missing = tid
                loss_layer = "recent_window" if strategy in ("N0", "N2") else "source_no_longer_observed"
                break

    # Strict event attribution: only attribute if event explains the corresponding layer going present -> absent
    candidate_events: list[str] = []
    if last_observed_retained is not None and first_observed_missing is not None:
        for e in evs:
            et = int(e.get("turn_id", 0))
            if not (last_observed_retained < et <= first_observed_missing):
                continue
            mech = e.get("mechanism", "")
            if strategy in ("N0", "N2") and ev_tid in (e.get("evicted_turn_ids") or []):
                candidate_events.append(e["event_id"])
            elif strategy in ("N8", "N11", "N12") and ev_tid in (e.get("archive_evicted_turn_ids") or []):
                candidate_events.append(e["event_id"])
            elif strategy == "N13":
                in_arch_drop = ev_tid in (e.get("archive_evicted_turn_ids") or [])
                if in_arch_drop:
                    candidate_events.append(e["event_id"])
            elif strategy in ("N3", "N9", "N10", "N14"):
                if re.search(rf"\[T{ev_tid}\b", e.get("before_text") or "") and not re.search(rf"\[T{ev_tid}\b", e.get("after_text") or ""):
                    candidate_events.append(e["event_id"])

    if first_observed_missing is None:
        if loss_layer == "uncertain_llm_paraphrase":
            attribution = "uncertain_llm_paraphrase"
        elif seen_retained:
            attribution = "retained_through_last_snapshot"
        else:
            attribution = "never_observed_in_snapshots"
    elif len(candidate_events) == 1:
        attribution = f"attributed:{candidate_events[0]}"
    elif len(candidate_events) > 1:
        attribution = f"ambiguous_multiple_events:{candidate_events}"
    else:
        attribution = "unattributed_in_interval"

    return {
        "last_observed_retained_turn": last_observed_retained,
        "first_observed_missing_turn": first_observed_missing,
        "observed_loss_interval": (
            f"(T{last_observed_retained}, T{first_observed_missing}]"
            if (last_observed_retained is not None and first_observed_missing is not None)
            else None
        ),
        "loss_layer": loss_layer,
        "attribution": attribution,
    }


def main():
    parser = argparse.ArgumentParser(description="Export data-driven LongMemEval evidence trajectory table.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--question-ids", default=None, help="Comma-separated question_ids (defaults to all in batch)")
    parser.add_argument("--strategies", default=None, help="Comma-separated strategies (defaults to all in batch)")
    parser.add_argument("--checkpoints", default=None, help="Optional comma-separated turn_ids; defaults to evidence turns + final turn")
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_absolute():
        batch_dir = ROOT / args.batch_dir
    ds_path = Path(args.dataset)
    if not ds_path.is_absolute():
        ds_path = ROOT / args.dataset

    with ds_path.open("rb") as source:
        dataset_hash = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        ds_items = {x["question_id"]: x for x in json.load(source)}
    batch_summary = _load_json(batch_dir / "batch_summary.json")
    runs_by_key = {
        (strategy, row["question_id"]): row["run_id"]
        for strategy, rows in batch_summary["strategies"].items() for row in rows
    }
    all_strats = list((batch_summary.get("strategies") or {}).keys())
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()] if args.strategies else all_strats
    qids = (
        [q.strip() for q in args.question_ids.split(",") if q.strip()]
        if args.question_ids
        else [r["question_id"] for r in (batch_summary["strategies"][strategies[0]])]
    )

    for run_id in runs_by_key.values():
        manifest = _load_json(batch_dir / "runs" / run_id / "manifest.json")
        if manifest["dataset"].get("file_sha256") != dataset_hash:
            raise SystemExit(f"Dataset SHA-256 mismatch or missing for {run_id}")
    output_records: list[dict[str, Any]] = []
    md_lines = [
        f"# LongMemEval 跨策略逐证据维护轨迹表（{batch_dir.name}）",
        "",
        "- **数据口径**：严格区分消息级 `has_answer`（保留原始 `source_message_indices`）与会话级 `answer_session_ids` 代理，分别给出分母；无消息标注时标为 `missing`。",
        "- **存储三层分离**：每检查点分列「近期窗口 / 外部存储 / 该次探针读出」；区间严格为实际观测快照 `(last_observed_retained_turn, first_observed_missing_turn]`。",
        "",
    ]

    for qid in qids:
        raw_item = ds_items[qid]
        ev_info = extract_longmemeval_evidence_units(raw_item)
        cp_turns = (
            [int(x.strip()) for x in args.checkpoints.split(",") if x.strip()]
            if args.checkpoints
            else sorted({u["turn_id"] for u in ev_info["message_level_units"]} | {ev_info["total_turns"]})
        )

        q_record = {
            "question_id": qid,
            "dataset_sha256": dataset_hash,
            "question_type": raw_item.get("question_type"),
            "question": raw_item.get("question"),
            "answer": raw_item.get("answer"),
            "has_answer_status": ev_info["has_answer_status"],
            "message_level_denominator": ev_info["message_level_denominator"],
            "evidence_exchange_count": ev_info["evidence_exchange_count"],
            "session_level_denominator": ev_info["session_level_denominator"],
            "checkpoints": cp_turns,
            "units": [],
        }

        md_lines.extend([
            f"## 题目 `{qid}`（{raw_item.get('question_type')}，共 {ev_info['total_turns']} 轮）",
            "",
            f"- **问题**：{raw_item.get('question')} | **终态参考答案**：`{raw_item.get('answer')}`",
            f"- **证据分母**：消息级 `has_answer` = **{ev_info['message_level_denominator']}**（映射到 {ev_info['evidence_exchange_count']} 轮，状态 `{ev_info['has_answer_status']}`）；会话级 `answer_session_ids` = **{ev_info['session_level_denominator']}** 轮",
            "",
        ])

        if ev_info["has_answer_status"] != "annotated":
            md_lines.append("> 消息级 `has_answer` 标注缺失（`missing`），不以整会话替代表达。\n")


        for unit in ev_info["message_level_units"]:
            u_rec = {
                "evidence_turn_id": unit["turn_id"],
                "session_id": unit["session_id"],
                "timestamp": unit["timestamp"],
                "source_message_indices": unit["source_message_indices"],
                "by_strategy": {},
            }
            md_lines.extend([
                f"### 证据轮 `T{unit['turn_id']}`（`{unit['session_id']}`，消息索引 `{unit['source_message_indices']}`，时间 `{unit['timestamp']}`）",
                "",
                "| 策略 | 共同检查点状态（窗口 / 外部存储 / 探针读出） | 观测失效区间 | 严格事件归因 |",
                "| :--- | :--- | :--- | :--- |",
            ])

            for st in strategies:
                run_id = runs_by_key.get((st, qid))
                if not run_id or not (batch_dir / "runs" / run_id).exists():
                    u_rec["by_strategy"][st] = {"status": "missing_run"}
                    continue
                run_dir = batch_dir / "runs" / run_id
                # Combine online probes and offline_probes (prefer online if both exist at same turn)
                probes_by_tid: dict[int, dict[str, Any]] = {}
                for p in _load_jsonl(run_dir / "offline_probes.jsonl"):
                    probes_by_tid[int(p["turn_id"])] = p
                for p in _load_jsonl(run_dir / "probes.jsonl"):
                    probes_by_tid[int(p["turn_id"])] = p

                cp_states = {}
                cp_strs = []
                for cp in cp_turns:
                    if cp < unit["turn_id"]:
                        continue
                    sf = run_dir / "snapshots" / f"turn_{cp:03d}.json"
                    if not sf.exists():
                        cp_states[str(cp)] = {"status": "missing_snapshot"}
                        cp_strs.append(f"T{cp}: 无快照")
                        continue
                    snap = _load_json(sf)
                    st_info = _inspect_unit_at_snapshot(st, snap, unit, ev_info["turns_by_id"], probes_by_tid.get(cp))
                    cp_states[str(cp)] = st_info
                    ext_s = str(st_info["in_external_storage"])
                    if st == "N11" and st_info["content_detail"]:
                        cd = st_info["content_detail"]
                        ext_s = f"src={st_info['in_external_storage']}, content={cd['retained']}(T{cd['via_turn_id']}@{cd['timestamp']})"
                    elif st == "N13" and st_info["content_detail"]:
                        cd = st_info["content_detail"]
                        ext_s = f"fact={cd['in_fact_table']}, arch={cd['in_archive']}"
                    p_s = (
                        f"{st_info['in_probe_readout']}(ok={st_info['probe_correct']},{st_info['probe_origin']})"
                        if st_info["in_probe_readout"] != "unobserved"
                        else "未观测"
                    )
                    cp_strs.append(f"`T{cp}`: 窗={st_info['in_recent_window']}, 外={ext_s}, 读={p_s}")

                loss_info = _trace_unit_loss_interval(st, run_dir, unit, ev_info["turns_by_id"])
                u_rec["by_strategy"][st] = {
                    "run_id": run_id,
                    "checkpoints": cp_states,
                    "loss_trace": loss_info,
                }
                md_lines.append(
                    f"| **{st}** | {' ; '.join(cp_strs)} | `{loss_info['observed_loss_interval'] or '—'}` | `{loss_info['attribution']}` |"
                )
            md_lines.append("")
            q_record["units"].append(u_rec)
        output_records.append(q_record)

    report_dir = ROOT / "reports" / batch_dir.parent.name / batch_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    out_md = Path(args.out_md) if args.out_md else (report_dir / "evidence_trajectory.md")
    out_json = Path(args.out_json) if args.out_json else (report_dir / "evidence_trajectory.json")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    out_json.write_text(json.dumps(output_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote evidence trajectory Markdown: {out_md}")
    print(f"Wrote evidence trajectory JSON:     {out_json}")


if __name__ == "__main__":
    main()
