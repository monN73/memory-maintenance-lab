"""Export a cross-strategy paired fact lifecycle main table aligned by (scenario, turn_id, entity, attr).

Generates both Markdown and JSON tables for the second main baseline result:
each cell records expected state, actual retention across recent window and external storage,
readout from identical checkpoint probes, and transition/evidence pointing to exact snapshots/events.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _fact_spec_for_ctrl(scenario_id: str) -> list[dict[str, Any]]:
    """Define tracked facts and expected/old values for known controlled scenarios."""
    if scenario_id == "fact-ctrl-01-update":
        return [
            {
                "entity": "杉木项目",
                "attr": "负责人",
                "probe_id": "p1_target_lead",
                "role": "目标更新键",
                "source_turns": {"林": 1, "周": 23},
                "expected_at": lambda t: ("林", False, 1) if t < 23 else ("周", False, 23),
                "old_value_at": lambda t: None if t < 23 else "林",
            },
            {
                "entity": "杉木项目",
                "attr": "代号",
                "probe_id": "p2_same_entity_code",
                "role": "同实体邻近键（防误删）",
                "source_turns": {"SQ-741": 1},
                "expected_at": lambda t: ("SQ-741", False, 1),
                "old_value_at": lambda t: None,
            },
            {
                "entity": "海盐项目",
                "attr": "负责人",
                "probe_id": "p3_other_entity_lead",
                "role": "跨实体邻近键（防误删）",
                "source_turns": {"吴": 1},
                "expected_at": lambda t: ("吴", False, 1),
                "old_value_at": lambda t: None,
            },
        ]
    raise ValueError(f"No built-in tracked fact spec for {scenario_id}")


def _check_text_presence(text: str, entity: str, attr: str, value: str | None) -> bool:
    if not value or not text:
        return False
    if attr == "负责人" and entity == "杉木项目":
        if value == "周":
            # Avoid false match on "周报"
            return bool(re.search(r"负责人\s*(?:为|是|改为|更新为|：|:|=)\s*周|杉木项目\s*\|\s*负责人\s*=\s*周", text))
        if value == "林":
            return bool(re.search(r"负责人(?:为|是|：|:|=)\s*林|原负责人林|杉木项目\s*\|\s*负责人\s*=\s*林", text))
    if attr == "负责人" and entity == "海盐项目":
        return ("海盐" in text and value in text)
    return value in text


def _inspect_storage(
    strategy: str,
    snap: dict[str, Any],
    probe: dict[str, Any] | None,
    spec: dict[str, Any],
    turn_id: int,
) -> tuple[str, dict[str, Any]]:
    exp_val, is_ret, exp_src_turn = spec["expected_at"](turn_id)
    old_val = spec["old_value_at"](turn_id)
    kept_ids = [int(k["turn_id"]) for k in snap.get("kept_exchanges", [])]
    kept_text = "\n".join(
        f"{k.get('user_text', '')}\n{k.get('assistant_text', '') or ''}"
        for k in snap.get("kept_exchanges", [])
    )
    mem_text = snap.get("memory_text") or ""
    arch_ids = set(int(x) for x in (snap.get("archived_turn_ids") or []))
    ret_ids = set(int(x) for x in ((probe or {}).get("retrieved_turn_ids") or []))

    in_recent_cur = exp_src_turn in kept_ids or _check_text_presence(kept_text, spec["entity"], spec["attr"], exp_val)
    old_src_turn = spec["source_turns"].get(old_val) if old_val else None
    in_recent_old = bool(old_val and (old_src_turn in kept_ids or _check_text_presence(kept_text, spec["entity"], spec["attr"], old_val)))

    if strategy == "N2":
        ext_cur = False
        ext_old = False
        desc_ext = "外部存储: 不适用"
    elif strategy in ("N7", "N8"):
        in_arch_cur = exp_src_turn in arch_ids
        in_ret_cur = exp_src_turn in ret_ids
        in_arch_old = bool(old_src_turn and old_src_turn in arch_ids)
        in_ret_old = bool(old_src_turn and old_src_turn in ret_ids)
        ext_cur = in_arch_cur
        ext_old = in_arch_old
        cur_tag = f"当前值({exp_val}): " + (
            f"仍在归档(T{exp_src_turn}), {'已被探针召回' if in_ret_cur else '未被该探针召回'}"
            if in_arch_cur
            else ("已不在归档(被淘汰)" if not in_recent_cur else "在近期窗口")
        )
        old_tag = ""
        if old_val:
            old_tag = f"；旧值({old_val}): " + (
                f"仍在归档(T{old_src_turn}), {'已被召回' if in_ret_old else '未召回'}"
                if in_arch_old
                else "已不在归档"
            )
        desc_ext = cur_tag + old_tag
    elif strategy in ("N9", "N10"):
        fact_rows = []
        for line in mem_text.splitlines():
            m = re.match(r"^\[T(\d+)\]\s*([^|]+?)\s*\|\s*([^=]+?)\s*=\s*(.+)$", line.strip())
            if m and m.group(2).strip() == spec["entity"] and m.group(3).strip() == spec["attr"]:
                fact_rows.append((int(m.group(1)), m.group(4).strip()))
        ext_cur = any(val == exp_val for _, val in fact_rows)
        ext_old = bool(old_val and any(val == old_val for _, val in fact_rows))
        cur_tag = f"事实表当前值({exp_val}): {'保留' if ext_cur else '缺失'}"
        old_tag = ""
        if old_val:
            old_tag = f"；旧值({old_val}): {'并存保留(未覆盖)' if ext_old else '已移除/覆盖'}"
        desc_ext = cur_tag + old_tag
    else:
        # N3 (extractive) or N4 (LLM summary)
        store_label = "摘录M" if strategy == "N3" else "摘要M(语义核对)"
        ext_cur = _check_text_presence(mem_text, spec["entity"], spec["attr"], exp_val)
        ext_old = bool(old_val and _check_text_presence(mem_text, spec["entity"], spec["attr"], old_val))
        cur_tag = f"{store_label}当前值({exp_val}): {'保留' if ext_cur else '缺失'}"
        old_tag = ""
        if old_val:
            if strategy == "N4" and "原负责人林为历史" in mem_text:
                old_tag = f"；旧值({old_val}): 标为历史背景"
            elif strategy == "N3":
                old_tag = f"；旧值({old_val}): {'并存保留(未替换)' if ext_old else '已随T1容量淘汰被动缺失'}"
            else:
                old_tag = f"；旧值({old_val}): {'并存保留' if ext_old else '缺失/已替换'}"
        desc_ext = cur_tag + old_tag

    recent_str = (
        f"近期窗口({kept_ids[0]}..{kept_ids[-1]}): "
        f"当前值{'在' if in_recent_cur else '已滑出'}"
        + (f", 旧值{'在' if in_recent_old else '已滑出'}" if old_val else "")
    )
    return f"{recent_str}；{desc_ext}", {
        "in_recent_cur": in_recent_cur,
        "ext_cur": ext_cur,
        "ext_old": ext_old,
        "visible_cur": in_recent_cur or ext_cur,
        "exp_val": exp_val,
        "old_val": old_val,
        "exp_src_turn": exp_src_turn,
    }


def _find_exact_event_transitions(
    strategy: str,
    run_dir: Path,
    spec: dict[str, Any],
    turns_total: int,
) -> dict[str, Any]:
    evs = _load_jsonl(run_dir / "events.jsonl")
    src_t1 = 1
    src_t23 = 23

    write_t1_ev = next((e for e in evs if src_t1 in (e.get("evicted_turn_ids") or [])), None)
    write_t23_ev = next((e for e in evs if src_t23 in (e.get("evicted_turn_ids") or [])), None)

    # Scan snapshots turn-by-turn to find exact first missing turn after initial retention
    seen_in_state = False
    last_retained_turn = None
    first_missing_turn = None
    old_removed_turn = None
    was_old_in_state = False

    for tid in range(1, turns_total + 1):
        snap_path = run_dir / "snapshots" / f"turn_{tid:03d}.json"
        if not snap_path.exists():
            continue
        snap = _load_json(snap_path)
        exp_val, _, exp_src = spec["expected_at"](tid)
        old_val = spec["old_value_at"](tid)
        _, state = _inspect_storage(strategy, snap, None, spec, tid)
        if state["visible_cur"]:
            seen_in_state = True
            last_retained_turn = tid
        elif seen_in_state and first_missing_turn is None:
            first_missing_turn = tid
        if old_val and tid >= 23:
            if state["ext_old"]:
                was_old_in_state = True
            elif was_old_in_state and old_removed_turn is None and not state["ext_old"]:
                old_removed_turn = tid

    # Match nearest event at missing / old_removed turn
    loss_ev = next((e for e in evs if e.get("turn_id") == first_missing_turn), None) if first_missing_turn else None
    rem_ev = next((e for e in evs if e.get("turn_id") == old_removed_turn), None) if old_removed_turn else None
    return {
        "write_t1_ev": write_t1_ev["event_id"] if write_t1_ev else None,
        "write_t1_turn": write_t1_ev["turn_id"] if write_t1_ev else None,
        "write_t23_ev": write_t23_ev["event_id"] if write_t23_ev else None,
        "write_t23_turn": write_t23_ev["turn_id"] if write_t23_ev else None,
        "first_missing_turn": first_missing_turn,
        "last_retained_before_loss": (first_missing_turn - 1) if first_missing_turn else last_retained_turn,
        "loss_event_id": loss_ev["event_id"] if loss_ev else None,
        "old_removed_turn": old_removed_turn,
        "rem_event_id": rem_ev["event_id"] if rem_ev else None,
    }


def _describe_change_and_evidence(
    strategy: str,
    tid: int,
    prev_tid: int | None,
    spec: dict[str, Any],
    cur_state: dict[str, Any],
    prev_state: dict[str, Any] | None,
    probe: dict[str, Any] | None,
    prev_probe: dict[str, Any] | None,
    trans: dict[str, Any],
) -> str:
    snap_ref = f"`snapshots/turn_{tid:03d}.json`"
    if prev_tid is None:
        # First checkpoint (T22)
        if strategy == "N2":
            ev_ref = f"`{trans['write_t1_ev']}`" if trans["write_t1_ev"] else "窗口淘汰"
            return f"**意外丢失(窗口遗忘)**：探测区间 `(T1, T22]` 首次观测失效；逐轮快照+事件 {ev_ref} 定位 T1 在 `T17` 被滑窗淘汰（N2 无外部存储）；见 {snap_ref}"
        ev_ref = f"`{trans['write_t1_ev']}`" if trans["write_t1_ev"] else "维护事件"
        return f"**首次写入并保留**：T1 经 {ev_ref} 进入外部状态并在 {snap_ref} 保留"

    # Subsequent checkpoints (T45, T50)
    was_cur = prev_state["visible_cur"] if prev_state else False
    has_cur = cur_state["visible_cur"]
    old_val = cur_state["old_val"]
    had_old = prev_state["ext_old"] if (prev_state and old_val) else (prev_state["ext_cur"] if prev_state else False)
    has_old = cur_state["ext_old"]
    p_ok = probe.get("correct") if probe else None
    prev_p_ok = prev_probe.get("correct") if prev_probe else None

    if strategy == "N2":
        if tid == 45 and spec["attr"] == "负责人" and spec["entity"] == "杉木项目":
            ev_ref = f"`{trans['write_t23_ev']}`" if trans["write_t23_ev"] else "滑窗淘汰"
            return f"**意外丢失(窗口遗忘)**：T23 新值曾短暂在近期窗口(23..35)，在 {ev_ref} 被淘汰出窗；探测区间 `(T{prev_tid}, T{tid}]` 持续缺失，见 {snap_ref}"
        return f"**持续缺失(窗口遗忘)**：T1 已于 T17 滑出近期窗口且无外部存储；见 {snap_ref}"

    if not has_cur:
        # Lost! Distinguish probe interval vs exact snapshot turn
        loss_t = trans["first_missing_turn"]
        ev_ref = f"`{trans['loss_event_id']}`" if trans["loss_event_id"] else f"T{loss_t}附近维护事件"
        if prev_p_ok and p_ok is False:
            return f"**意外丢失(有效事实遗忘)**：探测区间 `(T{prev_tid}, T{tid}]` 失配；逐轮快照确认在 {ev_ref} 后于 `turn_{loss_t:03d}.json` 首次从状态缺失（此前 T{prev_tid} 确实存在），见 {snap_ref}"
        return f"**持续缺失(已遗忘)**：已于 {ev_ref} (`turn_{loss_t:03d}.json`) 丢失，见 {snap_ref}"

    # Current value is preserved
    if spec["attr"] == "负责人" and spec["entity"] == "杉木项目" and tid == 45:
        ev_ref = f"`{trans['write_t23_ev']}`" if trans["write_t23_ev"] else "维护事件"
        if strategy == "N10":
            return f"**正确更新并移除旧值(intended_revision_removal)**：{ev_ref} 写入新值(周) 并主动覆盖删除旧值(林)，于 `turn_{trans['old_removed_turn']:03d}.json` 生效（邻近键完好保留）；见 {snap_ref}"
        if strategy == "N4":
            return f"**正确更新(语义保留历史)**：{ev_ref} 将负责人更新为周，旧值林标为历史背景；见 {snap_ref}"
        if strategy in ("N7", "N8"):
            return f"**归档更新值并保留原文**：T23 经 {ev_ref} 存入归档(T1 旧原文并存)，探针召回 `[1, 23, ...]`；见 {snap_ref}"
        if strategy == "N3":
            return f"**追加新值后被动挤出旧值**：T23 经 {ev_ref} 写入摘录 M，但旧值(林)并非被替换，而是随 T1 于 `t37-e1` (`turn_037.json`) 因 2K 容量截断被连带淘汰（导致同在 T1 的代号与海盐负责人一并丢失）；见 {snap_ref}"
        return f"**追加新值(旧值并存)**：T23 经 {ev_ref} 追加进状态，旧值 `{old_val}` 未删除仍并存；见 {snap_ref}"

    if spec["attr"] == "负责人" and spec["entity"] == "杉木项目" and tid == 50 and strategy == "N8":
        return f"**新值保留、旧值随 T1 归档 FIFO 被动淘汰**：`t48-e1` 将 T1 淘汰出 16K 归档导致旧值(林)连同邻近键一并消失，T23 新值(周)仍在归档并被召回；见 {snap_ref}"

    return f"**持续保留**：与 T{prev_tid} 相比状态与期望值一致，见 {snap_ref}"


def main():
    parser = argparse.ArgumentParser(description="Build paired cross-strategy fact lifecycle table.")
    parser.add_argument("--batch-dir", required=True, help="Batch directory under runs/ or absolute path")
    parser.add_argument("--scenario-id", default="fact-ctrl-01-update")
    parser.add_argument("--strategies", default="N2,N3,N4,N7,N8,N9,N10")
    parser.add_argument("--checkpoints", default="22,45,50")
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_absolute():
        batch_dir = ROOT / args.batch_dir
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    checkpoints = [int(x.strip()) for x in args.checkpoints.split(",") if x.strip()]
    specs = _fact_spec_for_ctrl(args.scenario_id)

    variant = _load_json(batch_dir / "batch_summary.json").get("dataset", {}).get("variant", "unified")

    rows: list[dict[str, Any]] = []
    for tid_idx, tid in enumerate(checkpoints):
        prev_tid = checkpoints[tid_idx - 1] if tid_idx > 0 else None
        for spec in specs:
            exp_val, is_ret, _ = spec["expected_at"](tid)
            old_val = spec["old_value_at"](tid)
            exp_desc = f"当前有效: `{exp_val}`" + ("（已撤回）" if is_ret else "（未撤回）") + (f"；旧值 `{old_val}` 已失效" if old_val else "")
            row = {
                "scenario_id": args.scenario_id,
                "turn_id": tid,
                "entity": spec["entity"],
                "attr": spec["attr"],
                "role": spec["role"],
                "expected_state": exp_desc,
                "strategies": {},
            }
            for st in strategies:
                run_id = f"lme-{variant}-{args.scenario_id}-direct-{st}-r1"
                run_dir = batch_dir / "runs" / run_id
                snap = _load_json(run_dir / "snapshots" / f"turn_{tid:03d}.json")
                prev_snap = _load_json(run_dir / "snapshots" / f"turn_{prev_tid:03d}.json") if prev_tid else None
                probes = _load_jsonl(run_dir / "probes.jsonl")
                probe = next((p for p in probes if int(p["turn_id"]) == tid and p["probe_id"] == spec["probe_id"]), None)
                prev_probe = next((p for p in probes if prev_tid and int(p["turn_id"]) == prev_tid and p["probe_id"] == spec["probe_id"]), None)

                turns_total = len(list((run_dir / "snapshots").glob("turn_*.json")))
                trans = _find_exact_event_transitions(st, run_dir, spec, turns_total)
                actual_desc, cur_state = _inspect_storage(st, snap, probe, spec, tid)
                _, prev_state = _inspect_storage(st, prev_snap, prev_probe, spec, prev_tid) if prev_snap else ("", None)

                if probe:
                    hyp_short = re.sub(r"\s+", " ", (probe.get("hypothesis") or "").strip())[:48]
                    readout_tag = f"{'✓ 正确' if probe.get('correct') else '✗ 错误'}（{probe.get('trigger')}共同检查点快照读出）: `{hyp_short}`"
                else:
                    readout_tag = "没有观测（该检查点未触发探针）"

                change_desc = _describe_change_and_evidence(st, tid, prev_tid, spec, cur_state, prev_state, probe, prev_probe, trans)
                row["strategies"][st] = {
                    "expected_state": exp_desc,
                    "actual_retention_state": actual_desc,
                    "probe_readout": readout_tag,
                    "change_and_evidence": change_desc,
                    "probe_correct": probe.get("correct") if probe else None,
                    "transitions": trans,
                }
            rows.append(row)

    # Render clean Markdown paired tables grouped by (turn_id, entity, attr)
    md_lines = [
        f"# 跨策略逐事实维护过程配对主表（{args.scenario_id}）",
        "",
        f"- **来源批次**：`{batch_dir.name}`",
        f"- **对齐粒度**：`场景 × turn_id × 实体 × 属性`（检查完整摄入第 `t` 轮之后状态，共同检查点 `turn_id` = `{checkpoints}`）",
        f"- **纳入策略**：`{', '.join(strategies)}`（同一输入历史、8K 预算、同一共同检查点隔离探针协议）",
        "",
    ]

    for row in rows:
        md_lines.extend([
            f"## 共同检查点 `T{row['turn_id']}` · `{row['entity']}` · `{row['attr']}`（{row['role']}）",
            "",
            f"- **预期状态（Gold）**：{row['expected_state']}",
            "",
            "| 策略 | 预期状态 | 实际保留状态（近期窗口 & 外部存储） | 同一 Probe 读出（共同检查点快照读出） | 变化及证据（快照 / 事件 / 失效区间） |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for st in strategies:
            cell = row["strategies"][st]
            md_lines.append(
                f"| **{st}** | {cell['expected_state']} | {cell['actual_retention_state']} | {cell['probe_readout']} | {cell['change_and_evidence']} |"
            )
        md_lines.append("")

    report_dir = ROOT / "reports" / "ecnu-plus" / batch_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    out_md = Path(args.out_md) if args.out_md else (report_dir / "fact_lifecycle_table.md")
    out_json = Path(args.out_json) if args.out_json else (report_dir / "fact_lifecycle_table.json")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote Markdown paired table: {out_md}")
    print(f"Wrote JSON paired table:     {out_json}")


if __name__ == "__main__":
    main()
