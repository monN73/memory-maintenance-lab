"""Summarize completed probe plan trajectories with regression intervals, gaps, and maintenance diffs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from probe_from_snapshots import _load_json, _load_jsonl  # noqa: E402
from run_probe_plan import (  # noqa: E402
    _find_saved_record,
    _resolve_source_batch,
    _resolve_turns,
    compute_regression_intervals,
)


def _snapshot_delta(run_dir: Path, turn: int) -> dict[str, Any] | None:
    cur_p = run_dir / "snapshots" / f"turn_{turn:03d}.json"
    prev_p = run_dir / "snapshots" / f"turn_{turn - 1:03d}.json"
    if not (cur_p.exists() and prev_p.exists()):
        return None
    cur = _load_json(cur_p)
    prev = _load_json(prev_p)
    prev_kept = [int(k["turn_id"]) for k in prev.get("kept_exchanges", [])]
    cur_kept = [int(k["turn_id"]) for k in cur.get("kept_exchanges", [])]
    prev_arch = [int(x) for x in (prev.get("archived_turn_ids") or [])]
    cur_arch = [int(x) for x in (cur.get("archived_turn_ids") or [])]
    return {
        "turn_id": turn,
        "window_removed": [x for x in prev_kept if x not in set(cur_kept)],
        "window_added": [x for x in cur_kept if x not in set(prev_kept)],
        "archive_removed": [x for x in prev_arch if x not in set(cur_arch)] if ("archived_turn_ids" in cur and "archived_turn_ids" in prev) else None,
        "archive_added": [x for x in cur_arch if x not in set(prev_arch)] if ("archived_turn_ids" in cur and "archived_turn_ids" in prev) else None,
        "memory_text_changed": (prev.get("memory_text") or "") != (cur.get("memory_text") or ""),
    }


def summarize_plan(plan_path: Path) -> dict[str, Any]:
    plan = _load_json(plan_path)
    plan_dir = plan_path.resolve().parent
    strategies = list(plan.get("strategies") or plan["source_batches"].keys())
    ds_path = ROOT / plan["dataset"]
    ds_items = {x["question_id"]: x for x in json.loads(ds_path.read_text(encoding="utf-8")) if x["question_id"] in plan["questions"]}

    rows: list[dict[str, Any]] = []
    by_strategy_summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for qid, spec in plan["questions"].items():
        total_turns = sum(len(s) // 2 + len(s) % 2 for s in ds_items[qid]["haystack_sessions"])
        turns = _resolve_turns(spec, total_turns + 100)  # bounds checked against saved records/runs below
        for strategy in strategies:
            source_batch = _resolve_source_batch(plan, qid, strategy)
            batch_dir = ROOT / "runs" / plan["model"] / source_batch
            run_dir, = (batch_dir / "runs").glob(f"*-{qid}-direct-{strategy}-r1")
            manifest = _load_json(run_dir / "manifest.json")
            events = _load_jsonl(run_dir / "events.jsonl")
            events_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for ev in events:
                events_by_turn[int(ev["turn_id"])].append(ev)

            obs: list[dict[str, Any]] = []
            term_gaps: list[dict[str, Any]] = []
            for turn in turns:
                rec_path = plan_dir / "records" / qid / strategy / f"{turn:03d}.json"
                if rec_path.exists():
                    rec = _load_json(rec_path)
                    if rec.get("status") in ("termination_turn_no_snapshot", "post_termination"):
                        term_gaps.append({"turn_id": turn, "status": rec["status"], "reason": rec.get("termination_reason")})
                        continue
                    obs.append(rec)
                else:
                    _, reused = _find_saved_record(plan_dir, plan, qid, strategy, turn)
                    if reused is not None:
                        obs.append(reused)

            reg = compute_regression_intervals(obs)
            lo = reg["last_ok_turn_before_regression"]
            hi = reg["first_regression_turn"]
            interval_events = []
            interval_deltas = []
            if lo is not None and hi is not None:
                for t in range(lo + 1, hi + 1):
                    for ev in events_by_turn.get(t, []):
                        interval_events.append({
                            "turn_id": t,
                            "event_id": ev.get("event_id"),
                            "mechanism": ev.get("mechanism"),
                            "evicted_turn_ids": ev.get("evicted_turn_ids"),
                            "archive_evicted_turn_ids": ev.get("archive_evicted_turn_ids"),
                        })
                    d = _snapshot_delta(run_dir, t)
                    if d is not None:
                        interval_deltas.append(d)

            # Check rechecks & skip-maintenance counterfactuals if present
            recheck_rows = []
            rc_dir = plan_dir / "rechecks" / qid / strategy
            if rc_dir.exists():
                for rcf in sorted(rc_dir.glob("*.jsonl")):
                    recheck_rows.extend(_load_jsonl(rcf))

            cf_rows = []
            cf_dir = plan_dir / "counterfactuals" / qid / strategy
            if cf_dir.exists():
                for cff in sorted(cf_dir.glob("*.json")):
                    cf_rows.append(_load_json(cff))

            framing = "no_regression_observed"
            if hi is not None:
                matched_cf = next((c for c in cf_rows if int(c["turn_id"]) == hi), None)
                if matched_cf and matched_cf.get("fits_budget") and matched_cf.get("correct") is True:
                    framing = "同轮配对重放中，跳过维护时答对，按原维护结果读出时答错"
                elif interval_events:
                    framing = "同轮/区间内发生维护与读出转错"
                else:
                    framing = "区间内未见维护事件（窗口自然滑出或读出波动）"

            st_sum = by_strategy_summary[strategy]
            st_sum["questions"] += 1
            st_sum["valid_observations"] += reg["valid_observations"]
            st_sum["api_error_gaps"] += len(reg["api_error_gaps"])
            st_sum["termination_gaps"] += len(term_gaps)
            if reg["valid_observations"] >= 2:
                st_sum["questions_ge2_valid"] += 1
            if reg["first_regression_turn"] is not None:
                st_sum["questions_with_regression"] += 1
            if reg["first_observation_incorrect"]:
                st_sum["questions_first_obs_incorrect"] += 1

            rows.append({
                "question_id": qid,
                "strategy": strategy,
                "source_batch": source_batch,
                "run_id": manifest["run_id"],
                "run_status": manifest["status"],
                **reg,
                "termination_gaps": len(term_gaps),
                "interval_events": interval_events,
                "interval_snapshot_deltas": interval_deltas,
                "rechecks": [
                    {"turn_id": r["turn_id"], "classification": r.get("recheck_classification"), "primary_correct": r.get("primary_correct"), "recheck_correct": r.get("correct")}
                    for r in recheck_rows
                ],
                "counterfactuals": [
                    {"turn_id": c["turn_id"], "fits_budget": c.get("fits_budget"), "skip_maint_correct": c.get("correct"), "status": c.get("status")}
                    for c in cf_rows
                ],
                "causal_framing": framing,
            })

    return {
        "plan_id": plan["plan_id"],
        "by_strategy": {st: dict(vals) for st, vals in sorted(by_strategy_summary.items())},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan_path = args.plan if args.plan.is_absolute() else (Path.cwd() / args.plan)
    summary = summarize_plan(plan_path)
    out_dir = plan_path.resolve().parent
    (out_dir / "audit_review.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        f"# 探针轨迹失效与维护对照审核表（{summary['plan_id']}）",
        "",
        "## 分策略完整计数汇总",
        "",
        "| 策略 | 题数 | ≥2次有效观测题数 | 首次即错题数 | 由对转错题数 | 有效探针数 | API 缺口 | 终止缺口 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for st, s in summary["by_strategy"].items():
        md.append(
            f"| {st} | {s.get('questions', 0)} | {s.get('questions_ge2_valid', 0)} | "
            f"{s.get('questions_first_obs_incorrect', 0)} | {s.get('questions_with_regression', 0)} | "
            f"{s.get('valid_observations', 0)} | {s.get('api_error_gaps', 0)} | {s.get('termination_gaps', 0)} |"
        )
    md.extend([
        "",
        "## 逐题 × 策略失效区间明细",
        "",
        "| 题目 | 策略 | 来源批次 | 有效观测 | 首错轮 | 转错区间 `(last_ok, first_reg]` | 跨过缺口 | 区间维护事件数 | 展示定性 |",
        "| :--- | :--- | :--- | ---: | :---: | :---: | :---: | ---: | :--- |",
    ])
    for r in summary["rows"]:
        md.append(
            f"| `{r['question_id']}` | `{r['strategy']}` | `{r['source_batch']}` | {r['valid_observations']} | "
            f"{r['first_observed_error_turn'] or '—'} | `{r['regression_interval'] or '—'}` | "
            f"{r['bridged_api_error_gaps'] or '—'} | {len(r['interval_events'])} | {r['causal_framing']} |"
        )
    (out_dir / "audit_review.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'audit_review.md'} and {out_dir / 'audit_review.json'}")


if __name__ == "__main__":
    main()
