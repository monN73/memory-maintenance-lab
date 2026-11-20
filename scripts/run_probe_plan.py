"""Run and audit a frozen snapshot-replay probe plan with strict reuse & gap tracking."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from budget_executor import ARCHIVE_POLICY_BY_STRATEGY, ARCHIVE_STRATEGIES, bound_archive, build_messages, build_readout_memory  # noqa: E402
from metrics import normalize_text, sha256_json  # noqa: E402
from probe_from_snapshots import (  # noqa: E402
    BudgetConfig,
    Exchange,
    LLMClient,
    SYSTEM_PROMPT_LONGMEMEVAL,
    _load_json,
    adapt_longmemeval_item,
    reconstruct_probe_inputs,
    reconstruct_skip_maintenance_inputs,
    run_isolated_probe,
)

TERMINATION_STATUSES = {
    "overflow",
    "maintenance_budget_error",
    "maintenance_empty_error",
    "maintenance_truncated_error",
    "maintenance_api_error",
}
FROZEN_PROBE_PARAMS = {"temperature": 0.0, "max_tokens": 256, "seed": 42}
FROZEN_JUDGE_PROMPT_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _resolve_turns(spec: dict[str, Any], total_turns: int) -> list[int]:
    if "turns" in spec:
        turns = [int(t) for t in spec["turns"]]
        if not turns or any(t < 1 or t > total_turns for t in turns):
            raise ValueError(f"Invalid discrete turn list {turns} for total_turns={total_turns}")
        return list(dict.fromkeys(turns))
    start, end = int(spec["first_turn"]), int(spec["last_turn"])
    if not 1 <= start <= end <= total_turns:
        raise ValueError(f"Invalid interval [{start}, {end}] for total_turns={total_turns}")
    return list(range(start, end + 1))


def _resolve_source_batch(plan: dict[str, Any], qid: str, strategy: str) -> str:
    cell_sources = plan.get("cell_sources") or {}
    if f"{qid}:{strategy}" in cell_sources:
        return str(cell_sources[f"{qid}:{strategy}"])
    entry = plan["source_batches"][strategy]
    if isinstance(entry, str):
        return entry
    if qid in entry:
        return str(entry[qid])
    if "overrides" in entry and qid in entry["overrides"]:
        return str(entry["overrides"][qid])
    if "default" in entry:
        return str(entry["default"])
    raise KeyError(f"No source_batch defined for ({qid}, {strategy})")


def _classify_missing_snapshot(run_dir: Path, turn: int) -> dict[str, Any]:
    manifest = _load_json(run_dir / "manifest.json")
    status = manifest.get("status", "unknown")
    term_reason = manifest.get("termination_reason") or status
    turns_path = run_dir / "turns.jsonl"
    term_turn = None
    if turns_path.exists():
        lines = [l for l in turns_path.read_text(encoding="utf-8").split("\n") if l.strip()]
        if lines:
            last_row = json.loads(lines[-1])
            term_turn = int(last_row["turn_id"])
    snap_dir = run_dir / "snapshots"
    snaps = sorted(int(p.stem.split("_")[1]) for p in snap_dir.glob("turn_*.json")) if snap_dir.exists() else []
    last_snap = snaps[-1] if snaps else None

    if status in TERMINATION_STATUSES and term_turn is not None:
        if turn == term_turn:
            gap_type = "termination_turn_no_snapshot"
        elif turn > term_turn:
            gap_type = "post_termination"
        else:
            gap_type = "unexpected_missing"
    else:
        gap_type = "unexpected_missing"

    return {
        "gap_type": gap_type,
        "termination_reason": term_reason,
        "termination_turn": term_turn,
        "last_available_turn": last_snap,
    }


def _expected_probe_params(provider: str) -> dict[str, Any]:
    return {
        **FROZEN_PROBE_PARAMS,
        "thinking": "disabled" if provider == "deepseek" else None,
    }


def verify_saved_record(
    saved: dict[str, Any],
    *,
    source_batch: str,
    qid: str,
    strategy: str,
    turn: int,
    request_sha256: str,
    expected: str,
    plan: dict[str, Any],
) -> tuple[str, list[str]]:
    """Verify saved probe record against frozen plan metadata, request hash, model/judge params, and verdict."""
    if saved.get("status") == "api_error_gap":
        return "pending_new", ["api_error_gap_retry"]

    reasons: list[str] = []
    if saved.get("source_batch") != source_batch:
        reasons.append(f"source_batch:{saved.get('source_batch')}!={source_batch}")
    if saved.get("question_id") != qid:
        reasons.append(f"question_id:{saved.get('question_id')}!={qid}")
    if saved.get("strategy") != strategy:
        reasons.append(f"strategy:{saved.get('strategy')}!={strategy}")
    if int(saved.get("turn_id", -1)) != turn:
        reasons.append(f"turn_id:{saved.get('turn_id')}!={turn}")
    if saved.get("request_sha256") != request_sha256:
        reasons.append("request_sha256_mismatch")
    if saved.get("expected") != expected:
        reasons.append("expected_mismatch")
    if saved.get("provider") != plan["provider"] or saved.get("model") != plan["model"]:
        reasons.append(f"answer_model:{saved.get('provider')}/{saved.get('model')}")
    exp_judge_prov = plan.get("judge_provider", plan["provider"])
    exp_judge_model = plan.get("judge_model", plan["model"])
    if saved.get("judge_provider") != exp_judge_prov or saved.get("judge_model") != exp_judge_model:
        reasons.append(f"judge_model:{saved.get('judge_provider')}/{saved.get('judge_model')}")
    if not isinstance(saved.get("correct"), bool):
        reasons.append(f"non_bool_correct:{saved.get('correct')}")
    if saved.get("status") != "ok":
        reasons.append(f"status:{saved.get('status')}")

    exp_params = _expected_probe_params(plan["provider"])
    if "probe_params" in saved:
        if saved["probe_params"] != exp_params:
            reasons.append("probe_params_mismatch")
    elif not plan.get("allow_legacy_unlogged_probe_params", True):
        reasons.append("missing_probe_params")

    exp_commit = plan.get("judge_prompt_commit", FROZEN_JUDGE_PROMPT_COMMIT)
    if "judge_prompt_commit" in saved and saved["judge_prompt_commit"] != exp_commit:
        reasons.append("judge_prompt_commit_mismatch")

    if reasons:
        return "mismatch_error", reasons
    return "reusable_ok", []


def _find_saved_record(
    plan_dir: Path,
    plan: dict[str, Any],
    qid: str,
    strategy: str,
    turn: int,
) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = [plan_dir / "records" / qid / strategy / f"{turn:03d}.json"]
    for extra_dir in plan.get("reuse_records_dirs", []):
        p = Path(extra_dir)
        if not p.is_absolute():
            p = ROOT / p
        candidates.append(p / qid / strategy / f"{turn:03d}.json")
    for cand in candidates:
        if cand.exists():
            rec = _load_json(cand)
            if rec.get("status") in ("termination_turn_no_snapshot", "post_termination"):
                continue
            return cand, rec
    return None, None


def compute_regression_intervals(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute first_observed_error, first_regression, (last_ok, first_regression], and recovery preserving gaps."""
    ordered = sorted(observations, key=lambda x: int(x["turn_id"]))
    valid = [o for o in ordered if isinstance(o.get("correct"), bool)]
    gaps = [int(o["turn_id"]) for o in ordered if o.get("status") == "api_error_gap" or o.get("correct") is None]

    first_obs_err = next((int(o["turn_id"]) for o in valid if o["correct"] is False), None)
    first_obs_ok = next((int(o["turn_id"]) for o in valid if o["correct"] is True), None)

    seen_ok = False
    last_ok_before_reg: int | None = None
    first_reg: int | None = None
    for o in valid:
        tid = int(o["turn_id"])
        if o["correct"] is True:
            if first_reg is None:
                last_ok_before_reg = tid
            seen_ok = True
        elif o["correct"] is False and seen_ok and first_reg is None:
            first_reg = tid
            break

    recovered_turns = (
        [int(o["turn_id"]) for o in valid if first_reg is not None and int(o["turn_id"]) > first_reg and o["correct"] is True]
    )
    reg_interval = (
        f"(T{last_ok_before_reg}, T{first_reg}]"
        if (last_ok_before_reg is not None and first_reg is not None)
        else None
    )
    bridged_gaps = (
        [g for g in gaps if last_ok_before_reg < g < first_reg]
        if (last_ok_before_reg is not None and first_reg is not None)
        else []
    )
    return {
        "valid_observations": len(valid),
        "api_error_gaps": gaps,
        "first_observed_error_turn": first_obs_err,
        "first_observation_incorrect": bool(valid and valid[0]["correct"] is False),
        "first_correct_turn": first_obs_ok,
        "last_ok_turn_before_regression": last_ok_before_reg,
        "first_regression_turn": first_reg,
        "regression_interval": reg_interval,
        "bridged_api_error_gaps": bridged_gaps,
        "recovered_turns": recovered_turns,
    }


def _build_verified_legacy_archive_map(
    run_dir: Path,
    strategy: str,
    budget: BudgetConfig,
    exchanges: dict[int, Exchange],
    adapted: Any,
) -> dict[int, list[int]]:
    """Deterministically rebuild per-turn archived_turn_ids from events.jsonl and verify against final_answer.json."""
    if strategy not in ARCHIVE_STRATEGIES:
        return {}
    policy = ARCHIVE_POLICY_BY_STRATEGY.get(strategy)
    evs_path = run_dir / "events.jsonl"
    if not evs_path.exists():
        raise SystemExit(f"Cannot verify legacy archive without events.jsonl: {run_dir}")
    evs = [json.loads(line) for line in evs_path.read_text(encoding="utf-8").split("\n") if line.strip()]
    evs_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ev in evs:
        evs_by_turn[int(ev["turn_id"])].append(ev)

    total_t = len(exchanges)
    arch: list[Exchange] = []
    by_turn: dict[int, list[int]] = {}
    for tid in range(1, total_t + 1):
        for ev in evs_by_turn.get(tid, []):
            if ev.get("evicted_turn_ids"):
                arch.extend(exchanges[int(x)] for x in ev["evicted_turn_ids"])
                if policy:
                    arch, _ = bound_archive(arch, budget.n8_archive_tokens, policy=policy)
        by_turn[tid] = [ex.turn_id for ex in arch]

    # Mandatory cryptographic check against requests/final_answer.json
    fa_path = run_dir / "requests" / "final_answer.json"
    last_snap_path = run_dir / "snapshots" / f"turn_{total_t:03d}.json"
    if not (fa_path.exists() and last_snap_path.exists()):
        raise SystemExit(f"Legacy archive verification requires final_answer.json and turn_{total_t:03d}.json in {run_dir}")
    fa = _load_json(fa_path)
    last_snap = _load_json(last_snap_path)
    kept = [
        Exchange(int(k["turn_id"]), str(k["session_id"]), str(k["user_text"]), k.get("assistant_text"), k.get("timestamp"))
        for k in last_snap.get("kept_exchanges", [])
    ]
    final_evs = evs_by_turn.get(total_t + 1, [])
    ev_set: set[int] = set()
    for fe in final_evs:
        if fe.get("evicted_turn_ids"):
            arch.extend(exchanges[int(x)] for x in fe["evicted_turn_ids"])
            ev_set.update(int(x) for x in fe["evicted_turn_ids"])
            if policy:
                arch, _ = bound_archive(arch, budget.n8_archive_tokens, policy=policy)
    kept_final = [k for k in kept if k.turn_id not in ev_set]
    fq = f"[Current Date: {adapted.question_date}]\n{adapted.question}" if adapted.question_date else adapted.question
    mem, _ = build_readout_memory(strategy, fq, last_snap.get("memory_text") or "", arch, budget.memory_tokens)
    msgs = build_messages(SYSTEM_PROMPT_LONGMEMEVAL, mem, kept_final, fq)
    if sha256_json(msgs) != sha256_json(fa["messages"]):
        raise SystemExit(f"Legacy archive reconstruction failed final_answer.json SHA-256 check in {run_dir}")
    return by_turn


def _load_snapshot_with_archive(
    plan: dict[str, Any],
    run_dir: Path,
    strategy: str,
    turn: int,
    budget: BudgetConfig,
    exchanges: dict[int, Exchange],
    adapted: Any,
    legacy_archive_cache: dict[tuple[str, str], dict[int, list[int]]],
    qid: str,
) -> dict[str, Any]:
    snapshot = _load_json(run_dir / "snapshots" / f"turn_{turn:03d}.json")
    if strategy in ARCHIVE_STRATEGIES and "archived_turn_ids" not in snapshot:
        if not plan.get("allow_legacy_event_archive_reconstruction", False):
            raise SystemExit(f"Snapshot missing archived_turn_ids and legacy reconstruction is disabled: {run_dir}/turn_{turn:03d}.json")
        key = (qid, strategy)
        if key not in legacy_archive_cache:
            legacy_archive_cache[key] = _build_verified_legacy_archive_map(run_dir, strategy, budget, exchanges, adapted)
        snapshot = {**snapshot, "archived_turn_ids": legacy_archive_cache[key][turn]}
    return snapshot


def analyze_plan(plan_path: Path, items_by_qid: dict[str, dict[str, Any]], dataset_hash: str) -> dict[str, Any]:
    """Perform full read-only verification of all cells, gaps, reuse eligibility, and batch provenance."""
    plan = _load_json(plan_path)
    if dataset_hash != plan["dataset_sha256"]:
        raise SystemExit("Dataset hash differs from the frozen plan.")
    plan_dir = plan_path.resolve().parent
    strategies = list(plan.get("strategies") or plan["source_batches"].keys())

    counts = Counter()
    by_strategy: dict[str, Counter] = defaultdict(Counter)
    cell_records: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    adapted_cache: dict[str, Any] = {}
    exchanges_cache: dict[str, dict[int, Exchange]] = {}
    manifest_cache: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    legacy_archive_cache: dict[tuple[str, str], dict[int, list[int]]] = {}

    for qid, spec in plan["questions"].items():
        adapted = adapt_longmemeval_item(items_by_qid[qid])
        adapted_cache[qid] = adapted
        exchanges = {
            t.turn_id: Exchange(t.turn_id, t.session_id, t.user_text, t.assistant_text, t.timestamp, t.source_message_indices)
            for t in adapted.turns
        }
        exchanges_cache[qid] = exchanges
        turns = _resolve_turns(spec, len(exchanges))
        expected = spec["expected"]

        for strategy in strategies:
            source_batch = _resolve_source_batch(plan, qid, strategy)
            batch_dir = ROOT / "runs" / plan["model"] / source_batch
            matches = list((batch_dir / "runs").glob(f"*-{qid}-direct-{strategy}-r1"))
            if len(matches) != 1:
                raise SystemExit(f"Expected 1 run dir for ({qid}, {strategy}) in {batch_dir}, got {len(matches)}")
            run_dir = matches[0]
            manifest = _load_json(run_dir / "manifest.json")
            if (
                manifest["dataset"]["file_sha256"] != dataset_hash
                or manifest["answer_model"] != plan["model"]
                or manifest["provider"] != plan["provider"]
            ):
                raise SystemExit(f"Run metadata differs from plan: {run_dir.name}")
            manifest_cache[(qid, strategy)] = (run_dir, manifest)
            budget = BudgetConfig(**manifest["effective_budget"])
            provenance_rows.append({
                "question_id": qid,
                "strategy": strategy,
                "source_batch": source_batch,
                "run_id": manifest["run_id"],
                "run_status": manifest["status"],
                "turns_planned": len(turns),
            })

            probe = replace(adapted.probes[0], active_from_turn=turns[0], expected=expected)
            for turn in turns:
                snap_path = run_dir / "snapshots" / f"turn_{turn:03d}.json"
                if not snap_path.exists():
                    gap_info = _classify_missing_snapshot(run_dir, turn)
                    state = gap_info["gap_type"]
                    counts[state] += 1
                    by_strategy[strategy][state] += 1
                    cell_records.append({
                        "question_id": qid,
                        "strategy": strategy,
                        "turn_id": turn,
                        "source_batch": source_batch,
                        "run_id": manifest["run_id"],
                        "state": state,
                        "gap_info": gap_info,
                    })
                    continue

                snapshot = _load_snapshot_with_archive(
                    plan, run_dir, strategy, turn, budget, exchanges, adapted, legacy_archive_cache, qid,
                )
                readout = reconstruct_probe_inputs(
                    strategy, snapshot, probe.question, adapted.question_date, budget, exchanges,
                )
                saved_path, saved = _find_saved_record(plan_dir, plan, qid, strategy, turn)
                if saved is None:
                    state = "pending_new"
                    reasons: list[str] = []
                else:
                    state, reasons = verify_saved_record(
                        saved,
                        source_batch=source_batch,
                        qid=qid,
                        strategy=strategy,
                        turn=turn,
                        request_sha256=readout["request_sha256"],
                        expected=expected,
                        plan=plan,
                    )
                counts[state] += 1
                by_strategy[strategy][state] += 1
                cell_records.append({
                    "question_id": qid,
                    "strategy": strategy,
                    "turn_id": turn,
                    "source_batch": source_batch,
                    "run_id": manifest["run_id"],
                    "state": state,
                    "reasons": reasons,
                    "saved_path": str(saved_path.relative_to(ROOT)) if saved_path else None,
                    "request_sha256": readout["request_sha256"],
                    "expected": expected,
                })

    return {
        "plan": plan,
        "plan_dir": plan_dir,
        "strategies": strategies,
        "total_cells": len(cell_records),
        "counts": dict(sorted(counts.items())),
        "by_strategy": {st: dict(sorted(c.items())) for st, c in sorted(by_strategy.items())},
        "provenance": provenance_rows,
        "cells": cell_records,
        "adapted_cache": adapted_cache,
        "exchanges_cache": exchanges_cache,
        "manifest_cache": manifest_cache,
        "legacy_archive_cache": legacy_archive_cache,
    }


def _write_verification_checklist(plan_path: Path, analysis: dict[str, Any]) -> tuple[Path, Path]:
    plan_dir = analysis["plan_dir"]
    c = analysis["counts"]
    termination_gaps = c.get("termination_turn_no_snapshot", 0) + c.get("post_termination", 0)
    unexpected = c.get("unexpected_missing", 0)
    mismatches = c.get("mismatch_error", 0)
    summary = {
        "plan_file": str(plan_path.relative_to(ROOT)),
        "total_cells": analysis["total_cells"],
        "reusable_ok": c.get("reusable_ok", 0),
        "pending_new": c.get("pending_new", 0),
        "termination_gaps_total": termination_gaps,
        "termination_turn_no_snapshot": c.get("termination_turn_no_snapshot", 0),
        "post_termination": c.get("post_termination", 0),
        "unexpected_missing": unexpected,
        "mismatch_error": mismatches,
        "passed": unexpected == 0 and mismatches == 0,
        "by_strategy": analysis["by_strategy"],
        "provenance": analysis["provenance"],
        "mismatch_details": [x for x in analysis["cells"] if x["state"] in ("unexpected_missing", "mismatch_error")],
    }
    json_path = plan_dir / "verification_checklist.json"
    _atomic_write_json(json_path, summary)

    md_lines = [
        f"# 探针计划只读核验清单（{plan_dir.name}）",
        "",
        f"- **计划文件**：`{summary['plan_file']}`",
        f"- **总格数（Total Cells）**：**{summary['total_cells']}**",
        f"- **通过复用核验（`reusable_ok`）**：**{summary['reusable_ok']}**",
        f"- **待补跑/重跑（`pending_new`）**：**{summary['pending_new']}**",
        f"- **已解释终止缺口**：**{summary['termination_gaps_total']}**（`termination_turn_no_snapshot={summary['termination_turn_no_snapshot']}`, `post_termination={summary['post_termination']}`）",
        f"- **意外缺失（`unexpected_missing`）**：**{unexpected}**",
        f"- **复用口径失配（`mismatch_error`）**：**{mismatches}**",
        f"- **核验结论**：**{'PASS' if summary['passed'] else 'FAIL'}**",
        "",
        "## 分策略统计",
        "",
        "| 策略 | reusable_ok | pending_new | termination_turn_no_snapshot | post_termination | unexpected_missing | mismatch_error |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for st, sc in sorted(analysis["by_strategy"].items()):
        md_lines.append(
            f"| {st} | {sc.get('reusable_ok', 0)} | {sc.get('pending_new', 0)} | "
            f"{sc.get('termination_turn_no_snapshot', 0)} | {sc.get('post_termination', 0)} | "
            f"{sc.get('unexpected_missing', 0)} | {sc.get('mismatch_error', 0)} |"
        )
    md_lines.extend(["", "## 题目 × 策略来源批次溯源", "", "| 题目 | 策略 | 来源批次 (`source_batch`) | Run ID | 原终态 | 计划轮数 |", "| :--- | :--- | :--- | :--- | :--- | ---: |"])
    for p in analysis["provenance"]:
        md_lines.append(
            f"| `{p['question_id']}` | `{p['strategy']}` | `{p['source_batch']}` | `{p['run_id']}` | `{p['run_status']}` | {p['turns_planned']} |"
        )
    md_path = plan_dir / "verification_checklist.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _execute_one_probe(
    cell: dict[str, Any],
    analysis: dict[str, Any],
    client: LLMClient,
    write_lock: threading.Lock,
) -> dict[str, Any]:
    plan = analysis["plan"]
    plan_dir = analysis["plan_dir"]
    qid, strategy, turn = cell["question_id"], cell["strategy"], int(cell["turn_id"])
    source_batch = cell["source_batch"]
    run_dir, manifest = analysis["manifest_cache"][(qid, strategy)]
    adapted = analysis["adapted_cache"][qid]
    exchanges = analysis["exchanges_cache"][qid]
    budget = BudgetConfig(**manifest["effective_budget"])
    spec = plan["questions"][qid]
    probe = replace(adapted.probes[0], active_from_turn=_resolve_turns(spec, len(exchanges))[0], expected=spec["expected"])
    snapshot = _load_snapshot_with_archive(
        plan, run_dir, strategy, turn, budget, exchanges, adapted, analysis["legacy_archive_cache"], qid,
    )
    readout = reconstruct_probe_inputs(strategy, snapshot, probe.question, adapted.question_date, budget, exchanges)

    row, tokens_in, tokens_out = run_isolated_probe(
        client,
        plan["model"],
        SYSTEM_PROMPT_LONGMEMEVAL,
        readout["memory_text_injected"],
        readout["kept_exchanges"],
        probe,
        turn,
        snapshot["session_id"],
        manifest["run_id"],
        "offline_snapshot_replay",
        adapted.question_date,
    )
    verdict = (row["judge_raw"] or "").strip().lower().rstrip(".")
    correct_bool = {"yes": True, "no": False}.get(verdict)
    is_gap = row.get("status") != "ok" or correct_bool is None
    record_status = "api_error_gap" if is_gap else "ok"
    row.update(
        status=record_status,
        correct=None if is_gap else correct_bool,
        source_batch=source_batch,
        strategy=strategy,
        question_id=qid,
        retrieved_turn_ids=readout["retrieved_turn_ids"],
        archived_turn_ids=readout["archived_turn_ids"],
        probe_tokens_in=tokens_in,
        probe_tokens_out=tokens_out,
        probe_params=_expected_probe_params(plan["provider"]),
        judge_prompt_commit=plan.get("judge_prompt_commit", FROZEN_JUDGE_PROMPT_COMMIT),
    )
    target = plan_dir / "records" / qid / strategy / f"{turn:03d}.json"
    with write_lock:
        _atomic_write_json(target, row)
    return row


def _classify_recheck(first_rec: dict[str, Any], new_rec: dict[str, Any]) -> str:
    same_verdict = first_rec.get("correct") == new_rec.get("correct")
    if same_verdict:
        return "consistent"
    same_answer = normalize_text(first_rec.get("hypothesis") or "") == normalize_text(new_rec.get("hypothesis") or "")
    return "scoring_inconsistent" if same_answer else "readout_instability"


def _run_rechecks(
    targets: list[tuple[str, str, int]],
    analysis: dict[str, Any],
    client: LLMClient,
) -> None:
    plan = analysis["plan"]
    plan_dir = analysis["plan_dir"]
    for qid, strategy, turn in targets:
        source_batch = _resolve_source_batch(plan, qid, strategy)
        run_dir, manifest = analysis["manifest_cache"][(qid, strategy)]
        adapted = analysis["adapted_cache"][qid]
        exchanges = analysis["exchanges_cache"][qid]
        budget = BudgetConfig(**manifest["effective_budget"])
        spec = plan["questions"][qid]
        probe = replace(adapted.probes[0], active_from_turn=_resolve_turns(spec, len(exchanges))[0], expected=spec["expected"])
        snapshot = _load_snapshot_with_archive(
            plan, run_dir, strategy, turn, budget, exchanges, adapted, analysis["legacy_archive_cache"], qid,
        )
        readout = reconstruct_probe_inputs(strategy, snapshot, probe.question, adapted.question_date, budget, exchanges)
        _, first_rec = _find_saved_record(plan_dir, plan, qid, strategy, turn)
        if first_rec is None:
            raise SystemExit(f"Cannot --recheck without primary record for {qid}:{strategy}:{turn}")

        row, tokens_in, tokens_out = run_isolated_probe(
            client,
            plan["model"],
            SYSTEM_PROMPT_LONGMEMEVAL,
            readout["memory_text_injected"],
            readout["kept_exchanges"],
            probe,
            turn,
            snapshot["session_id"],
            manifest["run_id"],
            "offline_snapshot_recheck",
            adapted.question_date,
        )
        verdict = (row["judge_raw"] or "").strip().lower().rstrip(".")
        correct_bool = {"yes": True, "no": False}.get(verdict)
        is_gap = row.get("status") != "ok" or correct_bool is None
        row.update(
            status="api_error_gap" if is_gap else "ok",
            correct=None if is_gap else correct_bool,
            source_batch=source_batch,
            strategy=strategy,
            question_id=qid,
            retrieved_turn_ids=readout["retrieved_turn_ids"],
            archived_turn_ids=readout["archived_turn_ids"],
            probe_tokens_in=tokens_in,
            probe_tokens_out=tokens_out,
            probe_params=_expected_probe_params(plan["provider"]),
            judge_prompt_commit=plan.get("judge_prompt_commit", FROZEN_JUDGE_PROMPT_COMMIT),
            primary_request_sha256=first_rec.get("request_sha256"),
            primary_correct=first_rec.get("correct"),
            recheck_classification="api_error_gap" if is_gap else _classify_recheck(first_rec, row),
        )
        out_file = plan_dir / "rechecks" / qid / strategy / f"{turn:03d}.jsonl"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[RECHECK] {qid} {strategy} T{turn}: {row['recheck_classification']} (primary={first_rec.get('correct')}, recheck={row['correct']})", flush=True)


def _run_skip_maintenance(
    targets: list[tuple[str, str, int]],
    analysis: dict[str, Any],
    client: LLMClient,
) -> None:
    plan = analysis["plan"]
    plan_dir = analysis["plan_dir"]
    for qid, strategy, turn in targets:
        if turn < 2:
            raise SystemExit(f"Skip-maintenance counterfactual requires turn >= 2, got {turn}")
        source_batch = _resolve_source_batch(plan, qid, strategy)
        run_dir, manifest = analysis["manifest_cache"][(qid, strategy)]
        adapted = analysis["adapted_cache"][qid]
        exchanges = analysis["exchanges_cache"][qid]
        budget = BudgetConfig(**manifest["effective_budget"])
        spec = plan["questions"][qid]
        probe = replace(adapted.probes[0], active_from_turn=_resolve_turns(spec, len(exchanges))[0], expected=spec["expected"])
        prev_snap = _load_snapshot_with_archive(
            plan, run_dir, strategy, turn - 1, budget, exchanges, adapted, analysis["legacy_archive_cache"], qid,
        )
        rec = reconstruct_skip_maintenance_inputs(
            strategy, prev_snap, exchanges[turn], probe.question, adapted.question_date, budget, exchanges,
        )
        out_path = plan_dir / "counterfactuals" / qid / strategy / f"{turn:03d}.json"
        if not rec["fits_budget"]:
            _atomic_write_json(out_path, {
                "question_id": qid,
                "strategy": strategy,
                "turn_id": turn,
                "source_batch": source_batch,
                "run_id": manifest["run_id"],
                "status": "skip_maintenance_over_budget",
                "fits_budget": False,
                "estimated_tokens": rec["estimated_tokens"],
                "budget_cap_tokens": rec["budget_cap_tokens"],
                "request_sha256": rec["request_sha256"],
                "correct": None,
            })
            print(f"[SKIP-MAINT] {qid} {strategy} T{turn}: over_budget ({rec['estimated_tokens']}>{rec['budget_cap_tokens']})", flush=True)
            continue

        row, tokens_in, tokens_out = run_isolated_probe(
            client,
            plan["model"],
            SYSTEM_PROMPT_LONGMEMEVAL,
            rec["memory_text_injected"],
            rec["kept_exchanges"],
            probe,
            turn,
            exchanges[turn].session_id,
            manifest["run_id"],
            "skip_maintenance_counterfactual",
            adapted.question_date,
        )
        verdict = (row["judge_raw"] or "").strip().lower().rstrip(".")
        correct_bool = {"yes": True, "no": False}.get(verdict)
        is_gap = row.get("status") != "ok" or correct_bool is None
        row.update(
            status="api_error_gap" if is_gap else "ok",
            correct=None if is_gap else correct_bool,
            fits_budget=True,
            estimated_tokens=rec["estimated_tokens"],
            budget_cap_tokens=rec["budget_cap_tokens"],
            source_batch=source_batch,
            strategy=strategy,
            question_id=qid,
            retrieved_turn_ids=rec["retrieved_turn_ids"],
            archived_turn_ids=rec["archived_turn_ids"],
            probe_tokens_in=tokens_in,
            probe_tokens_out=tokens_out,
            probe_params=_expected_probe_params(plan["provider"]),
            judge_prompt_commit=plan.get("judge_prompt_commit", FROZEN_JUDGE_PROMPT_COMMIT),
        )
        _atomic_write_json(out_path, row)
        print(f"[SKIP-MAINT] {qid} {strategy} T{turn}: correct={row['correct']}", flush=True)


def _parse_triples(spec: str | None) -> list[tuple[str, str, int]]:
    if not spec:
        return []
    out: list[tuple[str, str, int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise SystemExit(f"Expected QID:STRATEGY:TURN, got {item!r}")
        out.append((parts[0], parts[1], int(parts[2])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Read-only full cell & reuse verification + checklist export")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retry-rounds", type=int, default=3, help="Automatic fallback retry rounds for api_error_gap rows")
    parser.add_argument("--retry-workers", type=int, default=2, help="Lower worker count used during fallback retry rounds")
    parser.add_argument("--recheck", default=None, help="Comma-separated QID:STRATEGY:TURN targets for independent recheck")
    parser.add_argument("--skip-maintenance", default=None, help="Comma-separated QID:STRATEGY:TURN targets for counterfactual readout")
    args = parser.parse_args()

    plan_path = args.plan if args.plan.is_absolute() else (Path.cwd() / args.plan)
    plan = _load_json(plan_path)
    dataset = ROOT / plan["dataset"]
    with dataset.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        items = {x["question_id"]: x for x in json.load(source) if x["question_id"] in plan["questions"]}

    analysis = analyze_plan(plan_path, items, digest)
    json_ck, md_ck = _write_verification_checklist(plan_path, analysis)
    c = analysis["counts"]
    print(
        f"Checklist written: {md_ck}\n"
        f"Summary: total={analysis['total_cells']} reusable_ok={c.get('reusable_ok', 0)} "
        f"pending_new={c.get('pending_new', 0)} "
        f"termination_gaps={c.get('termination_turn_no_snapshot', 0) + c.get('post_termination', 0)} "
        f"(term_turn={c.get('termination_turn_no_snapshot', 0)}, post_term={c.get('post_termination', 0)}) "
        f"unexpected_missing={c.get('unexpected_missing', 0)} mismatch_error={c.get('mismatch_error', 0)}",
        flush=True,
    )

    if c.get("unexpected_missing", 0) > 0 or c.get("mismatch_error", 0) > 0:
        raise SystemExit(f"Verification failed: unexpected_missing={c.get('unexpected_missing', 0)}, mismatch_error={c.get('mismatch_error', 0)}")
    if args.dry_run:
        return

    recheck_targets = _parse_triples(args.recheck)
    skip_targets = _parse_triples(args.skip_maintenance)
    if recheck_targets or skip_targets:
        client = LLMClient(provider=plan["provider"])
        if recheck_targets:
            _run_rechecks(recheck_targets, analysis, client)
        if skip_targets:
            _run_skip_maintenance(skip_targets, analysis, client)
        return

    # Persist explicit termination gap markers for complete records/ accounting
    plan_dir = analysis["plan_dir"]
    for cell in analysis["cells"]:
        if cell["state"] in ("termination_turn_no_snapshot", "post_termination"):
            gap_path = plan_dir / "records" / cell["question_id"] / cell["strategy"] / f"{cell['turn_id']:03d}.json"
            if not gap_path.exists():
                _atomic_write_json(gap_path, {
                    "question_id": cell["question_id"],
                    "strategy": cell["strategy"],
                    "turn_id": cell["turn_id"],
                    "source_batch": cell["source_batch"],
                    "run_id": cell["run_id"],
                    "status": cell["state"],
                    "correct": None,
                    **cell["gap_info"],
                })

    pending = [cell for cell in analysis["cells"] if cell["state"] == "pending_new"]
    if not pending:
        print("All non-gap checkpoints already verified and saved.", flush=True)
        return

    client = LLMClient(provider=plan["provider"])
    write_lock = threading.Lock()

    def _run_pass(target_cells: list[dict[str, Any]], workers: int, label: str) -> tuple[int, int]:
        done_cnt = 0
        gaps_cnt = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_execute_one_probe, cell, analysis, client, write_lock): cell for cell in target_cells}
            for fut in concurrent.futures.as_completed(futs):
                row = fut.result()
                done_cnt += 1
                if row["status"] == "api_error_gap":
                    gaps_cnt += 1
                if done_cnt % 20 == 0 or done_cnt == len(target_cells):
                    print(f"[{label}] Progress: {done_cnt}/{len(target_cells)} (api_error_gaps={gaps_cnt})", flush=True)
        return done_cnt, gaps_cnt

    done, gaps_hit = _run_pass(pending, args.workers, "pass-1")
    print(f"Pass-1 finished {done} probes (api_error_gaps={gaps_hit}).", flush=True)

    for r_idx in range(1, max(0, args.retry_rounds) + 1):
        if gaps_hit == 0:
            break
        print(f"Waiting 5s before fallback retry round {r_idx} with workers={args.retry_workers}...", flush=True)
        time.sleep(5)
        analysis = analyze_plan(plan_path, items, digest)
        retry_cells = [cell for cell in analysis["cells"] if cell["state"] == "pending_new"]
        if not retry_cells:
            gaps_hit = 0
            break
        r_done, gaps_hit = _run_pass(retry_cells, max(1, args.retry_workers), f"retry-{r_idx}")
        print(f"Retry-{r_idx} finished {r_done} probes (remaining api_error_gaps={gaps_hit}).", flush=True)

    final_analysis = analyze_plan(plan_path, items, digest)
    _, final_md = _write_verification_checklist(plan_path, final_analysis)
    fc = final_analysis["counts"]
    print(
        f"Final checklist updated: {final_md}\n"
        f"Final Summary: total={final_analysis['total_cells']} reusable_ok={fc.get('reusable_ok', 0)} "
        f"pending_new={fc.get('pending_new', 0)} "
        f"termination_gaps={fc.get('termination_turn_no_snapshot', 0) + fc.get('post_termination', 0)} "
        f"unexpected_missing={fc.get('unexpected_missing', 0)} mismatch_error={fc.get('mismatch_error', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
