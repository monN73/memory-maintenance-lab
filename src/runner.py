"""Direct (no-Hermes) runner for scenario smoke tests and LongMemEval benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import concurrent.futures
import hashlib
import subprocess
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from artifact_paths import model_directory
from budget_executor import (
    ARCHIVE_POLICY_BY_STRATEGY,
    ARCHIVE_STRATEGIES,
    BOUNDED_ARCHIVE_STRATEGIES,
    EXECUTOR_VERSION,
    FACT_STRATEGIES,
    FULL_HISTORY_STRATEGIES,
    LLM_STRATEGIES,
    MEMORY_STRATEGIES,
    STRATEGIES,
    BudgetConfig,
    Exchange,
    apply_maintenance,
    bm25_retrieve,
    bound_archive,
    bound_archive_fifo,
    build_messages,
    build_readout_memory,
    fact_memory_retrieve,
    load_compaction_prompt_template,
)
from dataset_adapters import UnifiedItem, UnifiedProbe, adapt_longmemeval_item, adapt_unified_record
from eval_and_report import get_anscheck_prompt
from llm_client import DEFAULT_MODELS, LLMClient
from metrics import (
    NORMALIZATION_VERSION,
    TOKENIZER_VERSION,
    duplicate_ngram_rate,
    estimate_text_tokens,
    lexical_jaccard,
    sha256_json,
    sha256_text,
)

ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_SCENARIO = (
    "你是一个严谨的对话助手。请根据当前对话历史和显式记忆准确回答用户问题。"
    "如果信息已修订，以最新明确说明为准；如果信息已撤销或未提供，请明确说明未知或已撤销，不要编造。"
)
SYSTEM_PROMPT_LONGMEMEVAL = (
    "You are a helpful assistant. Answer the user's final question accurately based on the conversation "
    "history and memory provided. If the required information was never provided or cannot be determined "
    "from the conversation, clearly state that you do not know / the information is not available."
)


CST_TZ = timezone(timedelta(hours=8))


def utc_now() -> str:
    return datetime.now(CST_TZ).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def should_run_probe(
    schedule: str,
    turn_id: int,
    active_from_turn: int,
    has_events: bool,
    k: int = 5,
) -> bool:
    """Decide whether an isolated snapshot probe should fire after this turn."""
    if schedule == "final_only":
        return False
    if schedule == "every_turn":
        return True
    if turn_id < active_from_turn:
        return False
    if schedule == "on_maintenance":
        return has_events or turn_id == active_from_turn
    if schedule == "every_k_turns":
        return turn_id == active_from_turn or ((turn_id - active_from_turn) % max(1, k) == 0)
    return False


def run_isolated_probe(
    client: LLMClient,
    model: str,
    system_prompt: str,
    memory_text: str,
    kept_exchanges: list[Exchange],
    probe: UnifiedProbe,
    turn_id: int,
    session_id: str,
    run_id: str,
    trigger: str,
    question_date: str | None = None,
    judge_probe: bool = True,
) -> tuple[dict[str, Any], int, int]:
    """Execute a read-only snapshot_readout probe without mutating main trajectory."""
    expected = probe.expected_by_turn.get(str(turn_id), probe.expected)
    probe_q = f"[Current Date: {question_date}]\n{probe.question}" if question_date else probe.question
    probe_msgs = build_messages(system_prompt, memory_text, kept_exchanges, probe_q)
    eval_in = eval_out = 0
    try:
        p_res = client.chat(
            model=model,
            messages=probe_msgs,
            temperature=0.0,
            max_tokens=256,
            seed=42,
            thinking="disabled" if client.provider == "deepseek" else None,
        )
        probe_answer = p_res["text"]
        eval_in += p_res["usage"].get("prompt_tokens") or 0
        eval_out += p_res["usage"].get("completion_tokens") or 0
        probe_status = "ok"
    except Exception as e:
        probe_answer = ""
        probe_status = f"api_error: {e}"
        p_res = {"usage": {}}

    is_correct: bool | None = None
    judge_raw: str | None = None
    if judge_probe and probe_status == "ok" and probe_answer.strip():
        lme_tasks = {
            "single-session-user",
            "single-session-assistant",
            "multi-session",
            "temporal-reasoning",
            "knowledge-update",
            "single-session-preference",
        }
        if probe.question_type in lme_tasks:
            j_prompt = get_anscheck_prompt(
                probe.question_type,
                probe.question,
                expected,
                probe_answer,
                abstention=probe.is_abstention,
            )
        else:
            j_prompt = (
                f"Question: {probe.question}\nCorrect Answer: {expected}\nModel Response: {probe_answer}\n"
                "Is the model response correct? Answer yes or no only."
            )
        try:
            j_res = client.chat(
                model=model,
                messages=[{"role": "user", "content": j_prompt}],
                temperature=0.0,
                max_tokens=256,
                seed=42,
                thinking="disabled" if client.provider == "deepseek" else None,
            )
            judge_raw = j_res["text"]
            eval_in += j_res["usage"].get("prompt_tokens") or 0
            eval_out += j_res["usage"].get("completion_tokens") or 0
            is_correct = "yes" in judge_raw.lower() if judge_raw.strip() else None
        except Exception as e:
            judge_raw = f"judge_error: {e}"

    record = {
        "schema_version": "0.2",
        "run_id": run_id,
        "provider": client.provider,
        "model": model,
        "judge_provider": client.provider if judge_probe else None,
        "judge_model": model if judge_probe else None,
        "turn_id": turn_id,
        "session_id": session_id,
        "probe_id": probe.probe_id,
        "readout_mode": "snapshot_readout",
        "trigger": trigger,
        "active_from_turn": probe.active_from_turn,
        "question": probe.question,
        "expected": expected,
        "hypothesis": probe_answer,
        "correct": is_correct,
        "judge_raw": judge_raw,
        "status": probe_status,
        "request_sha256": sha256_json(probe_msgs),
        "memory_text_injected": memory_text,
        "usage": p_res.get("usage", {}),
    }
    return record, eval_in, eval_out


def run_scenario(
    strategy: str,
    scenario_path: Path,
    gold_path: Path,
    out_dir: Path,
    model: str,
    budget: BudgetConfig,
    client: LLMClient,
) -> dict[str, Any]:
    turns_raw = [json.loads(line) for line in scenario_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    scenario_id = gold["scenario_id"]
    run_id = f"scenario-{scenario_id}-direct-{strategy}-r1"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    comp_tpl, comp_tpl_hash = load_compaction_prompt_template()
    started_at = utc_now()

    memory_text = ""
    recent_exchanges: list[Exchange] = []
    archived_exchanges: list[Exchange] = []
    termination_reason = "completed"
    total_ans_in = total_ans_out = 0
    total_maint_in = total_maint_out = 0
    total_eval_in = total_eval_out = 0
    maint_count = 0

    for item in turns_raw:
        tid = int(item["turn_id"])
        sid = item["session_id"]
        user_text = item["user"]
        memory_before = memory_text
        history_before_ids = [ex.turn_id for ex in recent_exchanges]

        res = apply_maintenance(
            strategy=strategy,
            turn_id=tid,
            run_id=run_id,
            system_prompt=SYSTEM_PROMPT_SCENARIO,
            memory_text=memory_text,
            recent_exchanges=recent_exchanges,
            current_user_text=user_text,
            budget=budget,
            llm_client=client if strategy in LLM_STRATEGIES else None,
            compaction_model=model,
            compaction_prompt_template=comp_tpl,
        )
        m_in, m_out, m_lat = res["maint_usage"]
        total_maint_in += m_in
        total_maint_out += m_out
        for ev in res["events"]:
            ev["provider"] = client.provider if ev.get("mechanism") in ("llm_compaction", "chunked_llm_compaction") else None
            append_jsonl(run_dir / "events.jsonl", ev)
            maint_count += 1

        if res["status"] == "overflow":
            termination_reason = "overflow"
            append_jsonl(
                run_dir / "turns.jsonl",
                {
                    "schema_version": "0.2",
                    "run_id": run_id,
                    "provider": client.provider,
                    "scenario_id": scenario_id,
                    "session_id": sid,
                    "turn_id": tid,
                    "strategy_id": strategy,
                    "track": "direct",
                    "input_text": user_text,
                    "answer_text": None,
                    "status": "overflow",
                    "failure_labels": ["overflow"],
                },
            )
            break

        memory_used = res["memory_text"]
        kept_exchanges = res["recent_exchanges"]
        msgs = res["messages"]
        ctx_hash = sha256_json(msgs)

        ans_res = client.chat(
            model=model,
            messages=msgs,
            temperature=0.0,
            max_tokens=budget.max_output_tokens,
            seed=42,
        )
        ans_text = ans_res["text"]
        a_in = ans_res["usage"].get("prompt_tokens") or 0
        a_out = ans_res["usage"].get("completion_tokens") or 0
        total_ans_in += a_in
        total_ans_out += a_out

        write_json(
            run_dir / "requests" / f"turn_{tid:03d}.json",
            {"turn_id": tid, "provider": client.provider, "messages": msgs, "response": ans_res},
        )
        write_json(
            run_dir / "snapshots" / f"turn_{tid:03d}.json",
            {
                "turn_id": tid,
                "provider": client.provider,
                "history_before_turn_ids": history_before_ids,
                "memory_before": memory_before,
                "memory_used_for_answer": memory_used,
                "kept_turn_ids": [ex.turn_id for ex in kept_exchanges],
            },
        )

        # Isolated snapshot_readout probes (do NOT mutate main trajectory)
        probe_ids: list[str] = []
        probe_correct_count = 0
        probe_total = 0
        for p in gold.get("probes", []):
            if tid not in p.get("turns", []):
                continue
            probe_ids.append(p["probe_id"])
            probe_total += 1
            expected = p.get("expected_by_turn", {}).get(str(tid), p.get("expected"))
            probe_msgs = build_messages(
                SYSTEM_PROMPT_SCENARIO,
                memory_used,
                kept_exchanges + [Exchange(turn_id=tid, session_id=sid, user_text=user_text, assistant_text=ans_text)],
                p["question"],
            )
            p_res = client.chat(model=model, messages=probe_msgs, temperature=0.0, max_tokens=256, seed=42)
            total_eval_in += p_res["usage"].get("prompt_tokens") or 0
            total_eval_out += p_res["usage"].get("completion_tokens") or 0

            # Judge probe answer
            judge_prompt = (
                f"问题：{p['question']}\n参考答案：{expected}\n模型回答：{p_res['text']}\n"
                f"规则：{gold.get('grading', '')}\n"
                "请判断模型回答是否正确（若同时包含新旧值但未标明时序则判错）。只回答 yes 或 no。"
            )
            j_res = client.chat(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=256,
                seed=42,
            )
            total_eval_in += j_res["usage"].get("prompt_tokens") or 0
            total_eval_out += j_res["usage"].get("completion_tokens") or 0
            is_correct = "yes" in j_res["text"].lower()
            if is_correct:
                probe_correct_count += 1
            append_jsonl(
                run_dir / "probes.jsonl",
                {
                    "run_id": run_id,
                    "provider": client.provider,
                    "judge_provider": client.provider,
                    "turn_id": tid,
                    "probe_id": p["probe_id"],
                    "readout_mode": "snapshot_readout",
                    "question": p["question"],
                    "expected": expected,
                    "hypothesis": p_res["text"],
                    "correct": is_correct,
                    "judge_raw": j_res["text"],
                },
            )

        adj_overlap, adj_r = lexical_jaccard(memory_before, memory_used) if strategy in MEMORY_STRATEGIES else (None, "no_explicit_memory")
        dup_rate, dup_r = duplicate_ngram_rate(memory_used) if strategy in MEMORY_STRATEGIES else (None, "no_explicit_memory")
        ev_metrics = res["events"][0]["metrics"] if len(res["events"]) == 1 else {"lexical_jaccard": None, "lexical_retention": None, "compression_ratio": None}

        append_jsonl(
            run_dir / "turns.jsonl",
            {
                "schema_version": "0.2",
                "run_id": run_id,
                "provider": client.provider,
                "scenario_id": scenario_id,
                "session_id": sid,
                "turn_id": tid,
                "strategy_id": strategy,
                "track": "direct",
                "input_text": user_text,
                "answer_text": ans_text,
                "status": res["status"],
                "snapshots": {
                    "history_before_turn_ids": history_before_ids,
                    "memory_before": memory_before,
                    "memory_used_for_answer": memory_used,
                    "context_sha256": ctx_hash,
                },
                "event_ids": [e["event_id"] for e in res["events"]],
                "probe_ids": probe_ids,
                "usage": {
                    "input_tokens_estimate": res["estimated_tokens"],
                    "answer_input_tokens_actual": a_in,
                    "answer_output_tokens_actual": a_out,
                    "maintenance_input_tokens_actual": m_in,
                    "maintenance_output_tokens_actual": m_out,
                },
                "metrics": {
                    "lexical_jaccard": ev_metrics["lexical_jaccard"],
                    "lexical_retention": ev_metrics["lexical_retention"],
                    "adjacent_memory_overlap": adj_overlap,
                    "duplicate_ngram_rate": dup_rate,
                    "probe_accuracy": round(probe_correct_count / probe_total, 4) if probe_total > 0 else None,
                    "compression_ratio": ev_metrics["compression_ratio"],
                },
                "normalization_version": NORMALIZATION_VERSION,
                "memory_scope": "external" if strategy in MEMORY_STRATEGIES else "none",
            },
        )

        # Advance main trajectory state
        memory_text = memory_used
        recent_exchanges = kept_exchanges + [
            Exchange(turn_id=tid, session_id=sid, user_text=user_text, assistant_text=ans_text)
        ]

    summary = {
        "run_id": run_id,
        "provider": client.provider,
        "answer_model": model,
        "scenario_id": scenario_id,
        "strategy": strategy,
        "status": "completed" if termination_reason == "completed" else termination_reason,
        "termination_reason": termination_reason,
        "maintenance_events": maint_count,
        "tokens": {
            "answer_in": total_ans_in,
            "answer_out": total_ans_out,
            "maint_in": total_maint_in,
            "maint_out": total_maint_out,
            "eval_in": total_eval_in,
            "eval_out": total_eval_out,
        },
    }
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "0.2",
            "run_id": run_id,
            "protocol_version": "0.3",
            "experiment_cell_id": f"direct-{strategy}",
            "strategy_spec_id": strategy,
            "integration_mode": "direct",
            "provider": client.provider,
            "endpoint_hash": client.endpoint_hash,
            "answer_model": model,
            "maintenance_model": model if strategy in LLM_STRATEGIES else None,
            "maintenance_thinking": "disabled" if strategy in LLM_STRATEGIES and client.provider == "deepseek" else None,
            "answer_thinking": "disabled" if client.provider == "deepseek" else None,
            "compaction_prompt_hash": comp_tpl_hash if strategy in LLM_STRATEGIES else None,
            "budget_executor_version": EXECUTOR_VERSION,
            "effective_budget": asdict(budget),
            "n1_policy": "full_history_model_capacity" if strategy in FULL_HISTORY_STRATEGIES else None,
            "n4_policy": "llm_only_one_shorten_retry_no_fallback" if strategy in LLM_STRATEGIES else None,
            "tokenizer": TOKENIZER_VERSION,
            "started_at": started_at,
            "finished_at": utc_now(),
            "status": summary["status"],
            "termination_reason": termination_reason,
        },
    )
    write_json(run_dir / "summary.json", summary)
    return summary


class LiveRows(list):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def append(self, row):
        append_jsonl(self.path, row)
        super().append(row)


def source_identity():
    paths = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "prompts").glob("*.md"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "source_sha256": digest.hexdigest(), "source_hash_scope": "src/*.py and prompts/*.md"}


def run_longmemeval_item(
    item: dict[str, Any],
    strategy: str,
    batch_dir: Path,
    model: str,
    budget: BudgetConfig,
    dataset_meta: dict[str, Any],
    client: LLMClient,
    comp_tpl: str,
    comp_tpl_hash: str,
    probe_schedule: str = "final_only",
    probe_k: int = 5,
    judge_probes: bool = True,
    save_snapshots: bool = False,
    unified: UnifiedItem | None = None,
    probe_turns: set[int] | None = None,
) -> dict[str, Any]:
    adapted = unified or adapt_longmemeval_item(item, dataset_name=dataset_meta.get("name", "LongMemEval"))
    qid = adapted.item_id
    run_id = f"lme-{dataset_meta['variant']}-{qid}-direct-{strategy}-r1"
    run_dir = batch_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    identity = source_identity()
    write_json(run_dir / "manifest.json", {"run_id": run_id, "status": "running", "started_at": started_at, "provider": client.provider, "answer_model": model, "dataset": dataset_meta | {"question_id": qid}, "effective_budget": asdict(budget), "budget_executor_version": EXECUTOR_VERSION, "compaction_prompt_hash": comp_tpl_hash if strategy in LLM_STRATEGIES else None, "source_identity": identity, "probe_schedule": probe_schedule, "probe_k": probe_k, "probe_turns": sorted(probe_turns) if probe_turns else None, "judge_probes": judge_probes, "save_snapshots": save_snapshots})
    def progress(event):
        append_jsonl(run_dir / "progress.jsonl", {"timestamp": utc_now(), "run_id": run_id, **event})
        print(f"[{run_id}] {event}", flush=True)
    progress({"phase": "run_start"})

    exchanges: list[Exchange] = [
        Exchange(
            turn_id=t.turn_id,
            session_id=t.session_id,
            user_text=t.user_text,
            assistant_text=t.assistant_text,
            timestamp=t.timestamp,
            source_message_indices=t.source_message_indices,
        )
        for t in adapted.turns
    ]

    memory_text = ""
    recent_exchanges: list[Exchange] = []
    archived_exchanges: list[Exchange] = []
    termination_reason = "completed"
    total_maint_in = total_maint_out = total_maint_lat = 0
    total_eval_in = total_eval_out = 0
    maint_count = 0
    turns_lines = LiveRows(run_dir / "turns.jsonl")
    events_lines = LiveRows(run_dir / "events.jsonl")
    probes_lines = LiveRows(run_dir / "probes.jsonl")
    cumulative_ingested_tokens = 0

    # Ingestion phase (no final question leaked, no answer model called)
    for ex in exchanges:
        cumulative_ingested_tokens += estimate_text_tokens(ex.user_text) + estimate_text_tokens(ex.assistant_text or "")
        memory_before = memory_text
        # Check budget on candidate context including this exchange
        res = apply_maintenance(
            strategy=strategy,
            turn_id=ex.turn_id,
            run_id=run_id,
            system_prompt=SYSTEM_PROMPT_LONGMEMEVAL,
            memory_text=memory_text,
            recent_exchanges=recent_exchanges,
            current_user_text=f"[{ex.timestamp}] {ex.user_text}",
            current_assistant_text=ex.assistant_text,
            progress_callback=progress,
            budget=budget,
            llm_client=client if strategy in LLM_STRATEGIES else None,
            compaction_model=model,
            compaction_prompt_template=comp_tpl,
            archived_exchanges=archived_exchanges,
        )
        m_in, m_out, m_lat = res["maint_usage"]
        total_maint_in += m_in
        total_maint_out += m_out
        total_maint_lat += m_lat
        for ev in res["events"]:
            ev["provider"] = client.provider if ev.get("mechanism") in ("llm_compaction", "chunked_llm_compaction") else None
            events_lines.append(ev)
            maint_count += 1

        adj_overlap, _ = lexical_jaccard(memory_before, res["memory_text"]) if strategy in MEMORY_STRATEGIES else (None, "no_explicit_memory")
        ev_metrics = res["events"][0]["metrics"] if len(res["events"]) == 1 else {"lexical_jaccard": None, "lexical_retention": None, "compression_ratio": None}

        turns_lines.append({
            "schema_version": "0.2",
            "run_id": run_id,
            "provider": client.provider,
            "session_id": ex.session_id,
            "turn_id": ex.turn_id,
            "strategy_id": strategy,
            "track": "direct",
            "phase": "ingestion",
            "source_message_indices": ex.source_message_indices,
            "input_text": ex.user_text,
            "replayed_output_text": ex.assistant_text,
            "output_origin": "dataset_replay",
            "cumulative_ingested_tokens_est": cumulative_ingested_tokens,
            "context_tokens_est": res["estimated_tokens"],
            "memory_tokens_est": estimate_text_tokens(res["memory_text"]),
            "answer_text": None,
            "status": res["status"],
            "event_ids": [e["event_id"] for e in res["events"]],
            "metrics": {
                "lexical_jaccard": ev_metrics["lexical_jaccard"],
                "lexical_retention": ev_metrics["lexical_retention"],
                "adjacent_memory_overlap": adj_overlap,
                "compression_ratio": ev_metrics["compression_ratio"],
            },
        })

        if res["status"] in ("overflow", "maintenance_budget_error", "maintenance_empty_error", "maintenance_truncated_error", "maintenance_api_error"):
            termination_reason = res["status"]
            break

        memory_text = res["memory_text"]
        if strategy in ARCHIVE_STRATEGIES and res.get("evicted_exchanges"):
            archived_exchanges.extend(res["evicted_exchanges"])
            if strategy in BOUNDED_ARCHIVE_STRATEGIES:
                archived_exchanges, _ = bound_archive(
                    archived_exchanges,
                    budget.n8_archive_tokens,
                    policy=ARCHIVE_POLICY_BY_STRATEGY[strategy],
                )
        if strategy == "N0":
            recent_exchanges = []
        else:
            recent_exchanges = res["recent_exchanges"] + [ex]

        if save_snapshots or probe_schedule != "final_only" or probe_turns:
            write_json(
                run_dir / "snapshots" / f"turn_{ex.turn_id:03d}.json",
                {
                    "turn_id": ex.turn_id,
                    "session_id": ex.session_id,
                    "provider": client.provider,
                    "memory_text": memory_text,
                    "archived_turn_ids": [k.turn_id for k in archived_exchanges],
                    "kept_exchanges": [
                        {
                            "turn_id": k.turn_id,
                            "session_id": k.session_id,
                            "timestamp": k.timestamp,
                            "user_text": k.user_text,
                            "assistant_text": k.assistant_text,
                        }
                        for k in recent_exchanges
                    ],
                },
            )

        for probe in adapted.probes:
            hit_checkpoint = bool(probe_turns and ex.turn_id in probe_turns and ex.turn_id >= probe.active_from_turn)
            if hit_checkpoint or should_run_probe(
                probe_schedule,
                turn_id=ex.turn_id,
                active_from_turn=probe.active_from_turn,
                has_events=bool(res["events"]),
                k=probe_k,
            ):
                probe_mem, probe_ret_ids = build_readout_memory(
                    strategy,
                    probe.question,
                    memory_text,
                    archived_exchanges if strategy in ARCHIVE_STRATEGIES else None,
                    budget.memory_tokens,
                )
                p_rec, p_in, p_out = run_isolated_probe(
                    client=client,
                    model=model,
                    system_prompt=SYSTEM_PROMPT_LONGMEMEVAL,
                    memory_text=probe_mem,
                    kept_exchanges=recent_exchanges,
                    probe=probe,
                    turn_id=ex.turn_id,
                    session_id=ex.session_id,
                    run_id=run_id,
                    trigger="checkpoint" if hit_checkpoint else ("maintenance" if res["events"] else probe_schedule),
                    question_date=adapted.question_date,
                    judge_probe=judge_probes,
                )
                p_rec["retrieved_turn_ids"] = probe_ret_ids
                p_rec["archived_turn_ids"] = [k.turn_id for k in archived_exchanges]
                total_eval_in += p_in
                total_eval_out += p_out
                probes_lines.append(p_rec)

    hypothesis = None
    ans_in = ans_out = ans_lat = 0
    search_lat = 0
    final_ctx_tokens_est = 0

    if termination_reason == "completed":
        final_turn_id = len(exchanges) + 1
        final_q = f"[Current Date: {adapted.question_date}]\n{adapted.question}" if adapted.question_date else adapted.question
        res = apply_maintenance(
            strategy=strategy,
            turn_id=final_turn_id,
            run_id=run_id,
            system_prompt=SYSTEM_PROMPT_LONGMEMEVAL,
            memory_text=memory_text,
            recent_exchanges=recent_exchanges,
            current_user_text=final_q,
            progress_callback=progress,
            budget=budget,
            llm_client=client if strategy in LLM_STRATEGIES else None,
            compaction_model=model,
            compaction_prompt_template=comp_tpl,
            archived_exchanges=archived_exchanges,
            is_query=True,
        )
        m_in, m_out, m_lat = res["maint_usage"]
        search_lat = res.get("search_latency_ms", 0)
        total_maint_in += m_in
        total_maint_out += m_out
        total_maint_lat += m_lat
        for ev in res["events"]:
            ev["provider"] = client.provider if ev.get("mechanism") in ("llm_compaction", "chunked_llm_compaction") else None
            events_lines.append(ev)
            maint_count += 1

        final_ctx_tokens_est = res["estimated_tokens"]
        if res["status"] in ("overflow", "maintenance_budget_error", "maintenance_empty_error", "maintenance_truncated_error", "maintenance_api_error"):
            termination_reason = res["status"]
        else:
            try:
                ans_res = client.chat(
                    model=model,
                    messages=res["messages"],
                    temperature=0.0,
                    max_tokens=budget.max_output_tokens,
                    seed=42,
                    thinking="disabled" if client.provider == "deepseek" else None,
                )
                hypothesis = ans_res["text"]
                ans_in = ans_res["usage"].get("prompt_tokens") or 0
                ans_out = ans_res["usage"].get("completion_tokens") or 0
                ans_lat = ans_res["latency_ms"]
                write_json(
                    run_dir / "requests" / "final_answer.json",
                    {"turn_id": final_turn_id, "provider": client.provider, "messages": res["messages"], "response": ans_res},
                )
            except Exception as e:
                termination_reason = f"api_error: {e}"

        turns_lines.append({
            "schema_version": "0.2",
            "run_id": run_id,
            "provider": client.provider,
            "session_id": "final_question",
            "turn_id": final_turn_id,
            "strategy_id": strategy,
            "track": "direct",
            "phase": "final_answer",
            "input_text": final_q,
            "output_origin": "model_api",
            "cumulative_ingested_tokens_est": cumulative_ingested_tokens,
            "context_tokens_est": final_ctx_tokens_est,
            "memory_tokens_est": estimate_text_tokens(res["memory_text"]),
            "answer_text": hypothesis,
            "status": "ok" if hypothesis is not None else termination_reason,
            "event_ids": [e["event_id"] for e in res["events"]],
        })

    first_failure_turn = next(
        (p["turn_id"] for p in probes_lines if p.get("correct") is False),
        None,
    )

    write_json(
        run_dir / "snapshots" / "final_state.json",
        {
            "provider": client.provider,
            "memory_text": memory_text,
            "kept_turn_ids": [ex.turn_id for ex in recent_exchanges],
            "total_exchanges": len(exchanges),
            "cumulative_ingested_tokens_est": cumulative_ingested_tokens,
        },
    )

    status = "completed" if termination_reason == "completed" and hypothesis is not None else termination_reason
    summary = {
        "run_id": run_id,
        "provider": client.provider,
        "answer_model": model,
        "question_id": qid,
        "question_type": adapted.question_type,
        "is_abstention": adapted.is_abstention,
        "strategy": strategy,
        "status": status,
        "termination_reason": termination_reason,
        "hypothesis": hypothesis,
        "total_exchanges": len(exchanges),
        "cumulative_ingested_tokens_est": cumulative_ingested_tokens,
        "final_ctx_tokens_est": final_ctx_tokens_est,
        "maintenance_events": maint_count,
        "probe_schedule": probe_schedule,
        "probe_count": len(probes_lines),
        "first_failure_turn": first_failure_turn,
        "tokens": {
            "answer_in": ans_in,
            "answer_out": ans_out,
            "maint_in": total_maint_in,
            "maint_out": total_maint_out,
            "probe_eval_in": total_eval_in,
            "probe_eval_out": total_eval_out,
            "total_in": ans_in + total_maint_in,
            "total_out": ans_out + total_maint_out,
        },
        "latency_ms": {
            "maintenance": total_maint_lat,
            "search": search_lat,
            "answer": ans_lat,
        },
    }
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "0.2",
            "run_id": run_id,
            "protocol_version": "0.3",
            "experiment_cell_id": f"direct-{strategy}",
            "strategy_spec_id": strategy,
            "integration_mode": "direct",
            "dataset": dataset_meta | {"question_id": qid},
            "source_identity": identity,
            "probe_schedule": probe_schedule, "probe_k": probe_k, "judge_probes": judge_probes, "save_snapshots": save_snapshots,
            "provider": client.provider,
            "endpoint_hash": client.endpoint_hash,
            "answer_model": model,
            "maintenance_model": model if strategy in LLM_STRATEGIES else None,
            "maintenance_thinking": "disabled" if strategy in LLM_STRATEGIES and client.provider == "deepseek" else None,
            "answer_thinking": "disabled" if client.provider == "deepseek" else None,
            "compaction_prompt_hash": comp_tpl_hash if strategy in LLM_STRATEGIES else None,
            "budget_executor_version": EXECUTOR_VERSION,
            "effective_budget": asdict(budget),
            "n1_policy": "full_history_model_capacity" if strategy in FULL_HISTORY_STRATEGIES else None,
            "n4_policy": "llm_only_one_shorten_retry_no_fallback" if strategy in LLM_STRATEGIES else None,
            "tokenizer": TOKENIZER_VERSION,
            "started_at": started_at,
            "finished_at": utc_now(),
            "status": status,
            "termination_reason": termination_reason,
        },
    )
    write_json(run_dir / "summary.json", summary)
    progress({"phase": "run_end", "status": status})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct memory maintenance runner")
    parser.add_argument("--mode", choices=["scenario", "longmemeval", "unified"], required=True)
    parser.add_argument("--strategies", default="N0,N1,N2,N3,N4")
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="deepseek")
    parser.add_argument("--model", default=None, help="Model ID; defaults to the selected provider model")
    parser.add_argument("--budget", type=int, default=8192, help="Input budget for N0/N2/N3/N4; does not cap N1")
    parser.add_argument("--chunk-tokens", type=int, default=2048, help="Estimated serialized history tokens per chunk for N5/N6")
    parser.add_argument("--n8-archive-tokens", type=int, default=16384, help="Bounded archive token cap for N8-FIFO")
    parser.add_argument("--n1-context-tokens", type=int, help="Verified total model context capacity for N1, including answer reserve")
    parser.add_argument("--n4-output-tokens", type=int, default=4096, help="API generation cap, distinct from retained memory budget")
    parser.add_argument("--n4-target-ratio", type=float, default=0.65)
    parser.add_argument("--n4-retry-ratio", type=float, default=0.40)
    parser.add_argument("--batch-id", default=None, help="Optional logical batch prefix; auto-generated if omitted")
    parser.add_argument("--dataset", help="Path to LongMemEval or unified JSON/JSONL")
    parser.add_argument("--manifest", help="Path to subset manifest JSON (optional for unified mode)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--probe-schedule",
        choices=["final_only", "on_maintenance", "every_k_turns", "every_turn"],
        default="final_only",
        help="When to run isolated snapshot_readout probes during ingestion",
    )
    parser.add_argument("--probe-k", type=int, default=5, help="Turn interval for --probe-schedule every_k_turns")
    parser.add_argument("--probe-turns", default=None, help="Comma-separated common turn_ids to probe regardless of schedule")
    parser.add_argument("--no-probe-judge", action="store_true", help="Record probe answers without calling judge")
    parser.add_argument("--save-snapshots", action="store_true", help="Save per-turn memory/history snapshots")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if not strategies or set(strategies) - set(STRATEGIES) or len(set(strategies)) != len(strategies):
        parser.error("Provide a comma-separated list of distinct strategies from N0 to N14")
    if set(FULL_HISTORY_STRATEGIES).intersection(strategies) and (args.n1_context_tokens is None or args.n1_context_tokens <= 512):
        parser.error("N1/N5 require --n1-context-tokens > 512; supply the verified provider/model capacity")
    if args.n8_archive_tokens <= 0 or args.chunk_tokens <= 0 or args.budget <= 0 or args.n4_output_tokens <= 0 or not 0 < args.n4_retry_ratio < args.n4_target_ratio < 1:
        parser.error("Require positive budgets and 0 < n4-retry-ratio < n4-target-ratio < 1")
    budget = BudgetConfig(input_tokens=args.budget, n1_context_tokens=args.n1_context_tokens,
                          n4_output_tokens=args.n4_output_tokens, n4_target_ratio=args.n4_target_ratio,
                          n4_retry_ratio=args.n4_retry_ratio, chunk_tokens=args.chunk_tokens,
                          n8_archive_tokens=args.n8_archive_tokens)
    args.model = args.model or DEFAULT_MODELS[args.provider]
    if not args.batch_id:
        if args.mode == "scenario":
            args.batch_id = "scenario-revision-smoke"
        elif args.manifest:
            args.batch_id = f"{args.mode}-{Path(args.manifest).stem}"
        elif args.dataset:
            args.batch_id = f"{args.mode}-{Path(args.dataset).stem}"
        else:
            args.batch_id = args.mode
    client = LLMClient(provider=args.provider)
    if Path(args.batch_id).name != args.batch_id or args.batch_id in (".", ".."):
        raise SystemExit("--batch-id must be a single directory name.")
    batch_name = f"{args.batch_id}--{client.provider}--{datetime.now(CST_TZ).strftime('%Y%m%dT%H%M%S%f+0800')}"
    batch_dir = ROOT / "runs" / model_directory(args.model) / batch_name
    existing_summary = batch_dir / "batch_summary.json"
    if existing_summary.exists():
        prior = json.loads(existing_summary.read_text(encoding="utf-8"))
        if prior.get("provider") != client.provider or prior.get("answer_model") != args.model:
            raise SystemExit("Batch has a different or unrecorded provider/model; choose a new --batch-id.")
    elif (batch_dir / "runs").exists() and any((batch_dir / "runs").iterdir()):
        raise SystemExit("Batch contains unfinished or unrecorded runs; choose a new --batch-id.")
    batch_dir.mkdir(parents=True, exist_ok=False)
    print(f"Batch directory: {batch_dir}", flush=True)

    if args.mode == "scenario":
        scen_path = ROOT / "data" / "scenarios" / "revision-smoke.jsonl"
        gold_path = ROOT / "data" / "annotations" / "revision-smoke.gold.json"
        summaries = []
        for st in strategies:
            s = run_scenario(st, scen_path, gold_path, batch_dir / "runs", args.model, budget, client)
            summaries.append(s)
            print(f"[{st}] status={s['status']} maint={s['maintenance_events']}", flush=True)
        write_json(batch_dir / "batch_summary.json", {"batch_id": args.batch_id, "provider": client.provider, "answer_model": args.model, "runs": summaries})
        return

    # LongMemEval / Unified mode
    ds_path = Path(args.dataset)
    if not ds_path.is_absolute():
        ds_path = ROOT / ds_path

    if args.mode == "unified":
        raw_text = ds_path.read_text(encoding="utf-8").strip()
        if ds_path.suffix == ".jsonl":
            raw_records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
        else:
            raw_records = json.loads(raw_text)
        unified_items = [adapt_unified_record(r) for r in raw_records]
        if args.manifest:
            mf_path = Path(args.manifest)
            if not mf_path.is_absolute():
                mf_path = ROOT / mf_path
            manifest = json.loads(mf_path.read_text(encoding="utf-8"))
            qids = set(manifest["question_ids"])
            unified_items = [u for u in unified_items if u.item_id in qids]
            unified_items.sort(key=lambda u: manifest["question_ids"].index(u.item_id))
        else:
            manifest = {"question_ids": [u.item_id for u in unified_items], "variant": "unified"}
        dataset_meta = {
            "name": manifest.get("dataset", unified_items[0].dataset if unified_items else "unified"),
            "revision": manifest.get("dataset_revision"),
            "variant": manifest.get("variant", "unified"),
            "file_sha256": manifest.get("source_sha256"),
            "subset_manifest": str(args.manifest or ""),
        }
        items_for_jobs: list[tuple[dict[str, Any], UnifiedItem | None, str]] = [
            ({}, u, u.item_id) for u in unified_items
        ]
    else:
        mf_path = Path(args.manifest)
        if not mf_path.is_absolute():
            mf_path = ROOT / mf_path

        data = json.loads(ds_path.read_text(encoding="utf-8"))
        manifest = json.loads(mf_path.read_text(encoding="utf-8"))
        qids = set(manifest["question_ids"])
        items = [x for x in data if x["question_id"] in qids]
        items.sort(key=lambda x: manifest["question_ids"].index(x["question_id"]))

        dataset_meta = {
            "name": manifest.get("dataset", "LongMemEval"),
            "revision": manifest.get("dataset_revision"),
            "variant": manifest.get("variant"),
            "file_sha256": manifest.get("source_sha256"),
            "subset_manifest": str(mf_path.relative_to(ROOT)),
        }
        items_for_jobs = [(item, None, item["question_id"]) for item in items]

    comp_tpl, comp_tpl_hash = load_compaction_prompt_template()

    jobs = [(raw_item, u_item, item_id, st) for st in strategies for raw_item, u_item, item_id in items_for_jobs]
    print(f"Running {len(jobs)} runs ({len(items_for_jobs)} questions x {len(strategies)} strategies, probe={args.probe_schedule}) with {args.workers} workers...", flush=True)

    probe_turns_set = {int(x.strip()) for x in args.probe_turns.split(",") if x.strip()} if args.probe_turns else None
    summaries_by_strategy: dict[str, list[dict[str, Any]]] = {st: [] for st in strategies}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_map = {
            ex.submit(
                run_longmemeval_item,
                raw_item,
                st,
                batch_dir,
                args.model,
                budget,
                dataset_meta,
                client,
                comp_tpl,
                comp_tpl_hash,
                args.probe_schedule,
                args.probe_k,
                not args.no_probe_judge,
                args.save_snapshots,
                u_item,
                probe_turns_set,
            ): (item_id, st)
            for raw_item, u_item, item_id, st in jobs
        }
        done = 0
        for fut in concurrent.futures.as_completed(fut_map):
            qid, st = fut_map[fut]
            s = fut.result()
            summaries_by_strategy[st].append(s)
            done += 1
            print(f"Progress: {done}/{len(jobs)} ({st} {qid}: {s['status']})", flush=True)

    # Write per-strategy predictions.jsonl in manifest order
    order = {qid: idx for idx, qid in enumerate(manifest["question_ids"])}
    for st, rows in summaries_by_strategy.items():
        rows.sort(key=lambda r: order[r["question_id"]])
        pred_path = batch_dir / f"predictions_{st}.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(
                    json.dumps(
                        {
                            "question_id": r["question_id"],
                            "provider": r["provider"],
                            "answer_model": r["answer_model"],
                            "hypothesis": r["hypothesis"] if r["hypothesis"] is not None else "",
                            "status": r["status"],
                            "termination_reason": r["termination_reason"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    batch_summary_path = batch_dir / "batch_summary.json"
    merged_strategies: dict[str, Any] = {}
    if batch_summary_path.exists():
        existing = json.loads(batch_summary_path.read_text(encoding="utf-8"))
        merged_strategies.update(existing.get("strategies", {}))
    merged_strategies.update(summaries_by_strategy)
    write_json(batch_summary_path, {"batch_id": args.batch_id, "provider": client.provider, "answer_model": args.model, "strategies": merged_strategies})


if __name__ == "__main__":
    main()
