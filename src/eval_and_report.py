"""Evaluate LongMemEval predictions and generate CSV + offline HTML diagnostic report."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import html
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from report_view import DETERMINISTIC_AUDIT_STRATEGIES, read_jsonl, build_html_report, enrich_strategy, source_turns
from artifact_paths import model_directory
from llm_client import DEFAULT_MODELS, LLMClient

ROOT = Path(__file__).resolve().parent.parent


def get_anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    """Exact prompt templates from evaluators/evaluate_qa.py (commit 9e0b455f4ef0e2ab8f2e582289761153549043fc)."""
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = (
                "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
                "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the "
                "correct answer or contains all the intermediate steps to get the correct answer, you should also "
                "answer yes. If the response only contains a subset of the information required by the answer, "
                "answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
                "correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        elif task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
                "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the "
                "correct answer or contains all the intermediate steps to get the correct answer, you should also "
                "answer yes. If the response only contains a subset of the information required by the answer, "
                "answer no. In addition, do not penalize off-by-one errors for the number of days. If the question "
                "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
                "predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\n"
                "Correct Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        elif task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
                "response contains the correct answer. Otherwise, answer no. If the response contains some previous "
                "information along with an updated answer, the response should be considered as correct as long as "
                "the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel "
                "Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        elif task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model "
                "does not need to reflect all the points in the rubric. The response is correct as long as it recalls "
                "and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel "
                "Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        else:
            raise NotImplementedError(f"Unsupported task: {task}")
    else:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes "
            "if the model correctly identifies the question as unanswerable. The model could say that the information "
            "is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\n"
            "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as "
            "unanswerable? Answer yes or no only."
        )
        return template.format(question, answer, response)


def paired_bootstrap_ci(diffs: list[float], n_boot: int = 2000, seed: int = 42) -> list[float] | None:
    if not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return [round(lo, 4), round(hi, 4)]


def percentile(values: list[float], q: float) -> float | None:
    """Compute linear-interpolated percentile (q in [0, 1]) for a non-empty list."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return round(float(s[0]), 1)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return round(float(s[lo] + (s[hi] - s[lo]) * frac), 1)


def judge_one(
    client: LLMClient,
    judge_model: str,
    ref: dict[str, Any],
    pred: dict[str, Any],
) -> dict[str, Any]:
    qid = pred["question_id"]
    if pred["status"] != "completed" or not pred.get("hypothesis"):
        return {
            "question_id": qid,
            "judge_status": "skipped_no_prediction",
            "judge_label": None,
            "judge_raw": None,
        }
    prompt = get_anscheck_prompt(
        ref["question_type"],
        ref["question"],
        ref["answer"],
        pred["hypothesis"],
        abstention="_abs" in qid,
    )
    try:
        res = client.chat(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            seed=42,
            thinking="disabled" if client.provider == "deepseek" else None,
        )
        raw = res["text"]
        if not raw.strip():
            return {
                "question_id": qid,
                "judge_status": f"judge_error_empty_output (finish={res.get('finish_reason')})",
                "judge_label": None,
                "judge_raw": raw,
                "judge_usage": res["usage"],
            }
        label = "yes" in raw.lower()
        return {
            "question_id": qid,
            "judge_status": "ok",
            "judge_label": label,
            "judge_raw": raw,
            "judge_usage": res["usage"],
        }
    except Exception as e:
        return {
            "question_id": qid,
            "judge_status": f"judge_error: {e}",
            "judge_label": None,
            "judge_raw": None,
        }



def _parse_json_obj(raw: str) -> dict[str, Any]:
    txt = (raw or "").strip()
    if txt.startswith("```"):
        lines = txt.splitlines()
        if len(lines) >= 2:
            txt = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]).strip()
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        txt = txt[start : end + 1]
    return json.loads(txt)


def audit_n4_question(
    client: LLMClient,
    judge_model: str,
    qid: str,
    run_dir: Path,
    run_status: str,
    ref: dict[str, Any],
    src_turns: list[dict[str, Any]],
    cached_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run (or reuse cached) semantic maintenance-quality audit for an N4 run."""
    answer_sids = set(ref.get("answer_session_ids") or [])
    qtype = ref.get("question_type", "general")
    replay_turns = [t for t in src_turns if t.get("kind") == "replay"]
    turn_by_id = {int(t["turn_id"]): t for t in replay_turns}
    all_replay_tids = set(turn_by_id)
    evidence_tids = sorted(int(t["turn_id"]) for t in replay_turns if t.get("session_id") in answer_sids)
    evidence_set = set(evidence_tids)

    events_path = run_dir / "events.jsonl"
    events = (
        [json.loads(l) for l in events_path.read_text(encoding="utf-8").split("\n") if l.strip()]
        if events_path.exists()
        else []
    )

    out_rows: list[dict[str, Any]] = []
    cum_evicted: set[int] = set()
    prev_retained_ev_tids: set[int] = set()
    seen_lost_ev_tids: set[int] = set()
    last_memory = ""

    def format_turns(tids: list[int]) -> str:
        parts = []
        for tid in tids:
            t = turn_by_id.get(tid)
            if not t:
                continue
            parts.append(f"[T{tid} ({t.get('timestamp', '')})] User: {t.get('input', '')}")
            if t.get("output") is not None:
                parts.append(f"[T{tid}] Assistant: {t.get('output')}")
        return "\n".join(parts)

    for ev in events:
        if ev.get("mechanism") != "llm_compaction" or ev.get("status") != "ok":
            continue
        tid_now = int(ev.get("turn_id", 0))
        for etid in ev.get("evicted_turn_ids") or []:
            cum_evicted.add(int(etid))
        ingested_replay = {t for t in all_replay_tids if t < tid_now}
        kept_tids = ingested_replay - cum_evicted
        last_memory = ev.get("after_text") or ""

        seen_ev_tids = [etid for etid in evidence_tids if etid < tid_now]
        ev_id = ev.get("event_id", f"t{tid_now}")
        key = (qid, ev_id)
        if key in cached_by_key:
            row = cached_by_key[key]
            out_rows.append(row)
            ret_ids = set(row.get("retained_evidence_turn_ids") or [])
            prev_retained_ev_tids = ret_ids
            continue

        if not seen_ev_tids:
            row = {
                "question_id": qid,
                "event_id": ev_id,
                "turn_id": tid_now,
                "evidence_retention": None,
                "evidence_turns_retained": None,
                "evidence_turns_total": None,
                "retained_evidence_turn_ids": [],
                "fact_omission_count": 0,
                "false_addition_count": 0,
                "stale_fact_residue_count": 0,
                "skipped_no_evidence_yet": True,
            }
            out_rows.append(row)
            continue

        verbatim_kept = [etid for etid in seen_ev_tids if etid in kept_tids]
        evicted_ev = [etid for etid in seen_ev_tids if etid not in kept_tids]

        evicted_ev_text = format_turns(evicted_ev) if evicted_ev else "(无；所有已出现的证据仍在近期原文窗口中)"
        prompt = (
            "你是一个严谨的记忆维护审计裁判。请对比本次 LLM 压缩的输入与输出，判断记忆质量。\n"
            f"题型：{qtype}\n"
            f"需要检查是否保留的已移出证据轮次 ID：{evicted_ev}\n"
            f"<evicted_evidence_turns>\n{evicted_ev_text}\n</evicted_evidence_turns>\n"
            f"<before_text>\n{ev.get('before_text', '')}\n</before_text>\n"
            f"<after_text>\n{last_memory}\n</after_text>\n\n"
            "请严格输出一个 JSON 对象（不要包含其他解释）：\n"
            "{\n"
            '  "retained_evicted_turn_ids": [integer, ...], // 上述 evicted_evidence_turns 中，其核心事实在 <after_text> 中仍可被支持的轮次 ID（合理改写也算保留）\n'
            '  "false_addition_count": integer, // <after_text> 中凭空新增的、在 <before_text> 中没有依据的断言条数\n'
            '  "stale_fact_residue_count": integer // 仅当题型为 knowledge-update 且事实已被更新时，仍把旧值当作当前有效状态保留的条目数；否则填 0\n'
            "}"
        )
        retained_evicted: set[int] = set()
        false_add = 0
        stale_cnt = 0
        judge_err = None
        try:
            res = client.chat(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                seed=42,
                thinking="disabled" if client.provider == "deepseek" else None,
            )
            parsed = _parse_json_obj(res["text"])
            retained_evicted = {int(x) for x in parsed.get("retained_evicted_turn_ids", []) if int(x) in evicted_ev}
            false_add = max(0, int(parsed.get("false_addition_count", 0)))
            stale_cnt = max(0, int(parsed.get("stale_fact_residue_count", 0)))
        except Exception as exc:
            judge_err = str(exc)

        retained_now = set(verbatim_kept) | retained_evicted
        # First-loss counting: count evidence turns that were previously retained (or newly introduced) and are now missing, once.
        newly_lost = ((set(seen_ev_tids) & prev_retained_ev_tids) - retained_now) - seen_lost_ev_tids
        # Also catch evidence evicted for the very first time in this event
        newly_introduced_and_lost = (set(ev.get("evicted_turn_ids") or []) & set(seen_ev_tids) - retained_now) - seen_lost_ev_tids
        lost_here = newly_lost | newly_introduced_and_lost
        seen_lost_ev_tids.update(lost_here)
        prev_retained_ev_tids = retained_now

        row = {
            "question_id": qid,
            "event_id": ev_id,
            "turn_id": tid_now,
            "evidence_retention": round(len(retained_now) / len(seen_ev_tids), 4),
            "evidence_turns_retained": len(retained_now),
            "evidence_turns_total": len(seen_ev_tids),
            "retained_evidence_turn_ids": sorted(retained_now),
            "fact_omission_count": len(lost_here),
            "false_addition_count": false_add,
            "stale_fact_residue_count": stale_cnt,
            "audit_error": judge_err,
        }
        out_rows.append(row)

    # Common Final Readout checkpoint (only if completed and has evidence)
    if run_status == "completed" and evidence_tids:
        final_key = (qid, "final")
        if final_key in cached_by_key:
            out_rows.append(cached_by_key[final_key])
        else:
            final_kept_tids = all_replay_tids - cum_evicted
            verbatim_kept = [etid for etid in evidence_tids if etid in final_kept_tids]
            evicted_ev = [etid for etid in evidence_tids if etid not in final_kept_tids]
            retained_evicted: set[int] = set()
            judge_err = None
            if evicted_ev:
                evicted_ev_text = format_turns(evicted_ev)
                prompt = (
                    "你是一个严谨的记忆审计裁判。请判断在最终回答前的显式记忆中，是否仍保留了以下证据轮次的核心事实。\n"
                    f"需要检查的证据轮次 ID：{evicted_ev}\n"
                    f"<evidence_turns>\n{evicted_ev_text}\n</evidence_turns>\n"
                    f"<final_memory>\n{last_memory}\n</final_memory>\n\n"
                    "请严格输出一个 JSON 对象（不要包含其他解释）：\n"
                    '{"retained_evicted_turn_ids": [integer, ...]}'
                )
                try:
                    res = client.chat(
                        model=judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=512,
                        seed=42,
                        thinking="disabled" if client.provider == "deepseek" else None,
                    )
                    parsed = _parse_json_obj(res["text"])
                    retained_evicted = {int(x) for x in parsed.get("retained_evicted_turn_ids", []) if int(x) in evicted_ev}
                except Exception as exc:
                    judge_err = str(exc)
            final_retained = set(verbatim_kept) | retained_evicted
            out_rows.append({
                "question_id": qid,
                "checkpoint": "final",
                "evidence_retention": round(len(final_retained) / len(evidence_tids), 4),
                "all_evidence_retained": len(final_retained) == len(evidence_tids),
                "retained_evidence_turn_ids": sorted(final_retained),
                "evidence_turns_total": len(evidence_tids),
                "audit_error": judge_err,
            })
    return out_rows


def _partition_cached_evals(
    existing_evals: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    judge_provider: str,
    judge_model: str,
    endpoint_hash: str | None,
    offline: bool,
    retry_judge_errors: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Return reusable cached judgments and predictions that still require scoring."""
    pred_by_q = {p["question_id"]: p for p in preds}
    existing_by_q = {e["question_id"]: e for e in existing_evals}

    def _cache_matches(e: dict[str, Any]) -> bool:
        qid = e["question_id"]
        return (
            qid in pred_by_q
            and e.get("judge_provider") == judge_provider
            and e.get("judge_model", judge_model) == judge_model
            and (offline or e.get("judge_endpoint_hash") == endpoint_hash)
            and e.get("hypothesis") == pred_by_q[qid].get("hypothesis")
            and e.get("status") == pred_by_q[qid].get("status")
        )

    if retry_judge_errors:
        reusable: dict[str, dict[str, Any]] = {}
        for qid, e in existing_by_q.items():
            jst = e.get("judge_status")
            is_terminal_skip = jst == "skipped_no_prediction" and (
                e.get("status") != "completed" or not (e.get("hypothesis") or "").strip()
            )
            if _cache_matches(e) and (jst == "ok" or is_terminal_skip):
                reusable[qid] = e
        return reusable, [p for p in preds if p["question_id"] not in reusable]

    if (
        len(existing_evals) == len(preds)
        and set(existing_by_q) == set(pred_by_q)
        and all(_cache_matches(e) for e in existing_evals)
    ):
        return existing_by_q, []
    return {}, list(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions and generate report")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="Rebuild report from saved evaluations only; never calls an API")
    mode.add_argument("--unscored", action="store_true", help="Show run costs and traces without generating or using judgments")
    parser.add_argument("--batch-id", default=None, help="Batch directory name or prefix; defaults to the latest batch")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report-id", default=None, help="Optional independent report folder name; defaults to the experiment folder")
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="deepseek", help="Provider used for judging")
    parser.add_argument("--generation-provider", choices=sorted(DEFAULT_MODELS), default=None, help="Select the generation batch when using a logical batch ID")
    parser.add_argument("--generation-model", default=None, help="Select the model that produced the answers")
    parser.add_argument("--judge-model", default=None, help="Judge model ID; defaults to the selected provider model")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--audit-n4", action="store_true", help="Run LLM semantic maintenance-quality audit for N4 (cached in audit_N4.jsonl)")
    parser.add_argument(
        "--retry-judge-errors",
        action="store_true",
        help="Reuse valid cached judgments and re-score only rows with judge errors",
    )
    args = parser.parse_args()
    if args.unscored:
        args.offline = True
    if args.retry_judge_errors and args.offline:
        raise SystemExit("--retry-judge-errors cannot be combined with --offline or --unscored.")

    args.judge_model = args.judge_model or DEFAULT_MODELS[args.provider]
    runs_root = ROOT / "runs"
    model_roots = ([runs_root / model_directory(args.generation_model)] if args.generation_model
                   else [p for p in runs_root.iterdir() if p.is_dir()])
    if args.batch_id is not None and (Path(args.batch_id).name != args.batch_id or args.batch_id in (".", "..")):
        raise SystemExit("--batch-id must be a single directory name.")
    matches = []
    for model_root in model_roots:
        if not model_root.is_dir():
            continue
        for candidate in model_root.iterdir():
            if not candidate.is_dir():
                continue
            if args.batch_id is not None and candidate.name != args.batch_id and not candidate.name.startswith(args.batch_id + "--"):
                continue
            summary_path = candidate / "batch_summary.json"
            if not summary_path.exists():
                continue
            recorded = json.loads(summary_path.read_text(encoding="utf-8"))
            if "strategies" not in recorded:
                continue
            if args.generation_provider and recorded.get("provider") != args.generation_provider:
                continue
            matches.append(candidate)
    if args.batch_id is not None and not args.generation_model:
        legacy = runs_root / args.batch_id
        if (legacy / "batch_summary.json").exists():
            matches.append(legacy)
    matches = list(dict.fromkeys(matches))
    if args.batch_id is None:
        if not matches:
            raise SystemExit("No matching batch found; specify --batch-id.")
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        batch_dir = matches[0]
        args.batch_id = batch_dir.name
        print(f"Auto-selected latest batch: {batch_dir.name}", flush=True)
    else:
        if len(matches) != 1:
            raise SystemExit("Batch is missing or ambiguous; specify the full batch directory name and generation model/provider.")
        batch_dir = matches[0]
    batch_summary = json.loads((batch_dir / "batch_summary.json").read_text(encoding="utf-8"))
    if args.generation_provider and batch_summary.get("provider") != args.generation_provider:
        raise SystemExit("Generation provider does not match recorded batch metadata.")
    if args.generation_model and batch_summary.get("answer_model") != args.generation_model:
        raise SystemExit("Generation model does not match recorded batch metadata.")
    recorded_model = batch_summary.get("answer_model")
    report_model_dir = model_directory(recorded_model) if recorded_model else "unknown"
    report_name = args.report_id or (batch_dir.name + "--unscored" if args.unscored else batch_dir.name)
    if Path(report_name).name != report_name or report_name in (".", ".."):
        raise SystemExit("--report-id must be a single directory name.")
    report_dir = ROOT / "reports" / report_model_dir / report_name
    prior_summary = report_dir / "summary.json"
    if prior_summary.exists():
        prior = json.loads(prior_summary.read_text(encoding="utf-8"))
        if (prior.get("source_batch", batch_dir.name) != batch_dir.name
                or (args.unscored and prior.get("scoring_mode") != "unscored")
                or (not args.unscored and prior.get("scoring_mode") != "unscored"
                    and (prior.get("judge_provider") != args.provider
                                           or prior.get("judge_model") != args.judge_model))):
            raise SystemExit("Report belongs to another batch/judge; choose a new --report-id.")
    report_dir.mkdir(parents=True, exist_ok=True)

    ds_path = Path(args.dataset)
    if not ds_path.is_absolute():
        ds_path = ROOT / ds_path
    raw_ds_text = ds_path.read_text(encoding="utf-8").strip()
    if ds_path.suffix == ".jsonl":
        raw_ds_items = [json.loads(line) for line in raw_ds_text.splitlines() if line.strip()]
    else:
        raw_ds_items = json.loads(raw_ds_text)
    refs = {x.get("question_id") or x["item_id"]: x for x in raw_ds_items}

    strategies_map: dict[str, list[dict[str, Any]]] = batch_summary["strategies"]

    client = None if args.offline else LLMClient(provider=args.provider)
    all_rows: list[dict[str, Any]] = []
    overview: list[dict[str, Any]] = []
    timelines: dict[str, dict[str, Any]] = {}
    dataset_meta: dict[str, Any] = {}

    for st, runs_list in strategies_map.items():
        gen_provider = batch_summary.get("provider") or "unknown"
        candidates = [batch_dir / f"predictions_{st}.jsonl", batch_dir / f"predictions_{st}_{gen_provider}.jsonl"]
        pred_file = next((path for path in candidates if path.exists()), candidates[0])
        with pred_file.open("r", encoding="utf-8") as f:
            preds = [json.loads(line) for line in f if line.strip()]
        # Older predictions may not carry a provider. Only use recorded run metadata;
        # an absent provider remains unknown rather than being inferred from model names.
        n4_audit_by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if st == "N4":
            audit_path = report_dir / "audit_N4.jsonl"
            cached_by_key: dict[tuple[str, str], dict[str, Any]] = {}
            if audit_path.exists():
                for line in audit_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    ar = json.loads(line)
                    k = (ar["question_id"], ar.get("event_id") or ar.get("checkpoint", ""))
                    cached_by_key[k] = ar
            if args.audit_n4 and not args.offline:
                print(f"Running N4 semantic maintenance audit ({len(runs_list)} questions, workers={args.workers})...", flush=True)
                all_audit_rows: list[dict[str, Any]] = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = {
                        ex.submit(
                            audit_n4_question,
                            client,
                            args.judge_model,
                            r["question_id"],
                            batch_dir / "runs" / r["run_id"],
                            r["status"],
                            refs[r["question_id"]],
                            source_turns(refs[r["question_id"]]),
                            cached_by_key,
                        ): r["question_id"]
                        for r in runs_list
                    }
                    done_a = 0
                    for fut in concurrent.futures.as_completed(futs):
                        aqid = futs[fut]
                        q_rows = fut.result()
                        n4_audit_by_q[aqid] = q_rows
                        all_audit_rows.extend(q_rows)
                        done_a += 1
                        if done_a % 5 == 0 or done_a == len(runs_list):
                            print(f"N4 audit progress: {done_a}/{len(runs_list)}", flush=True)
                with audit_path.open("w", encoding="utf-8") as af:
                    for ar in all_audit_rows:
                        af.write(json.dumps(ar, ensure_ascii=False) + "\n")
            else:
                for (aqid, _), ar in cached_by_key.items():
                    n4_audit_by_q[aqid].append(ar)

        for r in runs_list:
            run_manifest = batch_dir / "runs" / r["run_id"] / "manifest.json"
            metadata = json.loads(run_manifest.read_text(encoding="utf-8")) if run_manifest.exists() else {}
            r["generation_provider"] = r.get("provider") or metadata.get("provider")
            r["generation_model"] = r.get("answer_model") or metadata.get("answer_model")
        pred_by_q = {p["question_id"]: p for p in preds}
        n4_audit_by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if st == "N4":
            audit_path = report_dir / "audit_N4.jsonl"
            cached_by_key: dict[tuple[str, str], dict[str, Any]] = {}
            if audit_path.exists():
                for line in audit_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    ar = json.loads(line)
                    k = (ar["question_id"], ar.get("event_id") or ar.get("checkpoint", ""))
                    cached_by_key[k] = ar
            if args.audit_n4 and not args.offline:
                print(f"Running N4 semantic maintenance audit ({len(runs_list)} questions, workers={args.workers})...", flush=True)
                all_audit_rows: list[dict[str, Any]] = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = {
                        ex.submit(
                            audit_n4_question,
                            client,
                            args.judge_model,
                            r["question_id"],
                            batch_dir / "runs" / r["run_id"],
                            r["status"],
                            refs[r["question_id"]],
                            source_turns(refs[r["question_id"]]),
                            cached_by_key,
                        ): r["question_id"]
                        for r in runs_list
                    }
                    done_a = 0
                    for fut in concurrent.futures.as_completed(futs):
                        aqid = futs[fut]
                        q_rows = fut.result()
                        n4_audit_by_q[aqid] = q_rows
                        all_audit_rows.extend(q_rows)
                        done_a += 1
                        if done_a % 5 == 0 or done_a == len(runs_list):
                            print(f"N4 audit progress: {done_a}/{len(runs_list)}", flush=True)
                with audit_path.open("w", encoding="utf-8") as af:
                    for ar in all_audit_rows:
                        af.write(json.dumps(ar, ensure_ascii=False) + "\n")
            else:
                for (aqid, _), ar in cached_by_key.items():
                    n4_audit_by_q[aqid].append(ar)

        for r in runs_list:
            if r["question_id"] in pred_by_q:
                pred_by_q[r["question_id"]].setdefault("provider", r["generation_provider"])
                pred_by_q[r["question_id"]].setdefault("answer_model", r["generation_model"])

        eval_out = report_dir / f"eval_{st}.jsonl"
        judge_results: dict[str, dict[str, Any]] = {}
        pending_preds: list[dict[str, Any]] = list(preds)
        if args.unscored:
            judge_results = {
                p["question_id"]: {"judge_status": "not_scored", "judge_label": None}
                for p in preds
            }
            pending_preds = []
        elif eval_out.exists():
            with eval_out.open("r", encoding="utf-8") as f:
                existing_evals = [json.loads(line) for line in f if line.strip()]
            judge_results, pending_preds = _partition_cached_evals(
                existing_evals,
                preds,
                args.provider,
                args.judge_model,
                client.endpoint_hash if client else None,
                args.offline,
                args.retry_judge_errors,
            )

        if pending_preds and args.offline:
            raise SystemExit(f"Offline rebuild needs matching saved evaluations: {eval_out}")
        if pending_preds:
            print(f"[{st}] Judging {len(pending_preds)}/{len(preds)} predictions...", flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [
                    ex.submit(judge_one, client, args.judge_model, refs[p["question_id"]], p)
                    for p in pending_preds
                ]
                for fut in concurrent.futures.as_completed(futs):
                    jr = fut.result()
                    judge_results[jr["question_id"]] = jr

            # Write eval results JSONL
            with eval_out.open("w", encoding="utf-8") as f:
                for p in preds:
                    jr = judge_results[p["question_id"]] | {
                        "judge_provider": args.provider,
                        "judge_model": args.judge_model,
                        "judge_endpoint_hash": client.endpoint_hash,
                    }
                    f.write(json.dumps(p | jr, ensure_ascii=False) + "\n")

        planned = len(runs_list)
        predicted = sum(1 for r in runs_list if r["status"] == "completed")
        judged = sum(1 for jr in judge_results.values() if jr["judge_status"] == "ok")
        correct = sum(1 for jr in judge_results.values() if jr["judge_label"] is True)
        overflow = sum(1 for r in runs_list if r["status"] == "overflow")
        errors = sum(1 for r in runs_list if r["status"] not in ("completed", "overflow"))
        total_maint = sum(r["maintenance_events"] for r in runs_list)
        total_tok = sum(r["tokens"]["total_in"] + r["tokens"]["total_out"] for r in runs_list)
        ans_lats = [
            float(r["latency_ms"]["answer"])
            for r in runs_list
            if r["status"] == "completed" and (r.get("latency_ms") or {}).get("answer") is not None
        ]
        search_lats = [
            float((r.get("latency_ms") or {}).get("search", 0) or 0)
            for r in runs_list
            if r["status"] == "completed" and r.get("latency_ms") is not None
        ]
        maint_lats = [
            float((r.get("latency_ms") or {}).get("maintenance", 0) or 0)
            for r in runs_list
            if r.get("latency_ms") is not None
        ]
        readout_lats = [
            float((r.get("latency_ms") or {}).get("search", 0) or 0)
            + float((r.get("latency_ms") or {}).get("answer", 0) or 0)
            for r in runs_list
            if r["status"] == "completed" and r.get("latency_ms") is not None
        ]
        tot_lats = [
            float((r.get("latency_ms") or {}).get("maintenance", 0) or 0)
            + float((r.get("latency_ms") or {}).get("search", 0) or 0)
            + float((r.get("latency_ms") or {}).get("answer", 0) or 0)
            for r in runs_list
            if r["status"] == "completed" and r.get("latency_ms") is not None
        ]

        overview_entry = {
            "strategy": st,
            "planned": planned,
            "predicted": predicted,
            "judged": judged,
            "correct": correct,
            "end_to_end_success": None if args.unscored else (round(correct / planned, 4) if planned else 0.0),
            "conditional_accuracy": round(correct / judged, 4) if judged else None,
            "failure_types": {status: sum(1 for r in runs_list if r["status"] == status) for status in sorted({r["status"] for r in runs_list if r["status"] != "completed"})},
            "scoring_pending": sum(1 for r in runs_list if r["status"] == "completed" and (r["question_id"] not in judge_results or judge_results[r["question_id"]].get("judge_status") != "ok")),
            "overflow": overflow,
            "errors": errors,
            "total_maint_events": total_maint,
            "total_tokens": total_tok,
            "maint_latency_p50_ms": percentile(maint_lats, 0.50),
            "maint_latency_p95_ms": percentile(maint_lats, 0.95),
            "search_latency_p50_ms": percentile(search_lats, 0.50),
            "search_latency_p95_ms": percentile(search_lats, 0.95),
            "answer_latency_p50_ms": percentile(ans_lats, 0.50),
            "answer_latency_p95_ms": percentile(ans_lats, 0.95),
            "readout_latency_p50_ms": percentile(readout_lats, 0.50),
            "readout_latency_p95_ms": percentile(readout_lats, 0.95),
            "total_latency_p50_ms": percentile(tot_lats, 0.50),
            "total_latency_p95_ms": percentile(tot_lats, 0.95),
        }
        ev_ret_vals: list[float] = []
        all_ev_retained_count = 0
        st_omissions = 0
        st_false_additions = 0
        st_stale_residues = 0
        outcome_counts: dict[str, int] = defaultdict(int)

        n4_audit_by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if st == "N4":
            audit_path = report_dir / "audit_N4.jsonl"
            cached_by_key: dict[tuple[str, str], dict[str, Any]] = {}
            if audit_path.exists():
                for line in audit_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    ar = json.loads(line)
                    k = (ar["question_id"], ar.get("event_id") or ar.get("checkpoint", ""))
                    cached_by_key[k] = ar
            if args.audit_n4 and not args.offline:
                print(f"Running N4 semantic maintenance audit ({len(runs_list)} questions, workers={args.workers})...", flush=True)
                all_audit_rows: list[dict[str, Any]] = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = {
                        ex.submit(
                            audit_n4_question,
                            client,
                            args.judge_model,
                            r["question_id"],
                            batch_dir / "runs" / r["run_id"],
                            r["status"],
                            refs[r["question_id"]],
                            source_turns(refs[r["question_id"]]),
                            cached_by_key,
                        ): r["question_id"]
                        for r in runs_list
                    }
                    done_a = 0
                    for fut in concurrent.futures.as_completed(futs):
                        aqid = futs[fut]
                        q_rows = fut.result()
                        n4_audit_by_q[aqid] = q_rows
                        all_audit_rows.extend(q_rows)
                        done_a += 1
                        if done_a % 5 == 0 or done_a == len(runs_list):
                            print(f"N4 audit progress: {done_a}/{len(runs_list)}", flush=True)
                with audit_path.open("w", encoding="utf-8") as af:
                    for ar in all_audit_rows:
                        af.write(json.dumps(ar, ensure_ascii=False) + "\n")
            else:
                for (aqid, _), ar in cached_by_key.items():
                    n4_audit_by_q[aqid].append(ar)

        for r in runs_list:
            qid = r["question_id"]
            jr = judge_results[qid]
            run_dir = batch_dir / "runs" / r["run_id"]
            mf = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            if not dataset_meta:
                dataset_meta = mf.get("dataset", {})

            lat = r.get("latency_ms")
            maint_lat_val = lat.get("maintenance", 0) if lat is not None else None
            search_lat_val = lat.get("search", 0) if lat is not None else None
            ans_lat_val = lat.get("answer") if lat is not None else None
            readout_lat_val = (
                (search_lat_val or 0) + ans_lat_val
                if ans_lat_val is not None
                else None
            )
            tot_lat_val = (
                (maint_lat_val or 0) + (search_lat_val or 0) + (ans_lat_val or 0)
                if lat is not None
                else None
            )

            row = {
                "question_id": qid,
                "question_type": r["question_type"],
                "generation_provider": r["generation_provider"],
                "generation_model": r["generation_model"],
                "retry": r.get("retry"),
                "judge_provider": args.provider,
                "judge_model": args.judge_model,
                "is_abstention": r["is_abstention"],
                "strategy": st,
                "status": r["status"],
                "judge_status": jr["judge_status"],
                "judge_label": jr["judge_label"],
                "ingested_tokens_est": r["cumulative_ingested_tokens_est"],
                "final_ctx_tokens_est": r["final_ctx_tokens_est"],
                "maintenance_events": r["maintenance_events"],
                "archive_capacity_tokens_est": (
                    "unbounded" if st == "N7" else
                    mf.get("effective_budget", {}).get("n8_archive_tokens")
                    if st in ("N8", "N11", "N12", "N13") else None
                ),
                "maintenance_llm_logical_calls": (
                    sum(len(event.get("attempts", [])) for event in
                        read_jsonl(run_dir / "events.jsonl", []))
                    if st in ("N4", "N6") and (run_dir / "events.jsonl").exists()
                    else None if st in ("N4", "N6") else 0
                ),
                "total_tokens": r["tokens"]["total_in"] + r["tokens"]["total_out"],
                "maint_latency_ms": maint_lat_val,
                "search_latency_ms": search_lat_val,
                "answer_latency_ms": ans_lat_val,
                "readout_latency_ms": readout_lat_val,
                "total_latency_ms": tot_lat_val,
            }
            all_rows.append(row)

            if qid not in timelines:
                timelines[qid] = {
                    "question_id": qid,
                    "question_type": r["question_type"],
                    "is_abstention": r["is_abstention"],
                    "question": refs[qid]["question"],
                    "answer": refs[qid]["answer"],
                    "source_turns": source_turns(refs[qid]),
                    "strategies": {},
                }
            norm_lat = (
                {"maintenance": maint_lat_val, "search": search_lat_val, "answer": ans_lat_val}
                if lat is not None
                else None
            )
            enriched = enrich_strategy(
                run_dir,
                {
                    "generation_provider": r["generation_provider"],
                    "generation_model": r["generation_model"],
                    "retry": r.get("retry"),
                    "judge_provider": args.provider,
                    "status": r["status"],
                    "hypothesis": r["hypothesis"],
                    "judge_label": jr["judge_label"],
                    "judge_raw": jr.get("judge_raw"),
                    "maintenance_events": r["maintenance_events"],
                    "latency_ms": norm_lat,
                },
                strategy=st,
                source_turns_list=timelines[qid]["source_turns"],
                answer_session_ids=refs[qid].get("answer_session_ids", []),
                question_type=r["question_type"],
                n4_audit_rows=n4_audit_by_q.get(qid) if st == "N4" else None,
            )
            all_ev_ret = enriched.get("all_evidence_retained")
            if r["status"] != "completed":
                outcome_cat = "execution_failure"
            elif jr["judge_label"] is True:
                outcome_cat = "correct"
            elif jr["judge_label"] is False:
                if all_ev_ret is False:
                    outcome_cat = "evidence_lost"
                elif all_ev_ret is True:
                    outcome_cat = "evidence_present_wrong_answer"
                else:
                    outcome_cat = "wrong_no_evidence_eval"
            else:
                outcome_cat = "unjudged"

            enriched["outcome_category"] = outcome_cat
            row["outcome_category"] = outcome_cat
            row["final_evidence_retention"] = enriched.get("final_evidence_retention")
            row["all_evidence_retained"] = all_ev_ret
            outcome_counts[outcome_cat] += 1

            timelines[qid]["strategies"][st] = enriched
            if enriched.get("final_evidence_retention") is not None:
                ev_ret_vals.append(float(enriched["final_evidence_retention"]))
            if all_ev_ret is True:
                all_ev_retained_count += 1
            st_omissions += enriched.get("total_fact_omissions", 0) or 0
            st_false_additions += enriched.get("total_false_additions", 0) or 0
            st_stale_residues += enriched.get("final_stale_residues", 0) or 0

        strategy_rows = [row for row in all_rows if row["strategy"] == st]
        capacities = {row["archive_capacity_tokens_est"] for row in strategy_rows}
        calls = [row["maintenance_llm_logical_calls"] for row in strategy_rows]
        overview_entry["archive_capacity_tokens_est"] = next(iter(capacities)) if len(capacities) == 1 else "mixed"
        overview_entry["maintenance_llm_logical_calls"] = sum(calls) if calls and all(x is not None for x in calls) else None

        if st in DETERMINISTIC_AUDIT_STRATEGIES or (st == "N4" and ev_ret_vals):
            overview_entry["evidence_evaluable_count"] = len(ev_ret_vals)
            overview_entry["avg_evidence_retention"] = round(sum(ev_ret_vals) / len(ev_ret_vals), 4) if ev_ret_vals else None
            overview_entry["all_evidence_retained_count"] = all_ev_retained_count
            overview_entry["total_fact_omissions"] = st_omissions
            overview_entry["total_false_additions"] = st_false_additions
            overview_entry["total_stale_residues"] = st_stale_residues
        else:
            overview_entry["evidence_evaluable_count"] = 0
            overview_entry["avg_evidence_retention"] = None
            overview_entry["all_evidence_retained_count"] = None
            overview_entry["total_fact_omissions"] = None
            overview_entry["total_false_additions"] = None
            overview_entry["total_stale_residues"] = None
        overview_entry["outcome_breakdown"] = dict(sorted(outcome_counts.items()))
        overview.append(overview_entry)

    # Compute paired comparisons on end_to_end_success
    score_by_st_q: dict[str, dict[str, int]] = defaultdict(dict)
    for row in all_rows:
        score_by_st_q[row["strategy"]][row["question_id"]] = 1 if row["judge_label"] is True else 0

    comparisons = []
    for treat, ctrl in [
        ("N14", "N10"), ("N13", "N10"), ("N13", "N7"),
        ("N11", "N8"), ("N12", "N8"), ("N12", "N11"),
        ("N10", "N9"), ("N8", "N7"), ("N7", "N4"), ("N7", "N3"), ("N7", "N2"),
        ("N6", "N4"), ("N4", "N3"), ("N4", "N2"), ("N3", "N2"), ("N2", "N1"), ("N5", "N1"),
    ]:
        if treat in score_by_st_q and ctrl in score_by_st_q:
            common = sorted(set(score_by_st_q[treat]) & set(score_by_st_q[ctrl]))
            diffs = [float(score_by_st_q[treat][q] - score_by_st_q[ctrl][q]) for q in common]
            comparisons.append({
                "name": f"{treat} - {ctrl}",
                "n": len(common),
                "mean_diff": round(sum(diffs) / len(diffs), 4) if diffs else 0.0,
                "ci95": paired_bootstrap_ci(diffs),
            })

    # Write CSV
    csv_path = report_dir / "question-results.csv"
    if all_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    summary_obj = {
        "batch_id": args.batch_id,
        "source_batch": batch_dir.name,
        **({"source_batches": batch_summary["source_batches"]} if "source_batches" in batch_summary else {}),
        "dataset": dataset_meta,
        "generation_providers": sorted({r["generation_provider"] or "unknown" for r in all_rows}),
        "judge_provider": args.provider,
        "judge_endpoint_hash": client.endpoint_hash if client else (prior.get("judge_endpoint_hash") if prior_summary.exists() else None),
        "judge_model": args.judge_model,
        "judge_thinking": ("disabled" if client.provider == "deepseek" else None) if client else (prior.get("judge_thinking") if prior_summary.exists() else None),
        "overview": overview,
        "comparisons": comparisons,
    }
    if args.unscored:
        comparisons.clear()
        summary_obj.update(judge_provider=None, judge_model=None, scoring_mode="unscored")
    elif args.offline and prior_summary.exists():
        summary_obj = prior | summary_obj
    (report_dir / "summary.json").write_text(
        json.dumps(summary_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    html_str = build_html_report(args.batch_id, dataset_meta, overview, comparisons, all_rows, timelines, summary_obj["judge_provider"], summary_obj["judge_model"], detail_dir=report_dir / "details")
    report_path = report_dir / "report.html"
    report_path.write_text(html_str, encoding="utf-8")

    print(json.dumps(summary_obj, ensure_ascii=False, indent=2))
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
