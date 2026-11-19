"""Offline snapshot-based probe replay & online/offline request-equivalence verifier.

Strictly enforces snapshot access permissions:
- N0, N1, N2, N3, N4, N5, N6, N9, N10, N14: uses ONLY snapshot memory_text + kept_exchanges (never reads full history).
- N7, N8, N11, N12, N13: restores archived exchanges ONLY from the snapshot's frozen `archived_turn_ids` whitelist
  (N13 restores both structured fact memory_text and its archived_turn_ids whitelist), preventing leakage of evicted turns.
Shares `build_readout_memory` and `build_messages` with `runner.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from budget_executor import (  # noqa: E402
    ARCHIVE_STRATEGIES,
    STRATEGIES,
    BudgetConfig,
    Exchange,
    build_messages,
    build_readout_memory,
)
from dataset_adapters import UnifiedProbe, adapt_longmemeval_item, align_longmemeval_turns  # noqa: E402
from llm_client import DEFAULT_MODELS, LLMClient  # noqa: E402
from metrics import sha256_json  # noqa: E402
from runner import SYSTEM_PROMPT_LONGMEMEVAL, run_isolated_probe  # noqa: E402

NO_ARCHIVE_ACCESS_STRATEGIES = set(STRATEGIES) - set(ARCHIVE_STRATEGIES)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def reconstruct_probe_inputs(
    strategy: str,
    snap: dict[str, Any],
    question: str,
    question_date: str | None,
    budget: BudgetConfig,
    all_exchanges_by_tid: dict[int, Exchange] | None = None,
) -> dict[str, Any]:
    """Reconstruct exact readout memory, retrieved_turn_ids, and normalized messages from a saved snapshot."""
    kept_exchanges = [
        Exchange(
            turn_id=int(k["turn_id"]),
            session_id=str(k["session_id"]),
            user_text=str(k["user_text"]),
            assistant_text=k.get("assistant_text"),
            timestamp=k.get("timestamp"),
        )
        for k in snap.get("kept_exchanges", [])
    ]
    raw_mem = snap.get("memory_text") or ""

    archived_whitelist: list[Exchange] | None = None
    if strategy in ARCHIVE_STRATEGIES:
        if all_exchanges_by_tid is None:
            raise ValueError(f"Strategy {strategy} requires dataset exchanges to resolve archived_turn_ids whitelist.")
        whitelist_ids = [int(tid) for tid in snap["archived_turn_ids"]]
        if any(tid < 1 or tid > int(snap["turn_id"]) or tid not in all_exchanges_by_tid for tid in whitelist_ids):
            raise ValueError("Archive whitelist contains a future or unknown turn.")
        archived_whitelist = [all_exchanges_by_tid[tid] for tid in whitelist_ids]
    elif strategy in NO_ARCHIVE_ACCESS_STRATEGIES:
        # Strictly forbid archive / full-history access for non-archive strategies
        archived_whitelist = None
    else:
        raise ValueError(f"Unknown strategy for snapshot replay: {strategy}")

    injected_mem, retrieved_ids = build_readout_memory(
        strategy=strategy,
        query=question,
        memory_text=raw_mem,
        archived_exchanges=archived_whitelist,
        max_tokens=budget.memory_tokens,
    )
    probe_q = f"[Current Date: {question_date}]\n{question}" if question_date else question
    msgs = build_messages(SYSTEM_PROMPT_LONGMEMEVAL, injected_mem, kept_exchanges, probe_q)
    return {
        "kept_exchanges": kept_exchanges,
        "memory_text_injected": injected_mem,
        "retrieved_turn_ids": retrieved_ids,
        "archived_turn_ids": [ex.turn_id for ex in archived_whitelist] if archived_whitelist is not None else [],
        "messages": msgs,
        "request_sha256": sha256_json(msgs),
    }


def verify_run_against_online(
    run_dir: Path,
    dataset_by_qid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Verify that offline snapshot reconstruction matches online probes.jsonl bit-for-bit on request_sha256."""
    manifest = _load_json(run_dir / "manifest.json")
    strategy = manifest["run_id"].split("-direct-")[1].rsplit("-r", 1)[0]
    qid = manifest["dataset"]["question_id"]
    budget = BudgetConfig(**manifest["effective_budget"])
    raw_item = dataset_by_qid.get(qid)
    all_ex_by_tid = None
    q_date = None
    if raw_item is not None:
        turns = align_longmemeval_turns(raw_item)
        all_ex_by_tid = {
            t.turn_id: Exchange(t.turn_id, t.session_id, t.user_text, t.assistant_text, t.timestamp, t.source_message_indices)
            for t in turns
        }
        q_date = raw_item.get("question_date")

    online_probes = _load_jsonl(run_dir / "probes.jsonl")
    checked = 0
    mismatches = []
    for p in online_probes:
        tid = int(p["turn_id"])
        snap_path = run_dir / "snapshots" / f"turn_{tid:03d}.json"
        if not snap_path.exists():
            mismatches.append({"turn_id": tid, "probe_id": p["probe_id"], "reason": "missing_snapshot"})
            continue
        snap = _load_json(snap_path)
        rec = reconstruct_probe_inputs(
            strategy=strategy,
            snap=snap,
            question=p["question"],
            question_date=q_date,
            budget=budget,
            all_exchanges_by_tid=all_ex_by_tid,
        )
        checked += 1
        fields = ("request_sha256", "retrieved_turn_ids", "memory_text_injected")
        diffs = {
            field: "missing_online_evidence" if field not in p else "different"
            for field in fields if field not in p or p[field] != rec[field]
        }
        if diffs:
            mismatches.append({"turn_id": tid, "probe_id": p["probe_id"], "diffs": diffs})

    return {
        "run_id": manifest["run_id"],
        "strategy": strategy,
        "question_id": qid,
        "checked": checked,
        "mismatches": mismatches,
        "passed": checked > 0 and not mismatches,
    }


def main():
    parser = argparse.ArgumentParser(description="Offline snapshot probe replay and verification.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--verify-only", action="store_true", help="Zero-API verification of offline reconstruction vs online probes.jsonl")
    parser.add_argument("--turns", default=None, help="Comma-separated turn_ids to probe offline (when not --verify-only)")
    parser.add_argument("--question-ids", default=None, help="Optional comma-separated question_ids filter")
    parser.add_argument("--strategies", default=None, help="Optional comma-separated strategies filter")
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="ecnu")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-judge", action="store_true")
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
        ds_items = json.load(source)
    dataset_by_qid = {x["question_id"]: x for x in ds_items}

    q_filter = {q.strip() for q in args.question_ids.split(",") if q.strip()} if args.question_ids else None
    s_filter = {s.strip() for s in args.strategies.split(",") if s.strip()} if args.strategies else None

    run_dirs = sorted((batch_dir / "runs").iterdir())
    if args.verify_only:
        results = []
        for rdir in run_dirs:
            if not (rdir / "manifest.json").exists():
                continue
            mf = _load_json(rdir / "manifest.json")
            st = mf["run_id"].split("-direct-")[1].rsplit("-r", 1)[0]
            qid = mf["dataset"]["question_id"]
            if q_filter and qid not in q_filter:
                continue
            if s_filter and st not in s_filter:
                continue
            if mf["dataset"].get("file_sha256") != dataset_hash:
                raise SystemExit(f"Dataset SHA-256 mismatch or missing for {mf['run_id']}")
            res = verify_run_against_online(rdir, dataset_by_qid)
            results.append(res)
            status_tag = "PASS" if res["passed"] else "FAIL"
            print(f"[{status_tag}] {res['run_id']}: checked={res['checked']} mismatches={len(res['mismatches'])}")
            if res["mismatches"]:
                print("  ", res["mismatches"][:3])
        all_ok = all(r["passed"] for r in results) and len(results) > 0
        print(f"\nVERIFY SUMMARY: {sum(r['passed'] for r in results)}/{len(results)} passed.")
        if not all_ok:
            raise SystemExit(1)
        return

    if not args.turns:
        raise SystemExit("Provide --turns (e.g. 35,175,245) when executing offline snapshot probes.")
    target_turns = [int(x.strip()) for x in args.turns.split(",") if x.strip()]
    client = LLMClient(provider=args.provider)
    model = args.model or DEFAULT_MODELS[args.provider]

    for rdir in run_dirs:
        if not (rdir / "manifest.json").exists():
            continue
        mf = _load_json(rdir / "manifest.json")
        st = mf["run_id"].split("-direct-")[1].rsplit("-r", 1)[0]
        qid = mf["dataset"]["question_id"]
        if q_filter and qid not in q_filter:
            continue
        if s_filter and st not in s_filter:
            continue
        if mf["dataset"].get("file_sha256") != dataset_hash:
            raise SystemExit(f"Dataset SHA-256 mismatch or missing for {mf['run_id']}")
        raw_item = dataset_by_qid[qid]
        adapted = adapt_longmemeval_item(raw_item)
        all_ex_by_tid = {
            t.turn_id: Exchange(t.turn_id, t.session_id, t.user_text, t.assistant_text, t.timestamp, t.source_message_indices)
            for t in adapted.turns
        }
        budget = BudgetConfig(**mf["effective_budget"])
        out_path = rdir / "offline_probes.jsonl"
        written = 0
        with out_path.open("w", encoding="utf-8") as f:
            for tid in target_turns:
                snap_path = rdir / "snapshots" / f"turn_{tid:03d}.json"
                if not snap_path.exists():
                    continue
                snap = _load_json(snap_path)
                for probe in adapted.probes:
                    rec = reconstruct_probe_inputs(
                        strategy=st,
                        snap=snap,
                        question=probe.question,
                        question_date=adapted.question_date,
                        budget=budget,
                        all_exchanges_by_tid=all_ex_by_tid,
                    )
                    p_rec, _, _ = run_isolated_probe(
                        client=client,
                        model=model,
                        system_prompt=SYSTEM_PROMPT_LONGMEMEVAL,
                        memory_text=rec["memory_text_injected"],
                        kept_exchanges=rec["kept_exchanges"],
                        probe=probe,
                        turn_id=tid,
                        session_id=snap["session_id"],
                        run_id=mf["run_id"],
                        trigger="offline_snapshot_replay",
                        question_date=adapted.question_date,
                        judge_probe=(not args.no_judge and (
                            str(tid) in probe.expected_by_turn or tid == len(adapted.turns)
                        )),
                    )
                    p_rec["origin"] = "offline_snapshot_replay"
                    p_rec["retrieved_turn_ids"] = rec["retrieved_turn_ids"]
                    p_rec["archived_turn_ids"] = rec["archived_turn_ids"]
                    f.write(json.dumps(p_rec, ensure_ascii=False) + "\n")
                    written += 1
        print(f"[DONE] {mf['run_id']}: wrote {written} offline snapshot probes -> {out_path.name}")


if __name__ == "__main__":
    main()
