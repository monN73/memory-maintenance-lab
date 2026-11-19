#!/usr/bin/env python3
"""Verify downloaded LongMemEval datasets, generate fixed subset manifests, and record hashes."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "longmemeval"
MANIFEST_DIR = ROOT / "data" / "manifests"
EVALUATOR_PATH = ROOT / "evaluators" / "evaluate_qa.py"
PILOT_PATH = ROOT / "configs" / "pilot.json"

DATASET_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
EVALUATOR_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_abstention(item: dict) -> bool:
    return "_abs" in item["question_id"]


def summarize(records: list[dict]) -> dict:
    by_type = Counter(r["question_type"] for r in records)
    abs_count = sum(1 for r in records if is_abstention(r))
    return {
        "total": len(records),
        "abstention_count": abs_count,
        "by_question_type": dict(sorted(by_type.items())),
    }


def stratified_sample(records: list[dict], target_n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for r in records:
        buckets[(r["question_type"], is_abstention(r))].append(r)

    total = len(records)
    # Allocate counts proportionally using largest-remainder method
    exact = {k: len(v) * target_n / total for k, v in buckets.items()}
    alloc = {k: int(exact[k]) for k in buckets}
    remainder = target_n - sum(alloc.values())
    for k in sorted(buckets.keys(), key=lambda x: (exact[x] - alloc[x], len(buckets[x])), reverse=True):
        if remainder <= 0:
            break
        if alloc[k] < len(buckets[k]):
            alloc[k] += 1
            remainder -= 1

    chosen: list[dict] = []
    for k in sorted(buckets.keys()):
        pool = sorted(buckets[k], key=lambda x: x["question_id"])
        rng.shuffle(pool)
        chosen.extend(pool[: alloc[k]])

    chosen.sort(key=lambda x: x["question_id"])
    return chosen


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    oracle_path = DATA_DIR / "longmemeval_oracle.json"
    s_path = DATA_DIR / "longmemeval_s_cleaned.json"

    if not oracle_path.exists() or not s_path.exists():
        raise FileNotFoundError("Missing longmemeval_oracle.json or longmemeval_s_cleaned.json")

    oracle_sha = sha256_file(oracle_path)
    s_sha = sha256_file(s_path)
    eval_sha = sha256_file(EVALUATOR_PATH)

    with oracle_path.open("r", encoding="utf-8") as f:
        oracle_data = json.load(f)
    with s_path.open("r", encoding="utf-8") as f:
        s_data = json.load(f)

    # Oracle-5: 1 item from each of 5 distinct question types (deterministic, seed=42)
    rng = random.Random(SEED)
    by_qtype: dict[str, list[dict]] = defaultdict(list)
    for item in oracle_data:
        by_qtype[item["question_type"]].append(item)

    selected_qtypes = [
        "single-session-user",
        "single-session-assistant",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
    ]
    oracle_5: list[dict] = []
    for qt in selected_qtypes:
        pool = sorted(by_qtype[qt], key=lambda x: x["question_id"])
        oracle_5.append(rng.choice(pool))

    oracle_5_ids = [x["question_id"] for x in oracle_5]
    oracle_5_manifest = {
        "manifest_id": "oracle-5",
        "dataset": "LongMemEval",
        "variant": "oracle",
        "source_file": "data/longmemeval/longmemeval_oracle.json",
        "source_sha256": oracle_sha,
        "dataset_revision": DATASET_REVISION,
        "seed": SEED,
        "purpose": "pipeline_smoke_check",
        "count": len(oracle_5),
        "question_ids": oracle_5_ids,
        "summary": summarize(oracle_5),
    }
    (MANIFEST_DIR / "oracle-5.json").write_text(
        json.dumps(oracle_5_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # S-dev-30: stratified by (question_type, is_abstention)
    s_dev_30 = stratified_sample(s_data, 30, SEED)
    s_dev_30_ids = [x["question_id"] for x in s_dev_30]
    oracle_5_set = set(oracle_5_ids)
    overlap_ids = sorted(q for q in s_dev_30_ids if q in oracle_5_set)

    s_dev_30_manifest = {
        "manifest_id": "s-dev-30",
        "dataset": "LongMemEval",
        "release": "cleaned",
        "variant": "S",
        "source_file": "data/longmemeval/longmemeval_s_cleaned.json",
        "source_sha256": s_sha,
        "dataset_revision": DATASET_REVISION,
        "seed": SEED,
        "stratification": ["question_type", "is_abstention"],
        "purpose": "development_screening_not_confirmatory",
        "count": len(s_dev_30),
        "question_ids": s_dev_30_ids,
        "oracle_5_overlap_question_ids": overlap_ids,
        "summary": summarize(s_dev_30),
    }
    (MANIFEST_DIR / "s-dev-30.json").write_text(
        json.dumps(s_dev_30_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Update pilot.json with verified dataset & evaluator metadata
    pilot = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
    pilot["dataset"]["revision"] = DATASET_REVISION
    pilot["dataset"]["sha256"] = s_sha
    pilot["dataset"]["subset_manifest"] = "data/manifests/s-dev-30.json"
    pilot["dataset"]["evaluator_commit"] = EVALUATOR_COMMIT
    PILOT_PATH.write_text(json.dumps(pilot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Oracle summary:", json.dumps(summarize(oracle_data), ensure_ascii=False))
    print("S cleaned summary:", json.dumps(summarize(s_data), ensure_ascii=False))
    print("Oracle-5 summary:", json.dumps(oracle_5_manifest["summary"], ensure_ascii=False))
    print("S-dev-30 summary:", json.dumps(s_dev_30_manifest["summary"], ensure_ascii=False))
    print("Oracle SHA256:", oracle_sha)
    print("S cleaned SHA256:", s_sha)
    print("Evaluator SHA256:", eval_sha)
    print("Overlap(oracle-5, s-dev-30):", overlap_ids)


if __name__ == "__main__":
    main()
