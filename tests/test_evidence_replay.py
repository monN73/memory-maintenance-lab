import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from budget_executor import BudgetConfig, Exchange, bound_archive, fact_memory_compact_with_stats, serialize_exchanges
from export_evidence_trajectory import _trace_unit_loss_interval
from metrics import estimate_text_tokens
from probe_from_snapshots import reconstruct_probe_inputs, verify_run_against_online


class EvidenceReplayTests(unittest.TestCase):
    def test_first_loss_keeps_last_observation_even_after_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "snapshots").mkdir()
            for turn, archived in [(2, [1]), (8, []), (11, [1])]:
                (root / "snapshots" / f"turn_{turn:03}.json").write_text(
                    json.dumps({"archived_turn_ids": archived}))
            result = _trace_unit_loss_interval("N8", root, {"turn_id": 1}, {})
            self.assertEqual(result["observed_loss_interval"], "(T2, T8]")

    def test_window_migration_is_not_archive_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "snapshots").mkdir()
            for turn, archives, recent in [(1, [], [1]), (2, [1], []), (3, [], [])]:
                snapshot = {"archived_turn_ids": archives,
                            "kept_exchanges": [{"turn_id": x} for x in recent]}
                (root / "snapshots" / f"turn_{turn:03d}.json").write_text(json.dumps(snapshot))
            events = [{"turn_id": 2, "event_id": "move", "evicted_turn_ids": [1]},
                      {"turn_id": 3, "event_id": "drop", "archive_evicted_turn_ids": [1]}]
            (root / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
            result = _trace_unit_loss_interval("N8", root, {"turn_id": 1}, {})
            self.assertEqual(result["observed_loss_interval"], "(T2, T3]")
            self.assertEqual(result["attribution"], "attributed:drop")

    def test_hybrid_replay_limits_bm25_to_whitelist_and_uses_fact_table(self):
        exchanges = {i: Exchange(i, "s", text) for i, text in
                     [(1, "forbidden orchid"), (2, "orchid flower"), (3, "future orchid")]}
        snapshot = {"turn_id": 2, "archived_turn_ids": [2], "memory_text": "[T1] user | plant = orchid"}
        readout = reconstruct_probe_inputs("N13", snapshot, "orchid", None, BudgetConfig(), exchanges)
        self.assertEqual(readout["retrieved_turn_ids"], [1, 2])
        self.assertNotIn("forbidden", readout["memory_text_injected"])
        with self.assertRaises(ValueError):
            reconstruct_probe_inputs("N13", snapshot | {"archived_turn_ids": [3]},
                                     "orchid", None, BudgetConfig(), exchanges)

    def test_missing_online_hash_cannot_pass_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "snapshots").mkdir()
            snapshot = {"turn_id": 1, "archived_turn_ids": []}
            (root / "snapshots" / "turn_001.json").write_text(json.dumps(snapshot))
            (root / "manifest.json").write_text(json.dumps({
                "run_id": "lme-smoke-q-direct-N2-r1", "dataset": {"question_id": "q"},
                "effective_budget": {},
            }))
            (root / "probes.jsonl").write_text(json.dumps({
                "turn_id": 1, "probe_id": "p", "question": "q",
                "retrieved_turn_ids": [], "memory_text_injected": "",
            }))
            self.assertFalse(verify_run_against_online(root, {})["passed"])

    def test_enhanced_extractor_replaces_recognized_progress_without_assistant_ops(self):
        text, stats = fact_memory_compact_with_stats(
            "[T1] user | short_stories = 4",
            [Exchange(2, "s", "I've completed 7 short stories.", "I stopped short stories.")],
            2048, allow_update=True, enhanced_extractor=True,
        )
        self.assertIn("short_stories = 7", text)
        self.assertNotIn("short_stories = 4", text)
        self.assertEqual(stats["value_changes"], 1)

    def test_archive_dedup_and_session_balance(self):
        a = Exchange(1, "a", "orchid", timestamp="2025-01-01")
        b = Exchange(2, "b", "orchid", timestamp="2025-01-02")
        kept, dropped = bound_archive([a, b], 2048, "dedup_fifo")
        self.assertEqual((kept, dropped), ([b], [a]))
        turns = [Exchange(i, session, "x" * 200) for i, session in
                 [(1, "short"), (2, "long"), (3, "long")]]
        cap = sum(estimate_text_tokens(serialize_exchanges([x])) for x in turns[:2])
        kept, dropped = bound_archive(turns, cap, "session_balanced")
        self.assertEqual([x.turn_id for x in kept], [1, 3])
        self.assertEqual([x.turn_id for x in dropped], [2])

    def test_apply_maintenance_logs_archive_evictions_when_archived_passed(self):
        from budget_executor import apply_maintenance
        b = BudgetConfig(input_tokens=1024, recent_history_tokens=400, memory_tokens=200, n8_archive_tokens=80)
        archived = [Exchange(1, "s1", "alpha " * 30), Exchange(2, "s1", "beta " * 30)]
        recent = [Exchange(3, "s1", "gamma " * 300), Exchange(4, "s1", "delta " * 300)]
        for strat in ("N8", "N11", "N12", "N13"):
            res = apply_maintenance(
                strategy=strat, turn_id=5, run_id="r", system_prompt="sys", memory_text="",
                recent_exchanges=recent, current_user_text="epsilon " * 40, budget=b, archived_exchanges=archived,
            )
            ev = res["events"][0]
            self.assertTrue(ev.get("archive_evicted_turn_ids"), f"{strat} should log archive_evicted_turn_ids")

    def test_retry_judge_errors_reuses_valid_and_skips_errors(self):
        from eval_and_report import _partition_cached_evals
        preds = [
            {"question_id": "q1", "hypothesis": "yes", "status": "completed"},
            {"question_id": "q2", "hypothesis": "ans", "status": "completed"},
            {"question_id": "q3", "hypothesis": "", "status": "overflow"},
        ]
        existing = [
            {"question_id": "q1", "hypothesis": "yes", "status": "completed", "judge_provider": "ecnu", "judge_model": "ecnu-plus", "judge_endpoint_hash": "h1", "judge_status": "ok", "judge_label": True},
            {"question_id": "q2", "hypothesis": "ans", "status": "completed", "judge_provider": "ecnu", "judge_model": "ecnu-plus", "judge_endpoint_hash": "h1", "judge_status": "judge_error: HTTP 429", "judge_label": None},
            {"question_id": "q3", "hypothesis": "", "status": "overflow", "judge_provider": "ecnu", "judge_model": "ecnu-plus", "judge_endpoint_hash": "h1", "judge_status": "skipped_no_prediction", "judge_label": None},
        ]
        reusable, pending = _partition_cached_evals(existing, preds, "ecnu", "ecnu-plus", "h1", False, True)
        self.assertEqual(sorted(reusable.keys()), ["q1", "q3"])
        self.assertEqual([p["question_id"] for p in pending], ["q2"])

    def test_probe_plan_reuse_verification_and_gap_classification(self):
        from run_probe_plan import _classify_missing_snapshot, _classify_recheck, compute_regression_intervals, verify_saved_record
        plan = {"provider": "ecnu", "model": "ecnu-plus", "judge_provider": "ecnu", "judge_model": "ecnu-plus"}
        good = {
            "source_batch": "b1", "question_id": "q1", "strategy": "N9", "turn_id": 10,
            "request_sha256": "abc", "expected": "42", "provider": "ecnu", "model": "ecnu-plus",
            "judge_provider": "ecnu", "judge_model": "ecnu-plus", "correct": True, "status": "ok",
        }
        self.assertEqual(verify_saved_record(good, source_batch="b1", qid="q1", strategy="N9", turn=10, request_sha256="abc", expected="42", plan=plan)[0], "reusable_ok")
        self.assertEqual(verify_saved_record(good | {"correct": None}, source_batch="b1", qid="q1", strategy="N9", turn=10, request_sha256="abc", expected="42", plan=plan)[0], "mismatch_error")
        self.assertEqual(verify_saved_record(good | {"judge_model": "other"}, source_batch="b1", qid="q1", strategy="N9", turn=10, request_sha256="abc", expected="42", plan=plan)[0], "mismatch_error")
        self.assertEqual(verify_saved_record(good | {"status": "api_error_gap", "correct": None}, source_batch="b1", qid="q1", strategy="N9", turn=10, request_sha256="abc", expected="42", plan=plan)[0], "pending_new")

        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            (rd / "snapshots").mkdir()
            (rd / "snapshots" / "turn_009.json").write_text("{}")
            (rd / "manifest.json").write_text(json.dumps({"status": "overflow", "termination_reason": "overflow"}))
            (rd / "turns.jsonl").write_text(json.dumps({"turn_id": 10}) + "\n")
            self.assertEqual(_classify_missing_snapshot(rd, 10)["gap_type"], "termination_turn_no_snapshot")
            self.assertEqual(_classify_missing_snapshot(rd, 11)["gap_type"], "post_termination")
            self.assertEqual(_classify_missing_snapshot(rd, 8)["gap_type"], "unexpected_missing")

        reg = compute_regression_intervals([
            {"turn_id": 10, "correct": False, "status": "ok"},
            {"turn_id": 20, "correct": True, "status": "ok"},
            {"turn_id": 21, "correct": None, "status": "api_error_gap"},
            {"turn_id": 30, "correct": False, "status": "ok"},
        ])
        self.assertEqual(reg["first_observed_error_turn"], 10)
        self.assertEqual(reg["last_ok_turn_before_regression"], 20)
        self.assertEqual(reg["first_regression_turn"], 30)
        self.assertEqual(reg["regression_interval"], "(T20, T30]")
        self.assertEqual(reg["bridged_api_error_gaps"], [21])

        self.assertEqual(_classify_recheck({"hypothesis": "25", "correct": True}, {"hypothesis": "25", "correct": False}), "scoring_inconsistent")
        self.assertEqual(_classify_recheck({"hypothesis": "25", "correct": True}, {"hypothesis": "27", "correct": False}), "readout_instability")

    def test_skip_maintenance_counterfactual_reconstruction(self):
        from probe_from_snapshots import reconstruct_skip_maintenance_inputs
        prev_snap = {"turn_id": 2, "memory_text": "[T1] user | k = 1", "archived_turn_ids": [], "kept_exchanges": [{"turn_id": 2, "session_id": "s1", "user_text": "hi", "assistant_text": "ok"}]}
        ex3 = Exchange(3, "s1", "update k to 2", "noted")
        rec = reconstruct_skip_maintenance_inputs("N10", prev_snap, ex3, "what is k?", None, BudgetConfig(input_tokens=8192), {2: Exchange(2, "s1", "hi", "ok"), 3: ex3})
        self.assertTrue(rec["fits_budget"])
        self.assertEqual([k.turn_id for k in rec["kept_exchanges"]], [2, 3])
        rec_over = reconstruct_skip_maintenance_inputs("N10", prev_snap, ex3, "what is k?", None, BudgetConfig(input_tokens=20), {2: Exchange(2, "s1", "hi", "ok"), 3: ex3})
        self.assertFalse(rec_over["fits_budget"])


if __name__ == "__main__":
    unittest.main()
