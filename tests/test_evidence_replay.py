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


if __name__ == "__main__":
    unittest.main()
