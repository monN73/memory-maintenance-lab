import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from budget_executor import BudgetConfig, Exchange, apply_maintenance, load_compaction_prompt_template

class Client:
    provider = 'ecnu'
    def __init__(self, texts):
        self.texts = iter(texts)
        self.calls = []
    def chat(self, **kwargs):
        self.calls.append(kwargs)
        text, finish = next(self.texts)
        return dict(text=text, finish_reason=finish, usage={'prompt_tokens': 10, 'completion_tokens': 10}, latency_ms=1)

class MaintenanceTests(unittest.TestCase):
    def run_strategy(self, strategy, budget, client=None):
        return apply_maintenance(strategy, 3, 'test', 'system', '',
            [Exchange(1, 's', 'a' * 20000, 'answer'), Exchange(2, 's', 'b' * 16000, 'answer')],
            'question', budget, client, 'fake', load_compaction_prompt_template()[0])
    def test_full_history_ignores_small_budget(self):
        result = self.run_strategy('N1', BudgetConfig(input_tokens=8192, n1_context_tokens=20000))
        self.assertEqual(result['status'], 'ok')
        self.assertGreater(result['estimated_tokens'], 8192)
        self.assertEqual(len(result['recent_exchanges']), 2)
    def test_capacity_reserves_answer(self):
        result = self.run_strategy('N1', BudgetConfig(n1_context_tokens=1000))
        self.assertEqual(result['status'], 'overflow')
        self.assertEqual(len(result['recent_exchanges']), 2)
    def test_assistant_reply_counts_toward_capacity(self):
        args = dict(strategy="N1", turn_id=1, run_id="test", system_prompt="system", memory_text="", recent_exchanges=[], current_user_text="small question", budget=BudgetConfig(n1_context_tokens=1024))
        self.assertEqual(apply_maintenance(**args)["status"], "ok")
        result = apply_maintenance(**args, current_assistant_text="x" * 5000)
        self.assertEqual(result["status"], "overflow")
        self.assertEqual(result["messages"][-1]["role"], "assistant")

    def test_retry_uses_draft(self):
        client = Client([('x' * 10000, 'stop'), ('[T1] concise fact', 'stop')])
        result = self.run_strategy('N4', BudgetConfig(), client)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(len(client.calls), 2)
        self.assertIn('<previous_attempt>', client.calls[1]['messages'][0]['content'])
        self.assertEqual(len(result['events'][0]['attempts']), 2)
    def test_no_silent_fallback(self):
        client = Client([('x' * 10000, 'stop')] * 2)
        result = self.run_strategy('N4', BudgetConfig(), client)
        self.assertEqual(result['status'], 'maintenance_budget_error')
        self.assertEqual(result['memory_text'], '')
    def test_truncation_is_failure(self):
        result = self.run_strategy('N4', BudgetConfig(), Client([('short', 'length')] * 2))
        self.assertEqual(result['status'], 'maintenance_truncated_error')

if __name__ == '__main__':
    unittest.main()
