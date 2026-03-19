from __future__ import annotations

import unittest

from kernel_autoresearch.python.decision_policy import make_decision


class DecisionPolicyTest(unittest.TestCase):
    def test_correctness_fail_must_revert(self) -> None:
        decision = make_decision(
            best_score=1.0,
            current_score=2.0,
            correctness_pass=False,
            min_improve_ratio=0.01,
        )
        self.assertEqual(decision["decision"], "revert")
        self.assertIn("correctness", decision["reason"])
        self.assertEqual(decision["new_best_score"], 1.0)

    def test_improve_enough_keeps_candidate(self) -> None:
        decision = make_decision(
            best_score=1.0,
            current_score=1.05,
            correctness_pass=True,
            min_improve_ratio=0.01,
        )
        self.assertEqual(decision["decision"], "keep")
        self.assertEqual(decision["new_best_score"], 1.05)

    def test_small_gain_reverts(self) -> None:
        decision = make_decision(
            best_score=1.0,
            current_score=1.005,
            correctness_pass=True,
            min_improve_ratio=0.01,
        )
        self.assertEqual(decision["decision"], "revert")
        self.assertEqual(decision["new_best_score"], 1.0)


if __name__ == "__main__":
    unittest.main()

