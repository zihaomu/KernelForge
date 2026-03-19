from __future__ import annotations

import unittest

from kernel_autoresearch.python.agent_proposer import choose_candidate, normalize_candidate


class AgentProposerTest(unittest.TestCase):
    def test_normalize_candidate_clamps_and_fallback(self) -> None:
        fallback = {
            "kernel_variant": "blocked",
            "bm": 64,
            "bn": 64,
            "bk": 64,
            "pack_a": False,
            "pack_b": False,
            "simd": True,
            "threads": 4,
            "unroll_k": 2,
        }
        raw = {"kernel_variant": "invalid", "bm": -3, "threads": 0}
        out = normalize_candidate(raw, fallback)
        self.assertEqual(out["kernel_variant"], "blocked")
        self.assertGreaterEqual(out["bm"], 0)
        self.assertGreaterEqual(out["threads"], 1)

    def test_choose_candidate_rules_only_returns_cursor(self) -> None:
        cursor = {
            "kernel_variant": "blocked",
            "bm": 64,
            "bn": 64,
            "bk": 64,
            "pack_a": False,
            "pack_b": False,
            "simd": True,
            "threads": 4,
            "unroll_k": 1,
        }
        res = choose_candidate(
            mode="rules_only",
            agent_cfg={},
            bucket="medium",
            cursor_candidate=cursor,
            pool=[cursor],
            seen_signatures=set(),
            history_tail=[],
            best_candidate=None,
            baseline=None,
            iteration_seed=7,
        )
        self.assertEqual(res["proposal_source"], "rules")
        self.assertEqual(res["candidate"]["kernel_variant"], "blocked")


if __name__ == "__main__":
    unittest.main()

