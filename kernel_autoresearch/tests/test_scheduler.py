from __future__ import annotations

import unittest

from kernel_autoresearch.python.scheduler import build_selection


class SchedulerTest(unittest.TestCase):
    def test_select_best_by_balance_score(self) -> None:
        shapes = [{"name": "s1", "m": 512, "n": 512, "k": 512}]
        buckets = {
            "small": {"max_volume": 2000000, "alpha_throughput": 0.4},
            "medium": {"max_volume": 200000000, "alpha_throughput": 0.5},
            "large": {"max_volume": None, "alpha_throughput": 0.6},
        }
        trials = {
            "s1": [
                {
                    "kernel_variant": "naive",
                    "bm": 0,
                    "bn": 0,
                    "bk": 0,
                    "pack_a": False,
                    "pack_b": False,
                    "simd": False,
                    "threads": 1,
                    "unroll_k": 1,
                    "valid": True,
                    "latency_ms_p50": 9.0,
                    "gflops": 30.0,
                    "risk_level": "low",
                },
                {
                    "kernel_variant": "blocked_pack",
                    "bm": 64,
                    "bn": 64,
                    "bk": 64,
                    "pack_a": True,
                    "pack_b": True,
                    "simd": True,
                    "threads": 4,
                    "unroll_k": 2,
                    "valid": True,
                    "latency_ms_p50": 6.0,
                    "gflops": 42.0,
                    "risk_level": "medium",
                },
            ]
        }
        sel = build_selection(
            trials_by_shape=trials,
            shapes=shapes,
            buckets_cfg=buckets,
            risk_penalties={"low": 0.0, "medium": 0.03, "high": 0.08},
            invalid_penalty=-1e6,
        )
        best = sel["best_by_shape"]["s1"]
        self.assertEqual(best["kernel_variant"], "blocked_pack")


if __name__ == "__main__":
    unittest.main()

