from __future__ import annotations

import unittest

from kernel_autoresearch.python.candidate_generator import generate_candidates


class CandidateGeneratorTest(unittest.TestCase):
    def test_generate_candidates_contains_naive_and_blocked(self) -> None:
        shapes = [{"name": "s1", "m": 256, "n": 256, "k": 256}]
        buckets = {
            "small": {"max_volume": 2000000, "alpha_throughput": 0.4},
            "medium": {"max_volume": 200000000, "alpha_throughput": 0.5},
            "large": {"max_volume": None, "alpha_throughput": 0.6},
        }
        scores = {"pack": 1.2, "vectorize": 0.9, "tiling": 1.0}
        cand_cfg = {
            "block_m": [16, 32, 64],
            "block_n": [16, 32, 64],
            "block_k": [16, 32, 64],
            "unroll_k": [1, 2],
            "thread_choices": [1, 2, 4],
            "allow_pack": True,
            "allow_simd": True,
        }
        out = generate_candidates(
            shapes=shapes,
            buckets_cfg=buckets,
            pattern_scores=scores,
            candidate_cfg=cand_cfg,
            max_trials_per_bucket=20,
            seed=7,
        )
        cands = out["s1"]
        variants = {x["kernel_variant"] for x in cands}
        self.assertIn("naive", variants)
        self.assertIn("blocked", variants)
        self.assertIn("blocked_pack", variants)
        self.assertLessEqual(len(cands), 20)


if __name__ == "__main__":
    unittest.main()

