from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kernel_autoresearch.python.progress_report import generate_progress_report


class ProgressReportTest(unittest.TestCase):
    def test_generate_progress_report_outputs_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results = root / "results.tsv"
            run_log = root / "run.log"
            out_dir = root / "progress"

            results.write_text(
                "ts\titeration\tbucket\tcandidate_signature\tcorrectness_pass\tavg_latency_ms\tavg_gflops\tscore\tbest_score_before\tbest_score_after\tdecision\treason\n"
                "2026-03-18T10:00:00\t1\tmedium\ta\t1\t1.0\t2.0\t1.1\t1.0\t1.1\tkeep\tscore_improved\n",
                encoding="utf-8",
            )
            run_log.write_text(
                "[2026-03-18T10:00:00] iter=1 something happened\n",
                encoding="utf-8",
            )
            result = generate_progress_report(results_tsv=results, run_log=run_log, out_dir=out_dir)
            self.assertTrue(Path(result["html"]).exists())
            self.assertTrue(Path(result["svg"]).exists())
            self.assertTrue(Path(result["timeline_tsv"]).exists())


if __name__ == "__main__":
    unittest.main()

