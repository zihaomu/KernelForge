from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kernel_autoresearch.python.local_patterns import extract_local_patterns


class LocalPatternsTest(unittest.TestCase):
    def test_extract_local_patterns_from_agent_gen(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gen = root / "code_base_agent_gen"
            repo_dir = gen / "xnnpack"
            (repo_dir / "code_base/manifests").mkdir(parents=True, exist_ok=True)
            (repo_dir / "knowledge/snippets").mkdir(parents=True, exist_ok=True)
            (repo_dir / "research_packs").mkdir(parents=True, exist_ok=True)
            (repo_dir / "reports").mkdir(parents=True, exist_ok=True)

            manifest = {
                "repo": "xnnpack",
                "op_map": {"gemm": ["gemm"]},
                "backend": {"tags": ["cpu", "simd"]},
            }
            (repo_dir / "code_base/manifests/xnnpack.yaml").write_text(
                "\n".join(
                    [
                        f"repo: {manifest['repo']}",
                        "op_map:",
                        "  gemm:",
                        "    - gemm",
                        "backend:",
                        "  tags:",
                        "    - cpu",
                        "    - simd",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            snippets = [
                {
                    "repo": "xnnpack",
                    "source_path": "src/abc.cc",
                    "symbol": "foo",
                    "op": "gemm",
                    "backend": ["cpu", "simd"],
                    "optimization_pattern": ["pack", "vectorize"],
                    "priority": "high",
                    "risk_note": "低风险",
                }
            ]
            (repo_dir / "knowledge/snippets/xnnpack.jsonl").write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in snippets) + "\n",
                encoding="utf-8",
            )
            (repo_dir / "research_packs/xnnpack.md").write_text(
                "1. foo\n可迁移建议: 使用 pack + tiling 调度\n",
                encoding="utf-8",
            )
            (repo_dir / "reports/xnnpack_triage.md").write_text("风险与不确定项: 待复核", encoding="utf-8")

            summary = {
                "entries": [
                    {
                        "repo": "xnnpack",
                        "manifest": "code_base_agent_gen/xnnpack/code_base/manifests/xnnpack.yaml",
                        "snippets_path": "code_base_agent_gen/xnnpack/knowledge/snippets/xnnpack.jsonl",
                        "research_pack": "code_base_agent_gen/xnnpack/research_packs/xnnpack.md",
                        "triage": "code_base_agent_gen/xnnpack/reports/xnnpack_triage.md",
                    }
                ]
            }
            (gen / "run_summary_kernel_triage.json").write_text(
                json.dumps(summary, ensure_ascii=False),
                encoding="utf-8",
            )
            out = root / "local_patterns.json"
            result = extract_local_patterns(root, gen, out)
            self.assertGreaterEqual(result["stats"]["record_count"], 1)
            self.assertIn("pack", result["stats"]["pattern_histogram"])


if __name__ == "__main__":
    unittest.main()
