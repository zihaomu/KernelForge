from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kernel_autoresearch.python.harness_guard import build_manifest, verify_manifest


class HarnessGuardTest(unittest.TestCase):
    def test_verify_manifest_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            harness = root / "harness"
            harness.mkdir(parents=True, exist_ok=True)
            f1 = harness / "bench.py"
            f2 = harness / "reference.py"
            manifest = harness / "manifest.json"
            f1.write_text("print('bench')\n", encoding="utf-8")
            f2.write_text("print('ref')\n", encoding="utf-8")

            build_manifest(
                repo_root=root,
                harness_files=[f1, f2],
                manifest_path=manifest,
            )
            ok, mismatches = verify_manifest(repo_root=root, manifest_path=manifest)
            self.assertTrue(ok)
            self.assertEqual(mismatches, [])

            f2.write_text("print('ref2')\n", encoding="utf-8")
            ok2, mismatches2 = verify_manifest(repo_root=root, manifest_path=manifest)
            self.assertFalse(ok2)
            self.assertGreaterEqual(len(mismatches2), 1)


if __name__ == "__main__":
    unittest.main()

