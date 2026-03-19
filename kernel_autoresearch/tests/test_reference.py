from __future__ import annotations

import unittest

from kernel_autoresearch.harness.reference import deterministic_reference_checksum


class ReferenceChecksumTest(unittest.TestCase):
    def test_deterministic_checksum_f32(self) -> None:
        ref = deterministic_reference_checksum(32, 32, 32, input_dtype="f32")
        self.assertEqual(ref["input_dtype"], "f32")
        self.assertIn("output_sum", ref)
        self.assertIn("output_l2", ref)

    def test_deterministic_checksum_i8(self) -> None:
        ref = deterministic_reference_checksum(32, 32, 32, input_dtype="i8")
        self.assertEqual(ref["input_dtype"], "i8")
        self.assertIn("output_sum", ref)
        self.assertIn("output_l2", ref)

    def test_deterministic_checksum_f16(self) -> None:
        ref = deterministic_reference_checksum(32, 32, 32, input_dtype="f16")
        self.assertEqual(ref["input_dtype"], "f16")
        self.assertIn("output_sum", ref)
        self.assertIn("output_l2", ref)


if __name__ == "__main__":
    unittest.main()
