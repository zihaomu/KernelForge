# TopK / Selection (GPU)

**Intent**: Find top-k elements without fully sorting, using selection networks, partial sorts, or radix-based selection.

**When It Works**
- Small `k` (e.g., 1..128) relative to vocabulary/length.
- Logits/topk in decoding loops where latency matters.

**Recognition Signals (Code)**
- Bitonic / sorting networks, `nth_element`-like logic
- Shared memory heaps or per-thread candidate lists
- Use of warp-level primitives to merge candidates

**Tradeoffs**
- Large vocab topk becomes memory-bound; consider two-stage (block candidates + final merge).
- Numerical ties and stable ordering requirements complicate correctness.
