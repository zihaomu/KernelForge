# Double Buffer (GPU GEMM)

**Intent**: Overlap global memory loads with compute by ping-ponging shared memory stages.

**When It Works**
- Large-K loops where load latency is significant.
- Often paired with `cp.async` (Ampere+) or manual prefetch.

**Recognition Signals (Code)**
- Two shared buffers or stage index `% 2` (or more stages).
- Load stage `n+1` while computing stage `n`.
- `cp.async` / pipeline primitives on sm80+.

**Tradeoffs / Failure Modes**
- Increases shared memory footprint.
- Harder correctness and synchronization; can regress for small workloads.
