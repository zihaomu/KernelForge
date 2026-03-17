# Gather/Embedding (GPU)

**Intent**: Optimize embedding lookup / gather by improving memory coalescing and caching behavior.

**When It Works**
- Large embedding tables with many lookups.
- When indices have locality (reuse within a block).

**Recognition Signals (Code)**
- Loads indexed rows; may use shared memory for indices or staging.
- Use of read-only cache (`__ldg`-like patterns) and vectorized loads.

**Tradeoffs**
- Random indices are bandwidth-bound; caching may not help.
