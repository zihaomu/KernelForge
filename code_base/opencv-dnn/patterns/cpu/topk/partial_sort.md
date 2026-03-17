# TopK / Partial Sort (CPU ARM)

**Intent**: Compute top-k using partial sorting (`nth_element`/heap) without full sort.

**When It Works**
- Small `k` relative to length/vocab.

**Recognition Signals (Code)**
- Heap maintenance or `nth_element` calls
- Two-stage: block candidates then merge

**Tradeoffs**
- Branching-heavy; SIMD helps less than cache locality.
