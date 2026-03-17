# KV Cache Layout (GPU)

**Intent**: Store and access K/V cache with a layout that enables coalesced loads in attention decode/prefill.

**When It Works**
- Decode: repeated small queries over large cached keys/values.
- Prefill: bulk writes; alignment and strides are important.

**Recognition Signals (Code)**
- Tokens: `kv_cache`, `paged_kv`, `slot_mapping`, `kv_cache_loc`
- Indirection via `page_table`/`block_table`
- Stride math for `[seq, head, dim]` or blocked layouts

**Tradeoffs**
- Indirection improves memory usage but adds pointer chasing.
- Layout choices affect which dimension is contiguous (head vs dim vs seq).
