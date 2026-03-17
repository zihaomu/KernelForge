# Paged Attention / Blocked KV (GPU)

**Intent**: Use paged/block tables to index KV cache blocks, reducing fragmentation and enabling efficient decode.

**When It Works**
- Variable-length sequences and long contexts.
- Systems with dynamic allocation and reuse of KV blocks.

**Recognition Signals (Code)**
- Tokens: `paged_attention`, `page_table`, `block_table`, `paged_kv`
- Gather/scatter from KV blocks, often with head-dim vectorization

**Tradeoffs**
- Indirection can dominate if not cached or coalesced.
- Requires careful bounds/mask handling for partially filled blocks.
