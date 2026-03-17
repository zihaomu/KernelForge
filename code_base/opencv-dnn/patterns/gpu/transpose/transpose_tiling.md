# Tiled Transpose (GPU)

**Intent**: Use shared memory tiling to transpose matrices/tensors while preserving coalesced accesses.

**When It Works**
- 2D/3D transposes where naive global loads/stores are uncoalesced.

**Recognition Signals (Code)**
- Shared-memory tile with padding to avoid bank conflicts.
- Two-phase load/store with swapped indices.

**Tradeoffs**
- Overhead can dominate for small tensors.
