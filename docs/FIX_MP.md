# Fix 2: Message Passing (MP) Kernel — Array-Strided Layout

## What Was Wrong

The original MP kernel used a **scalar reset strategy**: a single pair of scalar variables `(data, flag)` was reused across all iterations. After each iteration, P0 would reset `*data = 0; *flag = 0;` and call `__threadfence()`, then both threads would synchronize on a software spin-barrier.

```cuda
// OLD: scalar + reset per iteration
if (is_p0) {
    *data = 0; *flag = 0;
    __threadfence();  // NOT sufficient for GPU-scope atomics
}
global_spin_barrier(sync_barrier, (i * 2) + 2);  // software barrier
// ... then do MP operations
```

### Why This Caused False Weak Behavior

1. **L1 cache is per-SM and not coherent for non-atomic operations**: `*data = 0; *flag = 0;` are plain C++ stores, which map to `.weak` PTX stores (`st.global.u32`). These don't participate in the GPU-scope memory ordering domain.

2. **`__threadfence()` flushes P0's L1 to L2**, but it does **not** force the consumer (P1, on a different SM) to invalidate its L1 entries for `data` and `flag`.

3. **The spin barrier (`atomicAdd` + `volatile` spin) is insufficient**: Observing a `volatile` integer reach a threshold doesn't establish a happens-before relationship between P0's scoped stores/loads and P1's subsequent scoped loads. The PTX memory model requires matching acquire/release pairs at the **same scope** for this guarantee.

4. **Result**: P1 could exit the spin barrier but still read `flag=1` from iteration i-1 (stale in its L1 cache), then read `data=0` (reset by P0), counting as a false "MP weak" outcome.

This caused the absurd result: `Inter-Block FENCE SC (GPU)` showing **24.57% weak behavior** — more than the supposedly weaker `RELAXED GPU` — which is impossible under a correct implementation.

## The Fix: Array-Strided Layout

Replace the scalar+reset design with the same array-strided approach already used by the SB test:

```cuda
// NEW: each iteration uses a fresh memory location
__global__ void mp_kernel(int *arr_data, int *arr_flag, ...) {
    for (int i = 0; i < iterations; ++i) {
        int *data = &arr_data[i];  // fresh slot, initialized to 0 by host
        int *flag = &arr_flag[i];  // fresh slot, initialized to 0 by host

        if (is_p0) {
            // producer writes to fresh location — no reset needed
        }
        if (is_p1) {
            // consumer reads from fresh location
        }
    }
}
```

**Key advantages**:
- No `sync_barrier` needed — the two threads race independently on different iterations.
- No reset stores — the host pre-initializes all `arr_data` and `arr_flag` to 0 with a single `cudaMemset`.
- No cache pollution between iterations — each iteration's `data[i]` and `flag[i]` are distinct L2 cache lines.
- No interaction between plain C stores and GPU-scope atomics.

## Before vs After Results

| Variant | Before (scalar) | After (array) | Correct? |
|---|---|---|---|
| Inter-Block WEAK | 0.00% | 0.0000% | ✓ |
| Inter-Block RELAXED GPU | 15.09% | 0.0000% | ✓ Fixed |
| Inter-Block ACQ/REL GPU | 14.69% | 0.0000% | ✓ Fixed |
| Inter-Block FENCE SC GPU | 24.57% | 0.0000% | ✓ Fixed |
| Inter-Block FENCE SC SYS | 0.00% | 0.0000% | ✓ (was OK) |

All spurious weak behaviors eliminated. The fix reveals that the Tesla T4 enforces stronger-than-spec ordering for all tested MP variants.

## Why MP Shows 0% Weak on Turing

With the array fix, the consumer reads `flag[i]` and `data[i]` which start at 0. It sees either:
- `(flag=0, data=0)`: consumer ran before the producer
- `(flag=1, data=1)`: consumer ran after both producer writes
- `(flag=1, data=0)`: **MP weak** — would require the flag write to be visible before the data write

The 0% weak rate across all variants (including plain WEAK) indicates that even unordered stores from a single thread are observed in program order by other threads on Turing. This is an empirical demonstration that the T4 implements at least store-ordering (the "writes from the same thread are seen in order" property), even for `.weak` operations. This is stronger than what the PTX spec guarantees.

## Files Changed
- `litmus_tests.cu`: `mp_kernel` signature (`int *arr_data, int *arr_flag` replacing scalars)
- `litmus_tests.cu`: removed `sync_barrier` parameter and all barrier/reset logic from `mp_kernel`
- `litmus_tests.cu`: `run_mp_test` allocates arrays of size `iterations` for data and flag
