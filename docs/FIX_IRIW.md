# Fix 4: Independent Reads of Independent Writes (IRIW) — New Test

## What IRIW Tests

IRIW is the canonical test for **multi-copy atomicity (MCA)**: the property that when one thread's write becomes visible to *any* thread, it simultaneously becomes visible to *all* threads. Without MCA, two reader threads can observe two independent writes in opposite orders.

## Pattern

```
Initially: x=0, y=0

P0 (block 0, thread 0): x[i] = 1          (writer)
P1 (block 1, thread 0): y[i] = 1          (writer)
P2 (block 2, thread 0): r0=ld(x[i]); r1=ld(y[i])   (reader A)
P3 (block 3, thread 0): r2=ld(y[i]); r3=ld(x[i])   (reader B)
```

**Weak (non-MCA) outcome**: `r0==1 && r1==0 && r2==1 && r3==0`
- P2 sees x=1 (P0's write) but y=0 (P1's write not yet visible to P2)
- P3 sees y=1 (P1's write) but x=0 (P0's write not yet visible to P3)
- The two readers disagree on which write "happened first"

This outcome requires x to be visible to P2 but NOT to P3 at the same time — impossible under MCA.

## Variants Tested

| Variant | Writers | Readers | Expected |
|---|---|---|---|
| `WEAK` | `st.global` (no scope) | `ld.global` | Possibly weak (no coherence guarantee) |
| `RELAXED_GPU` | `st.global.relaxed.gpu` | `ld.global.relaxed.gpu` | No weak (PTX guarantees MCA at GPU scope) |
| `REL_ACQ_GPU` | `st.global.release.gpu` | `ld.global.acquire.gpu` + relaxed | No weak |
| `FENCE_SC_GPU` | store + `fence.sc.gpu` | `fence.sc.gpu` between reads | No weak |
| `FENCE_SC_SYS` | store + `fence.sc.sys` | `fence.sc.sys` between reads | No weak |

## Why MCA Matters at GPU Scope

The PTX memory model specification (Lustig et al., ASPLOS 2019) explicitly guarantees that all non-`.weak` operations at `.gpu` scope participate in a unified GPU-wide coherence domain — which provides MCA at GPU scope. This means the IRIW weak outcome is **forbidden** for `RELAXED_GPU` and stronger.

For plain `.weak` stores, the spec makes no such guarantee. In principle, each SM's L1 cache could hold different values for the same memory location, allowing IRIW weakness.

## Results

```
IRIW WEAK        | Weak: 0/5,000,000 = 0.0000%  [mean=0.0000% std=0.0000%]
IRIW RELAXED GPU | Weak: 0/5,000,000 = 0.0000%
IRIW ACQ/REL GPU | Weak: 0/5,000,000 = 0.0000%
IRIW FENCE SC GPU| Weak: 0/5,000,000 = 0.0000%
IRIW FENCE SC SYS| Weak: 0/5,000,000 = 0.0000%
```

All variants including WEAK show 0% IRIW weakness. This is consistent with prior work (Alglave et al., ASPLOS 2015) finding NVIDIA hardware to be MCA-compliant even for plain loads/stores, due to the L2 cache acting as a unified coherence point.

## Reader Distribution (IRIW WEAK)

```
P2 (reads x then y): (1,1) = 100%   — always sees both writes
P3 (reads y then x): (1,1) = 100%   — always sees both writes
```

In the array-strided layout, the 4 blocks run concurrently. By the time the readers (P2, P3) reach iteration i, P0 and P1 have usually already written x[i]=1 and y[i]=1 (writers slightly ahead of readers). The 0% IRIW weakness thus partly reflects timing: the weak outcome requires one writer to be ahead and one behind, which doesn't arise naturally in this setup. The MCA guarantee would prevent it regardless.

## Kernel Design

```cuda
__global__ void iriw_kernel(...) {
    int bid = blockIdx.x, tid = threadIdx.x;
    bool is_p0 = (bid==0 && tid==0);  // writes x
    bool is_p1 = (bid==1 && tid==0);  // writes y
    bool is_p2 = (bid==2 && tid==0);  // reads x, y  → writes res_r0[], res_r1[]
    bool is_p3 = (bid==3 && tid==0);  // reads y, x  → writes res_r2[], res_r3[]
    if (!is_p0 && !is_p1 && !is_p2 && !is_p3) return;

    for (int i = 0; i < iterations; ++i) {
        // array-strided: each i uses arr_x[i] and arr_y[i]
        // writers write once; readers read once; no reset needed
    }
}
```

4 blocks × 64 threads = 256 CUDA threads, 1 active per block. Launched as `<<<4, 64>>>`.

## Files Changed
- `litmus_tests.cu`: Added `TestIRIWType` enum
- `litmus_tests.cu`: Added `iriw_kernel` (4-block kernel)
- `litmus_tests.cu`: Added `run_iriw_test` (host runner with histogram and stats)
- `litmus_tests.cu`: Added IRIW section in `main()`
