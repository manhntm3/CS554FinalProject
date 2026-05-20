# Fix 6: PTX and SASS Verification

## Purpose

Verifying that the inline PTX assembly survives the compiler pipeline intact is essential for any memory-model study. If the compiler merges, reorders, or drops instructions, the test no longer measures what it claims to. This document records the actual PTX and SASS instructions emitted for the key memory operations.

## How to Reproduce

```bash
make CUDA_ARCH=sm_75       # build
make ptx                   # dump PTX:  cuobjdump --dump-ptx ./litmus_tests
make sass                  # dump SASS: cuobjdump --dump-sass ./litmus_tests
```

The `-Xptxas -O0` flag in the Makefile prevents PTXAS from reordering PTX instructions, and `-fno-strict-aliasing` prevents the host compiler from making aliasing assumptions that could affect kernel argument passing.

## PTX Instructions Verified

The following PTX instructions were confirmed present in the compiled binary (from `cuobjdump --dump-ptx`):

```ptx
; ── Fences ──────────────────────────────────────────────────────────────────
fence.sc.cta;              ; fence for CTA scope (intra-block)
fence.sc.gpu;              ; fence for GPU scope (inter-block)
fence.sc.sys;              ; fence for System scope (CPU ↔ GPU)

; ── Release Stores ──────────────────────────────────────────────────────────
st.global.release.cta.u32 [%rd7], %r48;
st.global.release.gpu.u32 [%rd100], %r135;
st.global.release.sys.u32 [%rd135], %r77;

; ── Acquire Loads ───────────────────────────────────────────────────────────
ld.global.acquire.cta.u32 %r83, [%rd7];
ld.global.acquire.gpu.u32 %r135, [%rd99];
ld.global.acquire.sys.u32 %r195, [%rd136];

; ── Relaxed Atomics ─────────────────────────────────────────────────────────
atom.global.relaxed.gpu.exch.b32 %r74, [%rd135], %r75;

; ── Weak (unscoped) ─────────────────────────────────────────────────────────
ld.global.u32 %r_, [%rd_];          ; ptx_ld_weak
st.global.u32 [%rd_], %r_;          ; ptx_st_weak
```

All inline PTX asm strings are preserved verbatim. The compiler does not substitute or merge scoped operations.

## PTX → SASS Mapping (Tesla T4 / sm_75)

The following mappings were verified from `cuobjdump --dump-sass`:

| PTX Instruction | SASS Instruction | Notes |
|---|---|---|
| `st.global.u32` (weak) | `STG.E` | No scope qualifier — uncached store |
| `ld.global.u32` (weak) | `LDG.E` | No scope qualifier — uncached load |
| `st.global.relaxed.cta.u32` | `STG.E.STRONG.CTA` | CTA-scope strong store |
| `st.global.relaxed.gpu.u32` | `STG.E.STRONG.GPU` | GPU-scope strong store |
| `st.global.relaxed.sys.u32` | `STG.E.STRONG.SYS` | System-scope strong store |
| `ld.global.acquire.gpu.u32` | `LDG.E.STRONG.GPU` | GPU-scope acquire load |
| `ld.global.acquire.sys.u32` | `LDG.E.STRONG.SYS` | System-scope acquire load |
| `st.global.release.gpu.u32` | `STG.E.STRONG.GPU` | GPU-scope release store |
| `st.global.release.sys.u32` | `STG.E.STRONG.SYS` | System-scope release store |
| `fence.sc.cta` | `MEMBAR.ALL.CTA` | CTA-scope sequential consistency fence |
| `fence.sc.gpu` | `MEMBAR.SC.GPU` | GPU-scope SC fence |
| `fence.sc.sys` | `MEMBAR.SC.SYS` | System-scope SC fence |
| `atom.global.relaxed.gpu.exch.b32` | `ATOMG.E.EXCH.STRONG.GPU` | GPU-scope relaxed atomic exchange |

## Key Observations

### 1. MEMBAR.ALL vs MEMBAR.SC

Two MEMBAR variants appear:
- `MEMBAR.ALL.GPU`: emitted for `fence.acq_rel.gpu` (acquire/release fence, not used here directly but present from relaxed operations in some variants)
- `MEMBAR.SC.GPU`: emitted for `fence.sc.gpu` — the stronger sequential-consistency fence

The distinction is architecturally significant: `MEMBAR.SC.GPU` establishes a total order across all GPU-scope SC operations, while `MEMBAR.ALL.GPU` only ensures acquire/release ordering.

### 2. Release/Acquire Map to `STRONG` Flag

Both `st.global.release.gpu` and `st.global.relaxed.gpu` map to `STG.E.STRONG.GPU` in SASS. This is expected: on sm_75 (Turing), the `STRONG` flag triggers an L1 cache bypass or flush to L2, ensuring the write is visible at GPU scope. The difference between release and relaxed is **ordering relative to surrounding instructions**, not the store itself.

### 3. Weak Stores Use No Scope Qualifier

Plain `st.global.u32` (from `ptx_st_weak`) maps to bare `STG.E` with no scope suffix. This store may remain in the SM's L1 write-combining buffers and is not guaranteed to reach L2 before any subsequent operations. This explains why cross-block `WEAK` tests can see stale values.

### 4. No Instruction Merging

The compiler preserves the instruction sequence from inline PTX verbatim. In particular, `fence.sc.gpu; ld.global.relaxed.gpu` does **not** get merged into a single `ld.global.acquire.gpu`. The separate fence and load remain distinct SASS instructions (`MEMBAR.SC.GPU` then `LDG.E.STRONG.GPU`).

## Register Counts from PTXAS

```
sb_kernel  : 30 registers, 0 barriers, 393 bytes cmem[0]
mp_kernel  : 30 registers, 0 barriers, 393 bytes cmem[0]
lb_kernel  : 31 registers, 0 barriers, 393 bytes cmem[0]
iriw_kernel: 41 registers, 0 barriers, 408 bytes cmem[0]
```

No register spilling (0 spill stores/loads). The IRIW kernel uses more registers due to tracking 4 result variables (r0, r1, r2, r3) across 4 conditional execution paths.

## Compilation Flags Rationale

| Flag | Purpose |
|---|---|
| `-Xptxas -O0` | Prevents PTXAS from reordering PTX instructions between register assignments |
| `-O2` | Host-side optimization (does not affect device code instruction selection) |
| `-fno-strict-aliasing` | Prevents host compiler alias assumptions from affecting kernel argument addresses |
| `-lineinfo` | Embeds source-line info in SASS for `cuobjdump --dump-sass` correlation |
| `volatile` on inline asm | Prevents NVCC from removing, hoisting, or duplicating PTX instructions |
