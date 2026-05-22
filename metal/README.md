# Metal port — weak-memory litmus tests on Apple GPU

Self-contained Metal Shading Language port of the CUDA suite in
[../litmus_tests.cu](../litmus_tests.cu). Targets Apple Silicon (M1+) on
macOS 13+ with Metal 3.

## Layout

| File | Role |
| --- | --- |
| `litmus_tests.metal` | Kernels + PTX-wrapper analogs for the 4 tests |
| `litmus_tests.cpp`   | Host driver (metal-cpp), mirrors CUDA host code |
| `Makefile`           | Two-stage `.metal → .air → .metallib` + host build |
| `metal-cpp/`         | Apple's header-only C++ bindings (vendored) |

## Prerequisites

You need the Metal compiler toolchain — **Command Line Tools alone are
not enough**. One of:

1. **Full Xcode** from the App Store (heaviest).
2. **Metal Developer Tools for macOS** standalone — smaller download
   from https://developer.apple.com/download/all/ (search "Metal").

Verify it's installed:

```
xcrun --find metal && xcrun --find metallib
```

Both must print a path. If they error with "unable to find utility",
the toolchain isn't installed yet.

## Build & run

```
make            # builds litmus_tests + litmus_tests.metallib
./litmus_tests  # runs the full suite; takes a few seconds on M1
```

The binary loads `litmus_tests.metallib` from the current working
directory at startup, so run it from this folder.

## Output format

Mirrors the CUDA suite line-for-line so the two outputs can be
diffed side-by-side:

```
SB Inter-Block WEAK                    | Weak(0,0):    12345/5000000 =  0.2469%  [mean=0.2470% std=0.0103%]
  (r0, r1) distribution — 5000000 total obs (5 runs):
    (0, 0)* :    12345  (  0.2469%)
    (0, 1)  :  2493728  ( 49.8746%)
    ...
```

The `*` marks the "weak" cell — the outcome forbidden under sequential
consistency but allowed under the weak memory model.

## Variants implemented

All inter-block, device-scope:

| Test | Variants |
| --- | --- |
| SB   | WEAK, RELAXED_GPU, REL_ACQ_GPU, FENCE_SC_GPU, ATOMIC_RELAXED |
| MP   | WEAK, RELAXED_GPU, REL_ACQ_GPU, FENCE_SC_GPU |
| LB   | WEAK, RELAXED_GPU, ACQ_REL_GPU, FENCE_SC_GPU |
| IRIW | WEAK, RELAXED_GPU, REL_ACQ_GPU, FENCE_SC_GPU |

## Variants not ported (and why)

| CUDA variant | Status on Metal |
| --- | --- |
| `*_SYS`  | Metal has no system-scope kernel atomics. Dropped. |
| `*_CTA`  | Would require threadgroup-resident atomics + per-iteration `threadgroup_barrier`. The CUDA suite's array-stride trick (1M × 8 bytes) doesn't fit in the 32KB threadgroup memory budget, so a faithful port would need either a barriered reset loop (which dampens weak behaviour) or a much smaller iteration count. Left as future work. |

## Auditing compiler reorder defense

Metal has no `-Xptxas -O0` equivalent. The kernels use
`atomic_*_explicit` + `volatile` and rely on the MSL spec semantics.
To sanity-check that the compiler isn't folding ops across the
litmus-critical instructions:

```
make air-dump          # writes litmus_tests.air.txt
$EDITOR litmus_tests.air.txt
```

For each variant, confirm:
- Producer side: the two stores appear in source order with no
  intervening folding.
- Consumer side: the two loads appear in source order.
- FENCE variants: an `@llvm.air.atomic_fence` (or similarly named)
  intrinsic sits between the relevant ops.

If anything looks suspicious, insert an extra
`atomic_thread_fence(memory_order_relaxed, memory_scope_device)`
between ops as a compiler barrier — that's the closest MSL analog.

## Known caveats

- **`atomic_int` ↔ `int` aliasing.** For the WEAK variants, the same
  buffer is viewed as `device atomic_int*` and accessed via a
  `device volatile int*` cast. Layout-compatible on Apple GPUs;
  `volatile` keeps the compiler honest. Strictly UB per the MSL
  spec, but works in practice across M1/M2/M3.
- **SIMD-group width is hardcoded to 32** (the intra-block role
  assignment uses `tid == 32` for P1). All Apple Silicon GPUs to
  date use 32-wide SIMD-groups, so this is fine — but note it if
  porting to non-Apple Metal targets in the future.
- **Power management**: M1's GPU clocks down quickly. The first run
  of each test may be slightly noisier than the rest. The 5-run
  mean/stddev makes this visible.
