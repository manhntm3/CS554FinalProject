/*
 * litmus_tests.cu  —  CUDA weak-memory litmus test suite
 *
 * Implements four canonical litmus tests, each as an inline-PTX CUDA kernel:
 *   SB   (Store Buffering)                     — store-load reordering
 *   MP   (Message Passing)                     — store-store / load-load ordering
 *   LB   (Load Buffering)                      — load-store causality cycles
 *   IRIW (Independent Reads of Independent Writes) — multi-copy atomicity
 *
 * Design principles:
 *   - Array-strided layout: each of the N iterations touches a fresh element,
 *     eliminating cross-iteration L1 cache pollution.
 *   - All memory ops are inline PTX volatile to prevent compiler reordering.
 *   - Every test runs N_RUNS independent times; results include per-run mean
 *     and stddev, plus a full outcome-distribution histogram.
 *
 * Fixes vs. previous version:
 *   [1] LB kernel: correct pattern (P0: r0=ld(y); st(x,1); P1: r1=ld(x); st(y,1))
 *       and correct weak-outcome counter (r0==1 && r1==1).
 *   [2] MP kernel: replaced scalar+barrier with array-strided layout, matching SB.
 *   [3] All tests: full (r0,r1) outcome histogram across all runs.
 *   [4] IRIW kernel: new 4-thread test for multi-copy atomicity.
 *   [5] Multi-run statistics: mean ± stddev of weak rate across N_RUNS.
 *   [6] PTX dump: see `make ptx` and docs/FIX_SASS_PTX.md for annotation.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _err = (expr);                                            \
        if (_err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s @ %s:%d -> %s\n",                 \
                    #expr, __FILE__, __LINE__, cudaGetErrorString(_err));      \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

static const int ITERATIONS = 1'000'000; // per run
static const int N_RUNS     = 5;          // independent repetitions

// ─────────────────────────────── enums ───────────────────────────────────────

enum TestSBType : int {
    TEST_SB_WEAK           = 0,
    TEST_SB_RELAXED_CTA    = 1,
    TEST_SB_RELAXED_GPU    = 2,
    TEST_SB_RELAXED_SYS    = 3,
    TEST_SB_FENCE_SC_GPU   = 4,
    TEST_SB_FENCE_SC_SYS   = 5,
    TEST_SB_REL_ACQ_GPU    = 6,
    TEST_SB_REL_ACQ_SYS    = 7,
    TEST_SB_ATOMIC_RELAXED = 8,
};

enum TestMPType : int {
    TEST_MP_WEAK          = 0,
    TEST_MP_RELAXED_CTA   = 1,
    TEST_MP_RELAXED_GPU   = 2,
    TEST_MP_RELAXED_SYS   = 3,
    TEST_MP_REL_ACQ_CTA   = 4,
    TEST_MP_REL_ACQ_GPU   = 5,
    TEST_MP_REL_ACQ_SYS   = 6,
    TEST_MP_FENCE_SC_CTA  = 7,
    TEST_MP_FENCE_SC_GPU  = 8,
    TEST_MP_FENCE_SC_SYS  = 9,
};

enum TestLBType : int {
    TEST_LB_WEAK          = 0,
    TEST_LB_RELAXED_CTA   = 1,
    TEST_LB_RELAXED_GPU   = 2,
    TEST_LB_RELAXED_SYS   = 3,
    TEST_LB_ACQ_REL_GPU   = 4,
    TEST_LB_FENCE_SC_GPU  = 5,
};

// [FIX 4] New enum for the IRIW test
enum TestIRIWType : int {
    TEST_IRIW_WEAK         = 0, // plain .weak st/ld
    TEST_IRIW_RELAXED_GPU  = 1, // relaxed.gpu
    TEST_IRIW_REL_ACQ_GPU  = 2, // writers: release.gpu; readers: acquire.gpu between loads
    TEST_IRIW_FENCE_SC_GPU = 3, // fence.sc.gpu between the two reader loads
    TEST_IRIW_FENCE_SC_SYS = 4, // fence.sc.sys
};

enum ScopeStrategy { SCOPE_CTA = 0, SCOPE_GPU = 1, SCOPE_SYS = 2 };

// ─────────────────────── PTX memory primitives ───────────────────────────────

__device__ __forceinline__ void ptx_st_weak(int *addr, int val) {
    asm volatile("st.global.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
}
__device__ __forceinline__ int ptx_ld_weak(int *addr) {
    int val;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    return val;
}

__device__ __forceinline__ void ptx_st_relaxed(int *addr, int val, int scope) {
    if      (scope == SCOPE_CTA)
        asm volatile("st.global.relaxed.cta.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
    else if (scope == SCOPE_GPU)
        asm volatile("st.global.relaxed.gpu.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
    else
        asm volatile("st.global.relaxed.sys.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
}
__device__ __forceinline__ int ptx_ld_relaxed(int *addr, int scope) {
    int val;
    if      (scope == SCOPE_CTA)
        asm volatile("ld.global.relaxed.cta.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    else if (scope == SCOPE_GPU)
        asm volatile("ld.global.relaxed.gpu.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    else
        asm volatile("ld.global.relaxed.sys.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    return val;
}

// Release store: all prior writes become visible before this store completes.
__device__ __forceinline__ void ptx_st_release(int *addr, int val, int scope) {
    if      (scope == SCOPE_CTA)
        asm volatile("st.global.release.cta.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
    else if (scope == SCOPE_GPU)
        asm volatile("st.global.release.gpu.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
    else
        asm volatile("st.global.release.sys.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
}

// Acquire load: all subsequent reads happen after this load's value is observed.
__device__ __forceinline__ int ptx_ld_acquire(int *addr, int scope) {
    int val;
    if      (scope == SCOPE_CTA)
        asm volatile("ld.global.acquire.cta.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    else if (scope == SCOPE_GPU)
        asm volatile("ld.global.acquire.gpu.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    else
        asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    return val;
}

__device__ __forceinline__ void ptx_atom_exch_relaxed(int *addr, int val, int scope) {
    if      (scope == SCOPE_CTA)
        asm volatile("atom.global.relaxed.cta.exch.b32 %0, [%1], %2;"
                     : "=r"(val) : "l"(addr), "r"(val) : "memory");
    else if (scope == SCOPE_GPU)
        asm volatile("atom.global.relaxed.gpu.exch.b32 %0, [%1], %2;"
                     : "=r"(val) : "l"(addr), "r"(val) : "memory");
    else
        asm volatile("atom.global.relaxed.sys.exch.b32 %0, [%1], %2;"
                     : "=r"(val) : "l"(addr), "r"(val) : "memory");
}

// Sequential-consistency fences at each scope.
// Maps to MEMBAR.CTA / MEMBAR.GL / MEMBAR.SYS in SASS.
__device__ __forceinline__ void ptx_fence_sc_cta() { asm volatile("fence.sc.cta;" ::: "memory"); }
__device__ __forceinline__ void ptx_fence_sc_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }
__device__ __forceinline__ void ptx_fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }

// ─────────────────────────── thread roles ────────────────────────────────────

__device__ __forceinline__ void get_roles(bool inter_block, bool &is_p0, bool &is_p1) {
    int tid = threadIdx.x, bid = blockIdx.x;
    is_p0 = is_p1 = false;
    if (inter_block) {
        // Different SMs: block 0 tid 0 = P0, block 1 tid 0 = P1
        if (tid == 0 && bid == 0) is_p0 = true;
        if (tid == 0 && bid == 1) is_p1 = true;
    } else {
        // Same SM, different warps: tid 0 = P0, tid 32 = P1 (avoids lockstep)
        if (bid == 0) {
            if (tid == 0)  is_p0 = true;
            if (tid == 32) is_p1 = true;
        }
    }
}

// ────────────────────── [FIX 3] Statistics and histogram helpers ──────────────

// 2-thread outcome histogram: c[r0][r1], r0,r1 ∈ {0,1}.
struct Hist2 {
    long long c[2][2];
    Hist2() { c[0][0] = c[0][1] = c[1][0] = c[1][1] = 0; }
    void record(int r0, int r1) {
        if (r0 >= 0 && r0 <= 1 && r1 >= 0 && r1 <= 1) c[r0][r1]++;
    }
    long long total() const { return c[0][0] + c[0][1] + c[1][0] + c[1][1]; }
    Hist2& operator+=(const Hist2& o) {
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                c[i][j] += o.c[i][j];
        return *this;
    }
};

struct Stats { double mean, stddev, mn, mx; };

static Stats compute_stats(const std::vector<double>& v) {
    double s = 0, s2 = 0, mn = v[0], mx = v[0];
    for (double x : v) { s += x; s2 += x * x; mn = std::min(mn, x); mx = std::max(mx, x); }
    int n = (int)v.size();
    double m = s / n;
    double var = std::max(0.0, s2 / n - m * m);
    return {m, std::sqrt(var), mn, mx};
}

// Print a 2×2 histogram. weak_XY = true marks that cell as the forbidden outcome.
static void print_hist2(const Hist2& h,
                        const char* r0_lbl, const char* r1_lbl,
                        bool w00, bool w10, bool w01, bool w11) {
    long long tot = h.total();
    if (tot == 0) { printf("    (no observations)\n"); return; }
    printf("  (%s, %s) distribution — %lld total obs (%d runs):\n",
           r0_lbl, r1_lbl, tot, N_RUNS);
    auto row = [&](int r0, int r1, bool is_weak) {
        printf("    (%d, %d)%s : %9lld  (%8.4f%%)\n",
               r0, r1, is_weak ? "*" : " ", h.c[r0][r1], 100.0 * h.c[r0][r1] / tot);
    };
    row(0, 0, w00); row(0, 1, w01); row(1, 0, w10); row(1, 1, w11);
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 1: Store Buffering (SB)
//
//   P0: st(x[i], 1);  r0 = ld(y[i]);
//   P1: st(y[i], 1);  r1 = ld(x[i]);
//   Weak outcome: r0==0 && r1==0  (both loads return initial value)
//
// Array-strided: x[i] and y[i] are always fresh (initialized to 0 by host).
// ─────────────────────────────────────────────────────────────────────────────

__global__ void sb_kernel(int *arr_x, int *arr_y,
                          int *result_0, int *result_1,
                          int iterations, int variant, bool inter_block) {
    bool is_p0, is_p1;
    get_roles(inter_block, is_p0, is_p1);
    if (!is_p0 && !is_p1) return;

    for (int i = 0; i < iterations; ++i) {
        int *ax = &arr_x[i], *ay = &arr_y[i];
        int r0 = -1, r1 = -1;

        if (is_p0) {
            switch (variant) {
                case TEST_SB_WEAK:
                    ptx_st_weak(ax, 1); r0 = ptx_ld_weak(ay); break;
                case TEST_SB_RELAXED_CTA:
                    ptx_st_relaxed(ax, 1, SCOPE_CTA); r0 = ptx_ld_relaxed(ay, SCOPE_CTA); break;
                case TEST_SB_RELAXED_GPU:
                    ptx_st_relaxed(ax, 1, SCOPE_GPU); r0 = ptx_ld_relaxed(ay, SCOPE_GPU); break;
                case TEST_SB_RELAXED_SYS:
                    ptx_st_relaxed(ax, 1, SCOPE_SYS); r0 = ptx_ld_relaxed(ay, SCOPE_SYS); break;
                case TEST_SB_FENCE_SC_GPU:
                    ptx_st_relaxed(ax, 1, SCOPE_GPU);
                    ptx_fence_sc_gpu();
                    r0 = ptx_ld_relaxed(ay, SCOPE_GPU); break;
                case TEST_SB_FENCE_SC_SYS:
                    ptx_st_relaxed(ax, 1, SCOPE_SYS);
                    ptx_fence_sc_sys();
                    r0 = ptx_ld_relaxed(ay, SCOPE_SYS); break;
                case TEST_SB_REL_ACQ_GPU:
                    // ACQ/REL prevents store-store and load-load reordering,
                    // but NOT store-load. SB is still theoretically allowed.
                    ptx_st_release(ax, 1, SCOPE_GPU);
                    r0 = ptx_ld_acquire(ay, SCOPE_GPU); break;
                case TEST_SB_REL_ACQ_SYS:
                    ptx_st_release(ax, 1, SCOPE_SYS);
                    r0 = ptx_ld_acquire(ay, SCOPE_SYS); break;
                case TEST_SB_ATOMIC_RELAXED:
                    ptx_atom_exch_relaxed(ax, 1, SCOPE_GPU);
                    r0 = ptx_ld_relaxed(ay, SCOPE_GPU); break;
            }
            result_0[i] = r0;
        }

        if (is_p1) {
            switch (variant) {
                case TEST_SB_WEAK:
                    ptx_st_weak(ay, 1); r1 = ptx_ld_weak(ax); break;
                case TEST_SB_RELAXED_CTA:
                    ptx_st_relaxed(ay, 1, SCOPE_CTA); r1 = ptx_ld_relaxed(ax, SCOPE_CTA); break;
                case TEST_SB_RELAXED_GPU:
                    ptx_st_relaxed(ay, 1, SCOPE_GPU); r1 = ptx_ld_relaxed(ax, SCOPE_GPU); break;
                case TEST_SB_RELAXED_SYS:
                    ptx_st_relaxed(ay, 1, SCOPE_SYS); r1 = ptx_ld_relaxed(ax, SCOPE_SYS); break;
                case TEST_SB_FENCE_SC_GPU:
                    ptx_st_relaxed(ay, 1, SCOPE_GPU);
                    ptx_fence_sc_gpu();
                    r1 = ptx_ld_relaxed(ax, SCOPE_GPU); break;
                case TEST_SB_FENCE_SC_SYS:
                    ptx_st_relaxed(ay, 1, SCOPE_SYS);
                    ptx_fence_sc_sys();
                    r1 = ptx_ld_relaxed(ax, SCOPE_SYS); break;
                case TEST_SB_REL_ACQ_GPU:
                    ptx_st_release(ay, 1, SCOPE_GPU);
                    r1 = ptx_ld_acquire(ax, SCOPE_GPU); break;
                case TEST_SB_REL_ACQ_SYS:
                    ptx_st_release(ay, 1, SCOPE_SYS);
                    r1 = ptx_ld_acquire(ax, SCOPE_SYS); break;
                case TEST_SB_ATOMIC_RELAXED:
                    ptx_atom_exch_relaxed(ay, 1, SCOPE_GPU);
                    r1 = ptx_ld_relaxed(ax, SCOPE_GPU); break;
            }
            result_1[i] = r1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 2: Message Passing (MP) — [FIX 2] array-strided, no barrier/reset
//
//   P0 (producer): data[i]=1; flag[i]=1;
//   P1 (consumer): r_flag=ld(flag[i]); r_data=ld(data[i]);
//   Weak outcome: r_flag==1 && r_data==0
//
// The previous implementation reset a single scalar (data,flag) per iteration
// using plain C++ stores and a software spin-barrier. This did not adequately
// flush per-SM L1 caches between iterations, producing false weak outcomes for
// GPU-scope ACQ/REL and FENCE SC variants. The array layout gives each
// iteration its own fresh, pre-zeroed memory locations.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void mp_kernel(int *arr_data, int *arr_flag,
                          int *result_flag, int *result_data,
                          int iterations, int variant, bool inter_block) {
    bool is_p0, is_p1;
    get_roles(inter_block, is_p0, is_p1);
    if (!is_p0 && !is_p1) return;

    for (int i = 0; i < iterations; ++i) {
        int *data = &arr_data[i], *flag = &arr_flag[i];
        int r_flag = -1, r_data = -1;

        if (is_p0) {
            switch (variant) {
                case TEST_MP_WEAK:
                    ptx_st_weak(data, 1); ptx_st_weak(flag, 1); break;
                case TEST_MP_RELAXED_CTA:
                    ptx_st_relaxed(data, 1, SCOPE_CTA);
                    ptx_st_relaxed(flag, 1, SCOPE_CTA); break;
                case TEST_MP_RELAXED_GPU:
                    ptx_st_relaxed(data, 1, SCOPE_GPU);
                    ptx_st_relaxed(flag, 1, SCOPE_GPU); break;
                case TEST_MP_RELAXED_SYS:
                    ptx_st_relaxed(data, 1, SCOPE_SYS);
                    ptx_st_relaxed(flag, 1, SCOPE_SYS); break;
                case TEST_MP_REL_ACQ_CTA:
                    ptx_st_relaxed(data, 1, SCOPE_CTA);
                    ptx_st_release(flag, 1, SCOPE_CTA); break;
                case TEST_MP_REL_ACQ_GPU:
                    // Release store to flag ensures data write is visible
                    // before flag becomes visible to any GPU-scope acquire reader.
                    ptx_st_relaxed(data, 1, SCOPE_GPU);
                    ptx_st_release(flag, 1, SCOPE_GPU); break;
                case TEST_MP_REL_ACQ_SYS:
                    ptx_st_relaxed(data, 1, SCOPE_SYS);
                    ptx_st_release(flag, 1, SCOPE_SYS); break;
                case TEST_MP_FENCE_SC_CTA:
                    ptx_st_relaxed(data, 1, SCOPE_CTA); break;
                case TEST_MP_FENCE_SC_GPU:
                    ptx_st_relaxed(data, 1, SCOPE_GPU);
                    ptx_fence_sc_gpu();
                    ptx_st_relaxed(flag, 1, SCOPE_GPU); break;
                case TEST_MP_FENCE_SC_SYS:
                    ptx_st_relaxed(data, 1, SCOPE_SYS);
                    ptx_fence_sc_sys();
                    ptx_st_relaxed(flag, 1, SCOPE_SYS); break;
            }
        }

        if (is_p1) {
            switch (variant) {
                case TEST_MP_WEAK:
                    r_flag = ptx_ld_weak(flag); r_data = ptx_ld_weak(data); break;
                case TEST_MP_RELAXED_CTA:
                    r_flag = ptx_ld_relaxed(flag, SCOPE_CTA);
                    r_data = ptx_ld_relaxed(data, SCOPE_CTA); break;
                case TEST_MP_RELAXED_GPU:
                    r_flag = ptx_ld_relaxed(flag, SCOPE_GPU);
                    r_data = ptx_ld_relaxed(data, SCOPE_GPU); break;
                case TEST_MP_RELAXED_SYS:
                    r_flag = ptx_ld_relaxed(flag, SCOPE_SYS);
                    r_data = ptx_ld_relaxed(data, SCOPE_SYS); break;
                case TEST_MP_REL_ACQ_CTA:
                    r_flag = ptx_ld_acquire(flag, SCOPE_CTA);
                    r_data = ptx_ld_relaxed(data, SCOPE_CTA); break;
                case TEST_MP_REL_ACQ_GPU:
                    // Acquire load of flag: if this returns 1, the release store
                    // has completed, so data==1 must follow at GPU scope.
                    r_flag = ptx_ld_acquire(flag, SCOPE_GPU);
                    r_data = ptx_ld_relaxed(data, SCOPE_GPU); break;
                case TEST_MP_REL_ACQ_SYS:
                    r_flag = ptx_ld_acquire(flag, SCOPE_SYS);
                    r_data = ptx_ld_relaxed(data, SCOPE_SYS); break;
                case TEST_MP_FENCE_SC_CTA:
                    r_flag = ptx_ld_relaxed(flag, SCOPE_CTA);
                    ptx_fence_sc_cta();
                    r_data = ptx_ld_relaxed(data, SCOPE_CTA); break;
                case TEST_MP_FENCE_SC_GPU:
                    r_flag = ptx_ld_relaxed(flag, SCOPE_GPU);
                    ptx_fence_sc_gpu();
                    r_data = ptx_ld_relaxed(data, SCOPE_GPU); break;
                case TEST_MP_FENCE_SC_SYS:
                    r_flag = ptx_ld_relaxed(flag, SCOPE_SYS);
                    ptx_fence_sc_sys();
                    r_data = ptx_ld_relaxed(data, SCOPE_SYS); break;
            }
            result_flag[i] = r_flag;
            result_data[i] = r_data;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 3: Load Buffering (LB) — [FIX 1] correct pattern + correct counter
//
//   P0: r0 = ld(y[i]);  st(x[i], 1);
//   P1: r1 = ld(x[i]);  st(y[i], 1);
//   Weak outcome: r0==1 && r1==1  (causality cycle: each load saw the other's store)
//
// Previous bug: P0 stored the loaded value (always 0) instead of 1, making
// weak behavior impossible and the test trivially vacuous.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void lb_kernel(int *arr_x, int *arr_y,
                          int *result_0, int *result_1,
                          int iterations, int variant, bool inter_block) {
    bool is_p0, is_p1;
    get_roles(inter_block, is_p0, is_p1);
    if (!is_p0 && !is_p1) return;

    for (int i = 0; i < iterations; ++i) {
        int *ax = &arr_x[i], *ay = &arr_y[i];
        int r0 = -1, r1 = -1;

        if (is_p0) {
            // FIXED: load from y first, then store 1 to x
            switch (variant) {
                case TEST_LB_WEAK:
                    r0 = ptx_ld_weak(ay);
                    ptx_st_weak(ax, 1); break;
                case TEST_LB_RELAXED_CTA:
                    r0 = ptx_ld_relaxed(ay, SCOPE_CTA);
                    ptx_st_relaxed(ax, 1, SCOPE_CTA); break;
                case TEST_LB_RELAXED_GPU:
                    r0 = ptx_ld_relaxed(ay, SCOPE_GPU);
                    ptx_st_relaxed(ax, 1, SCOPE_GPU); break;
                case TEST_LB_RELAXED_SYS:
                    r0 = ptx_ld_relaxed(ay, SCOPE_SYS);
                    ptx_st_relaxed(ax, 1, SCOPE_SYS); break;
                case TEST_LB_ACQ_REL_GPU:
                    // Acquire load then release store: prevents load-load and
                    // store-store reordering, but a causality cycle still requires
                    // an additional global ordering guarantee.
                    r0 = ptx_ld_acquire(ay, SCOPE_GPU);
                    ptx_st_release(ax, 1, SCOPE_GPU); break;
                case TEST_LB_FENCE_SC_GPU:
                    r0 = ptx_ld_relaxed(ay, SCOPE_GPU);
                    ptx_fence_sc_gpu();
                    ptx_st_relaxed(ax, 1, SCOPE_GPU); break;
            }
            result_0[i] = r0;
        }

        if (is_p1) {
            // FIXED: load from x first, then store 1 to y
            switch (variant) {
                case TEST_LB_WEAK:
                    r1 = ptx_ld_weak(ax);
                    ptx_st_weak(ay, 1); break;
                case TEST_LB_RELAXED_CTA:
                    r1 = ptx_ld_relaxed(ax, SCOPE_CTA);
                    ptx_st_relaxed(ay, 1, SCOPE_CTA); break;
                case TEST_LB_RELAXED_GPU:
                    r1 = ptx_ld_relaxed(ax, SCOPE_GPU);
                    ptx_st_relaxed(ay, 1, SCOPE_GPU); break;
                case TEST_LB_RELAXED_SYS:
                    r1 = ptx_ld_relaxed(ax, SCOPE_SYS);
                    ptx_st_relaxed(ay, 1, SCOPE_SYS); break;
                case TEST_LB_ACQ_REL_GPU:
                    r1 = ptx_ld_acquire(ax, SCOPE_GPU);
                    ptx_st_release(ay, 1, SCOPE_GPU); break;
                case TEST_LB_FENCE_SC_GPU:
                    r1 = ptx_ld_relaxed(ax, SCOPE_GPU);
                    ptx_fence_sc_gpu();
                    ptx_st_relaxed(ay, 1, SCOPE_GPU); break;
            }
            result_1[i] = r1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 4: Independent Reads of Independent Writes (IRIW) — [FIX 4] NEW
//
//   P0 (block 0): x[i] = 1             (writer for x)
//   P1 (block 1): y[i] = 1             (writer for y)
//   P2 (block 2): r0=ld(x[i]); r1=ld(y[i])   (reader A: x first, then y)
//   P3 (block 3): r2=ld(y[i]); r3=ld(x[i])   (reader B: y first, then x)
//
//   Weak (non-MCA) outcome: r0==1 && r1==0 && r2==1 && r3==0
//     P2 sees x's write but not y's; P3 sees y's write but not x's.
//     Readers "disagree" on which write became globally visible first.
//     This is only possible if the GPU lacks multi-copy atomicity (MCA).
//     PTX guarantees MCA for GPU-scope scoped operations, so RELAXED_GPU
//     and stronger variants should never produce this outcome.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void iriw_kernel(int *arr_x, int *arr_y,
                            int *res_r0, int *res_r1,  // P2: ld(x), ld(y)
                            int *res_r2, int *res_r3,  // P3: ld(y), ld(x)
                            int iterations, int variant) {
    int bid = blockIdx.x, tid = threadIdx.x;
    bool is_p0 = (bid == 0 && tid == 0); // writes x
    bool is_p1 = (bid == 1 && tid == 0); // writes y
    bool is_p2 = (bid == 2 && tid == 0); // reads x then y
    bool is_p3 = (bid == 3 && tid == 0); // reads y then x
    if (!is_p0 && !is_p1 && !is_p2 && !is_p3) return;

    for (int i = 0; i < iterations; ++i) {
        int *ax = &arr_x[i], *ay = &arr_y[i];

        if (is_p0) {
            switch (variant) {
                case TEST_IRIW_WEAK:         ptx_st_weak(ax, 1);                              break;
                case TEST_IRIW_RELAXED_GPU:  ptx_st_relaxed(ax, 1, SCOPE_GPU);               break;
                case TEST_IRIW_REL_ACQ_GPU:  ptx_st_release(ax, 1, SCOPE_GPU);               break;
                case TEST_IRIW_FENCE_SC_GPU: ptx_st_relaxed(ax, 1, SCOPE_GPU); ptx_fence_sc_gpu(); break;
                case TEST_IRIW_FENCE_SC_SYS: ptx_st_relaxed(ax, 1, SCOPE_SYS); ptx_fence_sc_sys(); break;
            }
        }
        if (is_p1) {
            switch (variant) {
                case TEST_IRIW_WEAK:         ptx_st_weak(ay, 1);                              break;
                case TEST_IRIW_RELAXED_GPU:  ptx_st_relaxed(ay, 1, SCOPE_GPU);               break;
                case TEST_IRIW_REL_ACQ_GPU:  ptx_st_release(ay, 1, SCOPE_GPU);               break;
                case TEST_IRIW_FENCE_SC_GPU: ptx_st_relaxed(ay, 1, SCOPE_GPU); ptx_fence_sc_gpu(); break;
                case TEST_IRIW_FENCE_SC_SYS: ptx_st_relaxed(ay, 1, SCOPE_SYS); ptx_fence_sc_sys(); break;
            }
        }
        if (is_p2) {
            int r0 = -1, r1 = -1;
            switch (variant) {
                case TEST_IRIW_WEAK:
                    r0 = ptx_ld_weak(ax);               r1 = ptx_ld_weak(ay);              break;
                case TEST_IRIW_RELAXED_GPU:
                    r0 = ptx_ld_relaxed(ax, SCOPE_GPU); r1 = ptx_ld_relaxed(ay, SCOPE_GPU); break;
                case TEST_IRIW_REL_ACQ_GPU:
                    r0 = ptx_ld_acquire(ax, SCOPE_GPU); r1 = ptx_ld_relaxed(ay, SCOPE_GPU); break;
                case TEST_IRIW_FENCE_SC_GPU:
                    r0 = ptx_ld_relaxed(ax, SCOPE_GPU);
                    ptx_fence_sc_gpu();
                    r1 = ptx_ld_relaxed(ay, SCOPE_GPU); break;
                case TEST_IRIW_FENCE_SC_SYS:
                    r0 = ptx_ld_relaxed(ax, SCOPE_SYS);
                    ptx_fence_sc_sys();
                    r1 = ptx_ld_relaxed(ay, SCOPE_SYS); break;
            }
            res_r0[i] = r0; res_r1[i] = r1;
        }
        if (is_p3) {
            int r2 = -1, r3 = -1;
            switch (variant) {
                case TEST_IRIW_WEAK:
                    r2 = ptx_ld_weak(ay);               r3 = ptx_ld_weak(ax);              break;
                case TEST_IRIW_RELAXED_GPU:
                    r2 = ptx_ld_relaxed(ay, SCOPE_GPU); r3 = ptx_ld_relaxed(ax, SCOPE_GPU); break;
                case TEST_IRIW_REL_ACQ_GPU:
                    r2 = ptx_ld_acquire(ay, SCOPE_GPU); r3 = ptx_ld_relaxed(ax, SCOPE_GPU); break;
                case TEST_IRIW_FENCE_SC_GPU:
                    r2 = ptx_ld_relaxed(ay, SCOPE_GPU);
                    ptx_fence_sc_gpu();
                    r3 = ptx_ld_relaxed(ax, SCOPE_GPU); break;
                case TEST_IRIW_FENCE_SC_SYS:
                    r2 = ptx_ld_relaxed(ay, SCOPE_SYS);
                    ptx_fence_sc_sys();
                    r3 = ptx_ld_relaxed(ax, SCOPE_SYS); break;
            }
            res_r2[i] = r2; res_r3[i] = r3;
        }
    }
}

// ─────────────── [FIX 5] host run functions with multi-run stats ─────────────

void run_sb_test(int iterations, bool inter_block, int variant, const char* label) {
    int *d_x, *d_y, *d_r0, *d_r1;
    CUDA_CHECK(cudaMalloc(&d_x,  iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y,  iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r0, iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r1, iterations * sizeof(int)));
    int *h_r0 = new int[iterations], *h_r1 = new int[iterations];

    Hist2 total;
    std::vector<double> rates;
    int blocks = inter_block ? 2 : 1;

    for (int run = 0; run < N_RUNS; ++run) {
        CUDA_CHECK(cudaMemset(d_x, 0, iterations * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_y, 0, iterations * sizeof(int)));
        sb_kernel<<<blocks, 64>>>(d_x, d_y, d_r0, d_r1, iterations, variant, inter_block);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_r0, d_r0, iterations * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r1, d_r1, iterations * sizeof(int), cudaMemcpyDeviceToHost));

        Hist2 h; long long weak = 0;
        for (int i = 0; i < iterations; i++) {
            h.record(h_r0[i], h_r1[i]);
            if (h_r0[i] == 0 && h_r1[i] == 0) weak++;
        }
        total += h;
        rates.push_back(100.0 * weak / iterations);
    }

    Stats s = compute_stats(rates);
    long long tot = total.total();
    printf("%-38s | Weak(0,0): %8lld/%lld = %7.4f%%  [mean=%7.4f%% std=%6.4f%%]\n",
           label, total.c[0][0], tot, 100.0 * total.c[0][0] / tot, s.mean, s.stddev);
    print_hist2(total, "r0", "r1", true, false, false, false);
    printf("\n");

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_r0); cudaFree(d_r1);
    delete[] h_r0; delete[] h_r1;
}

void run_mp_test(int iterations, bool inter_block, int variant, const char* label) {
    int *d_data, *d_flag, *d_rf, *d_rd;
    CUDA_CHECK(cudaMalloc(&d_data, iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flag, iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rf,   iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rd,   iterations * sizeof(int)));
    int *h_rf = new int[iterations], *h_rd = new int[iterations];

    Hist2 total;
    std::vector<double> rates;
    int blocks = inter_block ? 2 : 1;

    for (int run = 0; run < N_RUNS; ++run) {
        CUDA_CHECK(cudaMemset(d_data, 0, iterations * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_flag, 0, iterations * sizeof(int)));
        mp_kernel<<<blocks, 64>>>(d_data, d_flag, d_rf, d_rd, iterations, variant, inter_block);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_rf, d_rf, iterations * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rd, d_rd, iterations * sizeof(int), cudaMemcpyDeviceToHost));

        Hist2 h; long long weak = 0;
        for (int i = 0; i < iterations; i++) {
            h.record(h_rf[i], h_rd[i]);
            if (h_rf[i] == 1 && h_rd[i] == 0) weak++;
        }
        total += h;
        rates.push_back(100.0 * weak / iterations);
    }

    Stats s = compute_stats(rates);
    long long tot = total.total();
    // Weak outcome is c[1][0]: r_flag=1, r_data=0
    printf("%-38s | Weak(f=1,d=0): %8lld/%lld = %7.4f%%  [mean=%7.4f%% std=%6.4f%%]\n",
           label, total.c[1][0], tot, 100.0 * total.c[1][0] / tot, s.mean, s.stddev);
    print_hist2(total, "r_flag", "r_data", false, true, false, false);
    printf("\n");

    cudaFree(d_data); cudaFree(d_flag); cudaFree(d_rf); cudaFree(d_rd);
    delete[] h_rf; delete[] h_rd;
}

void run_lb_test(int iterations, bool inter_block, int variant, const char* label) {
    int *d_x, *d_y, *d_r0, *d_r1;
    CUDA_CHECK(cudaMalloc(&d_x,  iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y,  iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r0, iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r1, iterations * sizeof(int)));
    int *h_r0 = new int[iterations], *h_r1 = new int[iterations];

    Hist2 total;
    std::vector<double> rates;
    int blocks = inter_block ? 2 : 1;

    for (int run = 0; run < N_RUNS; ++run) {
        CUDA_CHECK(cudaMemset(d_x, 0, iterations * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_y, 0, iterations * sizeof(int)));
        lb_kernel<<<blocks, 64>>>(d_x, d_y, d_r0, d_r1, iterations, variant, inter_block);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_r0, d_r0, iterations * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r1, d_r1, iterations * sizeof(int), cudaMemcpyDeviceToHost));

        Hist2 h; long long weak = 0;
        for (int i = 0; i < iterations; i++) {
            h.record(h_r0[i], h_r1[i]);
            if (h_r0[i] == 1 && h_r1[i] == 1) weak++; // FIXED: weak = both see 1
        }
        total += h;
        rates.push_back(100.0 * weak / iterations);
    }

    Stats s = compute_stats(rates);
    long long tot = total.total();
    printf("%-38s | Weak(1,1): %8lld/%lld = %7.4f%%  [mean=%7.4f%% std=%6.4f%%]\n",
           label, total.c[1][1], tot, 100.0 * total.c[1][1] / tot, s.mean, s.stddev);
    print_hist2(total, "r0", "r1", false, false, false, true);
    printf("\n");

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_r0); cudaFree(d_r1);
    delete[] h_r0; delete[] h_r1;
}

void run_iriw_test(int iterations, int variant, const char* label) {
    int *d_x, *d_y, *d_r0, *d_r1, *d_r2, *d_r3;
    CUDA_CHECK(cudaMalloc(&d_x,  iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y,  iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r0, iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r1, iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r2, iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r3, iterations * sizeof(int)));
    int *h_r0 = new int[iterations], *h_r1 = new int[iterations];
    int *h_r2 = new int[iterations], *h_r3 = new int[iterations];

    Hist2 hist_p2, hist_p3; // P2: (ld x, ld y); P3: (ld y, ld x)
    long long total_weak = 0, total_obs = 0;
    std::vector<double> rates;

    for (int run = 0; run < N_RUNS; ++run) {
        CUDA_CHECK(cudaMemset(d_x, 0, iterations * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_y, 0, iterations * sizeof(int)));
        iriw_kernel<<<4, 64>>>(d_x, d_y, d_r0, d_r1, d_r2, d_r3, iterations, variant);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_r0, d_r0, iterations * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r1, d_r1, iterations * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r2, d_r2, iterations * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_r3, d_r3, iterations * sizeof(int), cudaMemcpyDeviceToHost));

        long long weak = 0, obs = 0;
        for (int i = 0; i < iterations; i++) {
            int r0=h_r0[i], r1=h_r1[i], r2=h_r2[i], r3=h_r3[i];
            bool valid = (r0>=0&&r0<=1&&r1>=0&&r1<=1&&r2>=0&&r2<=1&&r3>=0&&r3<=1);
            if (valid) {
                obs++;
                hist_p2.record(r0, r1);
                hist_p3.record(r2, r3);
                if (r0==1 && r1==0 && r2==1 && r3==0) weak++;
            }
        }
        total_weak += weak;
        total_obs  += obs;
        rates.push_back(obs > 0 ? 100.0 * weak / obs : 0.0);
    }

    Stats s = compute_stats(rates);
    printf("%-38s | Weak(r0=1,r1=0,r2=1,r3=0): %lld/%lld = %7.4f%%  [mean=%7.4f%% std=%6.4f%%]\n",
           label, total_weak, total_obs,
           total_obs > 0 ? 100.0 * total_weak / total_obs : 0.0,
           s.mean, s.stddev);

    // P2 view: reads x then y
    long long p2t = hist_p2.total(), p3t = hist_p3.total();
    if (p2t > 0) {
        printf("  P2 (x→y): (0,0)=%lld(%.1f%%) (1,0)*=%lld(%.1f%%) (0,1)=%lld(%.1f%%) (1,1)=%lld(%.1f%%)\n",
               hist_p2.c[0][0], 100.0*hist_p2.c[0][0]/p2t,
               hist_p2.c[1][0], 100.0*hist_p2.c[1][0]/p2t,
               hist_p2.c[0][1], 100.0*hist_p2.c[0][1]/p2t,
               hist_p2.c[1][1], 100.0*hist_p2.c[1][1]/p2t);
    }
    if (p3t > 0) {
        printf("  P3 (y→x): (0,0)=%lld(%.1f%%) (1,0)*=%lld(%.1f%%) (0,1)=%lld(%.1f%%) (1,1)=%lld(%.1f%%)\n",
               hist_p3.c[0][0], 100.0*hist_p3.c[0][0]/p3t,
               hist_p3.c[1][0], 100.0*hist_p3.c[1][0]/p3t,
               hist_p3.c[0][1], 100.0*hist_p3.c[0][1]/p3t,
               hist_p3.c[1][1], 100.0*hist_p3.c[1][1]/p3t);
    }
    printf("\n");

    cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_r0); cudaFree(d_r1); cudaFree(d_r2); cudaFree(d_r3);
    delete[] h_r0; delete[] h_r1; delete[] h_r2; delete[] h_r3;
}

// ─────────────────────────────── main ────────────────────────────────────────

int main() {
    printf("Hardware: Tesla T4 (sm_75) | CUDA 12.8 | %d runs x %d iter = %lld total obs/test\n\n",
           N_RUNS, ITERATIONS, (long long)N_RUNS * ITERATIONS);

    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("STORE BUFFERING (SB)   P0: st(x,1); r0=ld(y)  ||  P1: st(y,1); r1=ld(x)\n");
    printf("Weak outcome: r0==0 && r1==0   (* marks weak cell in distribution)\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");
    run_sb_test(ITERATIONS, true,  TEST_SB_WEAK,          "SB Inter-Block WEAK");
    run_sb_test(ITERATIONS, true,  TEST_SB_RELAXED_GPU,   "SB Inter-Block RELAXED (GPU)");
    run_sb_test(ITERATIONS, true,  TEST_SB_RELAXED_SYS,   "SB Inter-Block RELAXED (SYS)");
    run_sb_test(ITERATIONS, true,  TEST_SB_RELAXED_CTA,   "SB Inter-Block RELAXED (CTA)");
    run_sb_test(ITERATIONS, false, TEST_SB_RELAXED_CTA,   "SB Intra-Block RELAXED (CTA)");
    run_sb_test(ITERATIONS, true,  TEST_SB_REL_ACQ_GPU,   "SB Inter-Block ACQ/REL (GPU)");
    run_sb_test(ITERATIONS, true,  TEST_SB_FENCE_SC_GPU,  "SB Inter-Block FENCE SC (GPU)");
    run_sb_test(ITERATIONS, true,  TEST_SB_FENCE_SC_SYS,  "SB Inter-Block FENCE SC (SYS)");

    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("MESSAGE PASSING (MP)   P0: data=1; flag=1  ||  P1: r_flag=ld(flag); r_data=ld(data)\n");
    printf("Weak outcome: r_flag==1 && r_data==0\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");
    run_mp_test(ITERATIONS, true,  TEST_MP_WEAK,          "MP Inter-Block WEAK");
    run_mp_test(ITERATIONS, false, TEST_MP_RELAXED_CTA,   "MP Intra-Block RELAXED (CTA)");
    run_mp_test(ITERATIONS, true,  TEST_MP_RELAXED_GPU,   "MP Inter-Block RELAXED (GPU)");
    run_mp_test(ITERATIONS, true,  TEST_MP_RELAXED_SYS,   "MP Inter-Block RELAXED (SYS)");
    run_mp_test(ITERATIONS, false, TEST_MP_REL_ACQ_CTA,   "MP Intra-Block ACQ/REL (CTA)");
    run_mp_test(ITERATIONS, true,  TEST_MP_REL_ACQ_GPU,   "MP Inter-Block ACQ/REL (GPU)");
    run_mp_test(ITERATIONS, true,  TEST_MP_REL_ACQ_SYS,   "MP Inter-Block ACQ/REL (SYS)");
    run_mp_test(ITERATIONS, false, TEST_MP_FENCE_SC_CTA,  "MP Intra-Block FENCE SC (CTA)");
    run_mp_test(ITERATIONS, true,  TEST_MP_FENCE_SC_GPU,  "MP Inter-Block FENCE SC (GPU)");
    run_mp_test(ITERATIONS, true,  TEST_MP_FENCE_SC_SYS,  "MP Inter-Block FENCE SC (SYS)");

    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("LOAD BUFFERING (LB)   P0: r0=ld(y); st(x,1)  ||  P1: r1=ld(x); st(y,1)\n");
    printf("Weak outcome: r0==1 && r1==1   (causality cycle)\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");
    run_lb_test(ITERATIONS, true,  TEST_LB_WEAK,          "LB Inter-Block WEAK");
    run_lb_test(ITERATIONS, true,  TEST_LB_RELAXED_GPU,   "LB Inter-Block RELAXED (GPU)");
    run_lb_test(ITERATIONS, true,  TEST_LB_RELAXED_SYS,   "LB Inter-Block RELAXED (SYS)");
    run_lb_test(ITERATIONS, true,  TEST_LB_ACQ_REL_GPU,   "LB Inter-Block ACQ/REL (GPU)");
    run_lb_test(ITERATIONS, true,  TEST_LB_FENCE_SC_GPU,  "LB Inter-Block FENCE SC (GPU)");

    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("INDEP. READS OF INDEP. WRITES (IRIW)  — 4-thread / 4-block test\n");
    printf("  P0: x=1  ||  P1: y=1  ||  P2: r0=ld(x); r1=ld(y)  ||  P3: r2=ld(y); r3=ld(x)\n");
    printf("Weak (non-MCA) outcome: r0==1 && r1==0 && r2==1 && r3==0\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");
    run_iriw_test(ITERATIONS, TEST_IRIW_WEAK,         "IRIW WEAK");
    run_iriw_test(ITERATIONS, TEST_IRIW_RELAXED_GPU,  "IRIW RELAXED (GPU)");
    run_iriw_test(ITERATIONS, TEST_IRIW_REL_ACQ_GPU,  "IRIW ACQ/REL (GPU)");
    run_iriw_test(ITERATIONS, TEST_IRIW_FENCE_SC_GPU, "IRIW FENCE SC (GPU)");
    run_iriw_test(ITERATIONS, TEST_IRIW_FENCE_SC_SYS, "IRIW FENCE SC (SYS)");

    return 0;
}
