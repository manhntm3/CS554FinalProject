/*
 * litmus_tests.metal  —  Metal Shading Language port of the CUDA litmus suite.
 *
 * Mirrors the four kernels in ../litmus_tests.cu:
 *   SB   (Store Buffering)
 *   MP   (Message Passing)
 *   LB   (Load Buffering)
 *   IRIW (Independent Reads of Independent Writes)
 *
 * Coverage in this file:
 *   - WEAK            : volatile non-atomic device accesses
 *   - RELAXED_GPU     : device-scope atomics, memory_order_relaxed
 *   - REL_ACQ_GPU     : release store / acquire load
 *   - FENCE_SC_GPU    : relaxed ops + device-scope seq_cst fence
 *   - ATOMIC_RELAXED  : atomic_exchange_explicit (SB only)
 *
 * Variants NOT supported on Metal (documented in README):
 *   - *_SYS  : Metal has no system-scope kernel atomics
 *   - *_CTA  : would require threadgroup-resident atomics with iteration-
 *              level barriers; the array-stride trick doesn't fit in 32KB.
 *
 * Same-buffer aliasing note:
 *   For WEAK variants we view a `device atomic_int*` buffer through a
 *   `device volatile int*` pointer. Layout-compatible on Apple GPUs.
 *   `volatile` keeps the compiler from coalescing/dropping the racy ops.
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ============================================================
// Variant enums (must match host)
// ============================================================
constant int SB_WEAK            = 0;
constant int SB_RELAXED_GPU     = 1;
constant int SB_REL_ACQ_GPU     = 2;
constant int SB_FENCE_SC_GPU    = 3;
constant int SB_ATOMIC_RELAXED  = 4;

constant int MP_WEAK            = 0;
constant int MP_RELAXED_GPU     = 1;
constant int MP_REL_ACQ_GPU     = 2;
constant int MP_FENCE_SC_GPU    = 3;

constant int LB_WEAK            = 0;
constant int LB_RELAXED_GPU     = 1;
constant int LB_ACQ_REL_GPU     = 2;
constant int LB_FENCE_SC_GPU    = 3;

constant int IRIW_WEAK          = 0;
constant int IRIW_RELAXED_GPU   = 1;
constant int IRIW_REL_ACQ_GPU   = 2;
constant int IRIW_FENCE_SC_GPU  = 3;

// ============================================================
// Primitive wrappers — the analog of ptx_* helpers in litmus_tests.cu
// ============================================================

// --- WEAK (volatile non-atomic) ---
inline void msl_st_weak(device volatile int* addr, int val) {
    *addr = val;
}
inline int msl_ld_weak(device volatile int* addr) {
    return *addr;
}

// --- RELAXED (device-scope atomics) ---
inline void msl_st_relaxed_dev(device atomic_int* addr, int val) {
    atomic_store_explicit(addr, val, memory_order_relaxed);
}
inline int msl_ld_relaxed_dev(device atomic_int* addr) {
    return atomic_load_explicit(addr, memory_order_relaxed);
}

// --- RELEASE / ACQUIRE (device-scope) ---
inline void msl_st_release_dev(device atomic_int* addr, int val) {
    atomic_store_explicit(addr, val, memory_order_release);
}
inline int msl_ld_acquire_dev(device atomic_int* addr) {
    return atomic_load_explicit(addr, memory_order_acquire);
}

// --- ATOMIC EXCHANGE (relaxed) ---
inline void msl_atom_exch_relaxed_dev(device atomic_int* addr, int val) {
    (void)atomic_exchange_explicit(addr, val, memory_order_relaxed);
}

// --- SEQ_CST FENCE (device scope) ---
// Maps to whatever MEMBAR-equivalent the Apple GPU lowers it to.
inline void msl_fence_sc_device() {
    atomic_thread_fence(memory_order_seq_cst, memory_scope_device);
}

// ============================================================
// Thread role assignment — analog of get_roles() in litmus_tests.cu
// ============================================================
// inter_block != 0: block 0 / tid 0 = P0, block 1 / tid 0 = P1
// inter_block == 0: block 0 / tid 0 = P0, block 0 / tid 32 = P1
//                   (Apple GPU SIMD-group width is 32, mirroring CUDA's
//                    warp boundary trick — tids 0 and 32 are in separate
//                    SIMD-groups so they don't run in lockstep.)
inline void get_roles(int inter_block, uint tid, uint bid,
                      thread bool& is_p0, thread bool& is_p1)
{
    is_p0 = false;
    is_p1 = false;
    if (inter_block != 0) {
        if (tid == 0u && bid == 0u) is_p0 = true;
        if (tid == 0u && bid == 1u) is_p1 = true;
    } else {
        if (bid == 0u) {
            if (tid == 0u)  is_p0 = true;
            if (tid == 32u) is_p1 = true;
        }
    }
}

// ============================================================
// SB — Store Buffering
//   P0: st(x,1); r0 = ld(y);
//   P1: st(y,1); r1 = ld(x);
// Weak outcome: r0 == 0 && r1 == 0
// ============================================================
kernel void sb_kernel(
    device atomic_int* arr_x_a   [[buffer(0)]],
    device atomic_int* arr_y_a   [[buffer(1)]],
    device int*        result_0  [[buffer(2)]],
    device int*        result_1  [[buffer(3)]],
    constant int& iterations     [[buffer(4)]],
    constant int& variant        [[buffer(5)]],
    constant int& inter_block    [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    device volatile int* arr_x = (device volatile int*)arr_x_a;
    device volatile int* arr_y = (device volatile int*)arr_y_a;

    bool is_p0, is_p1;
    get_roles(inter_block, tid, bid, is_p0, is_p1);
    if (!is_p0 && !is_p1) return;

    for (int i = 0; i < iterations; ++i) {
        int r0 = -1, r1 = -1;

        if (is_p0) {
            switch (variant) {
              case SB_WEAK:
                msl_st_weak(&arr_x[i], 1);
                r0 = msl_ld_weak(&arr_y[i]);
                break;
              case SB_RELAXED_GPU:
                msl_st_relaxed_dev(&arr_x_a[i], 1);
                r0 = msl_ld_relaxed_dev(&arr_y_a[i]);
                break;
              case SB_REL_ACQ_GPU:
                // Release/acquire orders Store-Store and Load-Load but NOT
                // Store-Load. SB should still fail under this variant.
                msl_st_release_dev(&arr_x_a[i], 1);
                r0 = msl_ld_acquire_dev(&arr_y_a[i]);
                break;
              case SB_FENCE_SC_GPU:
                msl_st_relaxed_dev(&arr_x_a[i], 1);
                msl_fence_sc_device();
                r0 = msl_ld_relaxed_dev(&arr_y_a[i]);
                break;
              case SB_ATOMIC_RELAXED:
                msl_atom_exch_relaxed_dev(&arr_x_a[i], 1);
                r0 = msl_ld_relaxed_dev(&arr_y_a[i]);
                break;
            }
            result_0[i] = r0;
        }

        if (is_p1) {
            switch (variant) {
              case SB_WEAK:
                msl_st_weak(&arr_y[i], 1);
                r1 = msl_ld_weak(&arr_x[i]);
                break;
              case SB_RELAXED_GPU:
                msl_st_relaxed_dev(&arr_y_a[i], 1);
                r1 = msl_ld_relaxed_dev(&arr_x_a[i]);
                break;
              case SB_REL_ACQ_GPU:
                msl_st_release_dev(&arr_y_a[i], 1);
                r1 = msl_ld_acquire_dev(&arr_x_a[i]);
                break;
              case SB_FENCE_SC_GPU:
                msl_st_relaxed_dev(&arr_y_a[i], 1);
                msl_fence_sc_device();
                r1 = msl_ld_relaxed_dev(&arr_x_a[i]);
                break;
              case SB_ATOMIC_RELAXED:
                msl_atom_exch_relaxed_dev(&arr_y_a[i], 1);
                r1 = msl_ld_relaxed_dev(&arr_x_a[i]);
                break;
            }
            result_1[i] = r1;
        }
    }
}

// ============================================================
// MP — Message Passing
//   x is data, y is flag.
//   P0 (producer):  data = 1;  flag = 1;
//   P1 (consumer):  r_flag = ld(flag);  r_data = ld(data);
// Weak outcome: r_flag == 1 && r_data == 0
// ============================================================
kernel void mp_kernel(
    device atomic_int* arr_data_a   [[buffer(0)]],
    device atomic_int* arr_flag_a   [[buffer(1)]],
    device int*        result_flag  [[buffer(2)]],
    device int*        result_data  [[buffer(3)]],
    constant int& iterations        [[buffer(4)]],
    constant int& variant           [[buffer(5)]],
    constant int& inter_block       [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    device volatile int* arr_data = (device volatile int*)arr_data_a;
    device volatile int* arr_flag = (device volatile int*)arr_flag_a;

    bool is_p0, is_p1;
    get_roles(inter_block, tid, bid, is_p0, is_p1);
    if (!is_p0 && !is_p1) return;

    for (int i = 0; i < iterations; ++i) {
        int r_flag = -1, r_data = -1;

        if (is_p0) {
            switch (variant) {
              case MP_WEAK:
                msl_st_weak(&arr_data[i], 1);
                msl_st_weak(&arr_flag[i], 1);
                break;
              case MP_RELAXED_GPU:
                msl_st_relaxed_dev(&arr_data_a[i], 1);
                msl_st_relaxed_dev(&arr_flag_a[i], 1);
                break;
              case MP_REL_ACQ_GPU:
                // Release on flag publishes the preceding data write.
                msl_st_relaxed_dev(&arr_data_a[i], 1);
                msl_st_release_dev(&arr_flag_a[i], 1);
                break;
              case MP_FENCE_SC_GPU:
                msl_st_relaxed_dev(&arr_data_a[i], 1);
                msl_fence_sc_device();
                msl_st_relaxed_dev(&arr_flag_a[i], 1);
                break;
            }
        }

        if (is_p1) {
            switch (variant) {
              case MP_WEAK:
                r_flag = msl_ld_weak(&arr_flag[i]);
                r_data = msl_ld_weak(&arr_data[i]);
                break;
              case MP_RELAXED_GPU:
                r_flag = msl_ld_relaxed_dev(&arr_flag_a[i]);
                r_data = msl_ld_relaxed_dev(&arr_data_a[i]);
                break;
              case MP_REL_ACQ_GPU:
                r_flag = msl_ld_acquire_dev(&arr_flag_a[i]);
                r_data = msl_ld_relaxed_dev(&arr_data_a[i]);
                break;
              case MP_FENCE_SC_GPU:
                r_flag = msl_ld_relaxed_dev(&arr_flag_a[i]);
                msl_fence_sc_device();
                r_data = msl_ld_relaxed_dev(&arr_data_a[i]);
                break;
            }
            result_flag[i] = r_flag;
            result_data[i] = r_data;
        }
    }
}

// ============================================================
// LB — Load Buffering
//   P0: r0 = ld(y); st(x, 1);
//   P1: r1 = ld(x); st(y, 1);
// Weak outcome: r0 == 1 && r1 == 1   (causality cycle)
// ============================================================
kernel void lb_kernel(
    device atomic_int* arr_x_a   [[buffer(0)]],
    device atomic_int* arr_y_a   [[buffer(1)]],
    device int*        result_0  [[buffer(2)]],
    device int*        result_1  [[buffer(3)]],
    constant int& iterations     [[buffer(4)]],
    constant int& variant        [[buffer(5)]],
    constant int& inter_block    [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    device volatile int* arr_x = (device volatile int*)arr_x_a;
    device volatile int* arr_y = (device volatile int*)arr_y_a;

    bool is_p0, is_p1;
    get_roles(inter_block, tid, bid, is_p0, is_p1);
    if (!is_p0 && !is_p1) return;

    for (int i = 0; i < iterations; ++i) {
        int r0 = -1, r1 = -1;

        if (is_p0) {
            switch (variant) {
              case LB_WEAK:
                r0 = msl_ld_weak(&arr_y[i]);
                msl_st_weak(&arr_x[i], 1);
                break;
              case LB_RELAXED_GPU:
                r0 = msl_ld_relaxed_dev(&arr_y_a[i]);
                msl_st_relaxed_dev(&arr_x_a[i], 1);
                break;
              case LB_ACQ_REL_GPU:
                r0 = msl_ld_acquire_dev(&arr_y_a[i]);
                msl_st_release_dev(&arr_x_a[i], 1);
                break;
              case LB_FENCE_SC_GPU:
                r0 = msl_ld_relaxed_dev(&arr_y_a[i]);
                msl_fence_sc_device();
                msl_st_relaxed_dev(&arr_x_a[i], 1);
                break;
            }
            result_0[i] = r0;
        }

        if (is_p1) {
            switch (variant) {
              case LB_WEAK:
                r1 = msl_ld_weak(&arr_x[i]);
                msl_st_weak(&arr_y[i], 1);
                break;
              case LB_RELAXED_GPU:
                r1 = msl_ld_relaxed_dev(&arr_x_a[i]);
                msl_st_relaxed_dev(&arr_y_a[i], 1);
                break;
              case LB_ACQ_REL_GPU:
                r1 = msl_ld_acquire_dev(&arr_x_a[i]);
                msl_st_release_dev(&arr_y_a[i], 1);
                break;
              case LB_FENCE_SC_GPU:
                r1 = msl_ld_relaxed_dev(&arr_x_a[i]);
                msl_fence_sc_device();
                msl_st_relaxed_dev(&arr_y_a[i], 1);
                break;
            }
            result_1[i] = r1;
        }
    }
}

// ============================================================
// IRIW — Independent Reads of Independent Writes
//   P0 (block 0): x = 1
//   P1 (block 1): y = 1
//   P2 (block 2): r0 = ld(x); r1 = ld(y)
//   P3 (block 3): r2 = ld(y); r3 = ld(x)
// Weak outcome: r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0
// ============================================================
kernel void iriw_kernel(
    device atomic_int* arr_x_a   [[buffer(0)]],
    device atomic_int* arr_y_a   [[buffer(1)]],
    device int*        res_r0    [[buffer(2)]],
    device int*        res_r1    [[buffer(3)]],
    device int*        res_r2    [[buffer(4)]],
    device int*        res_r3    [[buffer(5)]],
    constant int& iterations     [[buffer(6)]],
    constant int& variant        [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    device volatile int* arr_x = (device volatile int*)arr_x_a;
    device volatile int* arr_y = (device volatile int*)arr_y_a;

    bool is_p0 = (bid == 0u && tid == 0u); // writes x
    bool is_p1 = (bid == 1u && tid == 0u); // writes y
    bool is_p2 = (bid == 2u && tid == 0u); // reads x then y
    bool is_p3 = (bid == 3u && tid == 0u); // reads y then x
    if (!is_p0 && !is_p1 && !is_p2 && !is_p3) return;

    for (int i = 0; i < iterations; ++i) {
        if (is_p0) {
            switch (variant) {
              case IRIW_WEAK:         msl_st_weak(&arr_x[i], 1); break;
              case IRIW_RELAXED_GPU:  msl_st_relaxed_dev(&arr_x_a[i], 1); break;
              case IRIW_REL_ACQ_GPU:  msl_st_release_dev(&arr_x_a[i], 1); break;
              case IRIW_FENCE_SC_GPU:
                msl_st_relaxed_dev(&arr_x_a[i], 1);
                msl_fence_sc_device();
                break;
            }
        }
        if (is_p1) {
            switch (variant) {
              case IRIW_WEAK:         msl_st_weak(&arr_y[i], 1); break;
              case IRIW_RELAXED_GPU:  msl_st_relaxed_dev(&arr_y_a[i], 1); break;
              case IRIW_REL_ACQ_GPU:  msl_st_release_dev(&arr_y_a[i], 1); break;
              case IRIW_FENCE_SC_GPU:
                msl_st_relaxed_dev(&arr_y_a[i], 1);
                msl_fence_sc_device();
                break;
            }
        }
        if (is_p2) {
            int r0 = -1, r1 = -1;
            switch (variant) {
              case IRIW_WEAK:
                r0 = msl_ld_weak(&arr_x[i]);
                r1 = msl_ld_weak(&arr_y[i]);
                break;
              case IRIW_RELAXED_GPU:
                r0 = msl_ld_relaxed_dev(&arr_x_a[i]);
                r1 = msl_ld_relaxed_dev(&arr_y_a[i]);
                break;
              case IRIW_REL_ACQ_GPU:
                r0 = msl_ld_acquire_dev(&arr_x_a[i]);
                r1 = msl_ld_relaxed_dev(&arr_y_a[i]);
                break;
              case IRIW_FENCE_SC_GPU:
                r0 = msl_ld_relaxed_dev(&arr_x_a[i]);
                msl_fence_sc_device();
                r1 = msl_ld_relaxed_dev(&arr_y_a[i]);
                break;
            }
            res_r0[i] = r0;
            res_r1[i] = r1;
        }
        if (is_p3) {
            int r2 = -1, r3 = -1;
            switch (variant) {
              case IRIW_WEAK:
                r2 = msl_ld_weak(&arr_y[i]);
                r3 = msl_ld_weak(&arr_x[i]);
                break;
              case IRIW_RELAXED_GPU:
                r2 = msl_ld_relaxed_dev(&arr_y_a[i]);
                r3 = msl_ld_relaxed_dev(&arr_x_a[i]);
                break;
              case IRIW_REL_ACQ_GPU:
                r2 = msl_ld_acquire_dev(&arr_y_a[i]);
                r3 = msl_ld_relaxed_dev(&arr_x_a[i]);
                break;
              case IRIW_FENCE_SC_GPU:
                r2 = msl_ld_relaxed_dev(&arr_y_a[i]);
                msl_fence_sc_device();
                r3 = msl_ld_relaxed_dev(&arr_x_a[i]);
                break;
            }
            res_r2[i] = r2;
            res_r3[i] = r3;
        }
    }
}
