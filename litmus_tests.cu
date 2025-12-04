#include <iostream>
#include <map>
#include <iomanip>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define CUDA_CHECK(expr)                                                  \
    do                                                                    \
    {                                                                     \
        cudaError_t _err = (expr);                                        \
        if (_err != cudaSuccess)                                          \
        {                                                                 \
            fprintf(stderr, "CUDA error %s @ %s:%d -> %s\n",              \
                    #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

enum TestSBType : int
{
    TEST_SB_WEAK            = 0,
    TEST_SB_RELAXED_CTA     = 1,
    TEST_SB_RELAXED_GPU     = 2,
    TEST_SB_RELAXED_SYS     = 3,
    TEST_SB_FENCE_SC_GPU    = 4,
    TEST_SB_FENCE_SC_SYS    = 5,
    TEST_SB_REL_ACQ_GPU     = 6,
    TEST_SB_REL_ACQ_SYS     = 7,
    TEST_SB_ATOMIC_RELAXED  = 8,
};

enum TestMPType : int
{
    // --- Message Passing (MP) variants ---
    // Pattern:
    //   Initially: data = 0, flag = 0
    //   P0 (producer):  data = 1;  flag = 1;   (with chosen semantics)
    //   P1 (consumer):  r_flag = flag;  r_data = data;
    //
    // "Bad" MP outcome (weak behaviour):  r_flag == 1 && r_data == 0
    TEST_MP_WEAK            = 0,  // plain ld/st (implicitly .weak)
    TEST_MP_RELAXED_GPU     = 1,  // relaxed.gpu on both data & flag
    TEST_MP_RELAXED_SYS     = 2,  // relaxed.sys on both data & flag
    TEST_MP_REL_ACQ_GPU     = 3,  // release store to flag, acquire load of flag (GPU scope)
    TEST_MP_REL_ACQ_SYS     = 4,  // release/acquire pair on flag (SYS scope)
    TEST_MP_FENCE_SC_GPU    = 5,  // fence.sc.gpu around flag/data
    TEST_MP_FENCE_SC_SYS    = 6,  // fence.sc.sys around flag/data
};

enum ScopeStrategy
{
    SCOPE_CTA = 0, // Threads in the same block (different warps)
    SCOPE_GPU = 1, // Threads in different blocks
    SCOPE_SYS = 2  // 
};

// --- WEAK (Default) ---
// Default standard load/store. The hardware is free to reorder
__device__ __forceinline__ void ptx_st_weak(int *addr, int val)
{
    // non-synchronize; could be mapped to non-atomic c++ store
    asm volatile("st.global.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
}
__device__ __forceinline__ int ptx_ld_weak(int *addr)
{
    int val;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    return val;
}

// --- RELAXED ---
// 
__device__ __forceinline__ void ptx_st_relaxed(int *addr, int val, int scope)
{
    if (scope == SCOPE_CTA) {
        asm volatile("st.global.relaxed.cta.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
    } else if (scope == SCOPE_GPU) {
        asm volatile("st.global.relaxed.gpu.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
    } else {
        asm volatile("st.global.relaxed.sys.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
    }
}
__device__ __forceinline__ int ptx_ld_relaxed(int *addr, int scope)
{
    int val;
    if (scope == SCOPE_CTA) {
        asm volatile("ld.global.relaxed.cta.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    } else if (scope == SCOPE_GPU) {
        asm volatile("ld.global.relaxed.gpu.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    } else {
        asm volatile("ld.global.relaxed.sys.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    }
    return val;
}

// --- RELEASE (Write) ---
// All previous writes/stores are visible before this write/store
__device__ __forceinline__ void ptx_st_release(int *addr, int val, int scope)
{
    if (scope == SCOPE_CTA) {
        asm volatile("st.global.release.cta.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
    } else if (scope == SCOPE_GPU) {
        asm volatile("st.global.release.gpu.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
    } else {
        asm volatile("st.global.release.sys.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
    }
}

// --- ACQUIRE (Read) ---
// Ensure this read happen before subsequent reads/writes
__device__ __forceinline__ int ptx_ld_acquire(int *addr, int scope)
{
    int val;
    if (scope == SCOPE_CTA) {
        asm volatile("ld.global.acquire.cta.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    } else if (scope == SCOPE_GPU) {
        asm volatile("ld.global.acquire.gpu.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    } else {
        asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    }
    return val;
}

// --- ATOMIC EXCHANGE (RELAXED) ---
__device__ __forceinline__ void ptx_atom_exch_relaxed(int *addr, int val, int scope)
{
    if (scope == SCOPE_CTA) {
        asm volatile("atom.global.relaxed.cta.exch.b32 %0, [%1], %2;" : "=r"(val) : "l"(addr), "r"(val) : "memory");
    } else if (scope == SCOPE_GPU) {
        asm volatile("atom.global.relaxed.gpu.exch.b32 %0, [%1], %2;" : "=r"(val) : "l"(addr), "r"(val) : "memory");
    } else {
        asm volatile("atom.global.relaxed.sys.exch.b32 %0, [%1], %2;" : "=r"(val) : "l"(addr), "r"(val) : "memory");
    }
}

// --- FENCE ---
__device__ __forceinline__ void ptx_fence_sc_gpu() {
    asm volatile("fence.sc.gpu;" ::: "memory");
}
__device__ __forceinline__ void ptx_fence_sc_sys() {
    asm volatile("fence.sc.sys;" ::: "memory");
}


__device__ void global_spin_barrier(volatile int* barrier, int val_to_wait_for) {
    // Thread 0 of participating blocks call this.
    // Simple arrival count.
    atomicAdd((int*)barrier, 1);
    while (*barrier < val_to_wait_for);
}

/*
Warp-divergence could cause kernel to behave SC like.
the reason is the if (tid==0) part, which will activate only one lane when execute, causing divergence
*/


/*
Store Buffering (SB) kernel. 
non-atomic
weak load/store

p0: x = 1; r0 = y;
p1: y = 1; r1 = x;

Weak behaviour: r0 = r1 = 0
*/
__global__ void sb_kernel_litmus_test(int *x, int *y,
                                     int *result_0, int *result_1,
                                     int *sync_barrier,
                                     int iterations,
                                     int variant,
                                     bool inter_block)
{

    // Thread Identification: 
    // Inter-block: Block 0 is P0, Block 1 is P1. Thread 0 of each block acts.
    // Intra-block: Thread 0 is P0, Thread 32 (next warp) is P1.
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    bool is_p0 = false;
    bool is_p1 = false;

    if (inter_block) {
        if (tid == 0 && bid == 0) is_p0 = true;
        if (tid == 0 && bid == 1) is_p1 = true;
    } else {
        if (bid == 0) { 
            // Single block test
            if (tid == 0) is_p0 = true;
            if (tid == 32) is_p1 = true; // Use different warp to avoid lockstep
        }
    }

    if (!is_p0 && !is_p1) return;

    for (int i = 0; i < iterations; ++i)
    {

        // Synchronization: reset x and y to 0
        if (inter_block) {
            if (is_p0) {
                // Need a strong barrier to ensure previous iteration is done
                // __threadfence(); 
                *x = 0;
                *y = 0;
                __threadfence(); 
            }
            // Simple handshake for sync (using volatile to force visibility)
            // In production, use grid barriers.
            global_spin_barrier(sync_barrier, (i * 2) + 2); 
        } else {
            __syncthreads(); // Wait for previous iter
            if (tid == 0) {
                // Using relaxed stores for setup to ensure visibility within CTA
                *x = 0; 
                *y = 0;
            }
            __syncthreads(); // Ensure Reset is visible
            __threadfence_block(); // Flush L1 write buffer
        }
        

        // NOTE: For this simple demo, we rely on the large number of iterations 
        // and probabilistic overlap. Perfect sync requires `cooperative_groups`.
        // To ensure we don't read stale data from previous iter, we do a quick busy-wait sync.
        // (Omitting complex sync code for brevity, assuming probabilistic hits).

        // --- THE ACTUAL TEST ---
        int r0 = -1;
        int r1 = -1;
        // P0
        if (is_p0) {
            switch (variant) {
                case TEST_SB_WEAK:  // Weak
                    ptx_st_weak(x, 1);
                    r0 = ptx_ld_weak(y);
                    break;
                case TEST_SB_RELAXED_CTA:
                    ptx_st_relaxed(x, 1, SCOPE_CTA); // Too weak for Inter-block
                    r0 = ptx_ld_relaxed(y, SCOPE_CTA);
                    break;
                case TEST_SB_RELAXED_GPU:
                    ptx_st_relaxed(x, 1, SCOPE_GPU);
                    r0 = ptx_ld_relaxed(y, SCOPE_GPU);
                    break;
                case TEST_SB_RELAXED_SYS:
                    ptx_st_relaxed(x, 1, SCOPE_SYS);
                    r0 = ptx_ld_relaxed(y, SCOPE_SYS);
                    break;
                case TEST_SB_FENCE_SC_GPU:
                    ptx_st_relaxed(x, 1, SCOPE_GPU);
                    ptx_fence_sc_gpu(); // CRITICAL: Store-Load Fence
                    r0 = ptx_ld_relaxed(y, SCOPE_GPU);
                    break;
                case TEST_SB_FENCE_SC_SYS:
                    ptx_st_relaxed(x, 1, SCOPE_SYS);
                    ptx_fence_sc_sys(); // Strongest fence
                    r0 = ptx_ld_relaxed(y, SCOPE_SYS);
                    break;
                case TEST_SB_REL_ACQ_GPU:
                    // Release store, Acquire load.
                    // IMPORTANT: This provides Store-Store and Load-Load ordering,
                    // but NOT Store-Load ordering. SB should still FAIL.
                    ptx_st_release(x, 1, SCOPE_GPU);
                    r0 = ptx_ld_acquire(y, SCOPE_GPU);
                    break;
                case TEST_SB_REL_ACQ_SYS:
                    // Release store, Acquire load.
                    // IMPORTANT: This provides Store-Store and Load-Load ordering,
                    // but NOT Store-Load ordering. SB should still FAIL.
                    ptx_st_release(x, 1, SCOPE_SYS);
                    r0 = ptx_ld_acquire(y, SCOPE_SYS);
                    break;
                case TEST_SB_ATOMIC_RELAXED:
                    ptx_atom_exch_relaxed(x, 1, SCOPE_GPU);
                    r0 = ptx_ld_relaxed(y, SCOPE_GPU);
                    break;
            }

            result_0[i] = r0;
        }

        // P1
        if (is_p1) {
            switch (variant) {
                case TEST_SB_WEAK:
                    ptx_st_weak(y, 1);
                    r1 = ptx_ld_weak(x);
                    break;
                case TEST_SB_RELAXED_CTA:
                    ptx_st_relaxed(y, 1, SCOPE_CTA);
                    r1 = ptx_ld_relaxed(x, SCOPE_CTA);
                    break;
                case TEST_SB_RELAXED_GPU:
                    ptx_st_relaxed(y, 1, SCOPE_GPU);
                    r1 = ptx_ld_relaxed(x, SCOPE_GPU);
                    break;
                case TEST_SB_RELAXED_SYS:
                    ptx_st_relaxed(y, 1, SCOPE_SYS);
                    r1 = ptx_ld_relaxed(x, SCOPE_SYS);
                    break;
                case TEST_SB_FENCE_SC_GPU:
                    ptx_st_relaxed(y, 1, SCOPE_GPU);
                    ptx_fence_sc_gpu();
                    r1 = ptx_ld_relaxed(x, SCOPE_GPU);
                    break;
                case TEST_SB_FENCE_SC_SYS:
                    ptx_st_relaxed(y, 1, SCOPE_SYS);
                    ptx_fence_sc_sys();
                    r1 = ptx_ld_relaxed(x, SCOPE_SYS);
                    break;
                case TEST_SB_REL_ACQ_GPU:
                    ptx_st_release(y, 1, SCOPE_GPU);
                    r1 = ptx_ld_acquire(x, SCOPE_GPU);
                    break;
                case TEST_SB_REL_ACQ_SYS:
                    ptx_st_release(y, 1, SCOPE_SYS);
                    r1 = ptx_ld_acquire(x, SCOPE_SYS);
                    break;
                case TEST_SB_ATOMIC_RELAXED:
                    ptx_atom_exch_relaxed(y, 1, SCOPE_GPU);
                    r1 = ptx_ld_relaxed(x, SCOPE_GPU);
                    break;
            }
            result_1[i] = r1;
        }

        if (inter_block) {
            // Must wait for both to finish before resetting X and Y in next loop
            global_spin_barrier(sync_barrier, (i * 2) + 3);
        }
        
        // Very basic delay to shift phases (helps mitigate lockstep)
        // for(volatile int k=0; k<100; k++); 
    }
}

void run_test(int iterations, bool inter_block, int variant, const char* label) {
    int *d_x, *d_y, *d_r0, *d_r1, *d_barrier;
    int *h_r0 = new int[iterations];
    int *h_r1 = new int[iterations];

    cudaMalloc(&d_x, sizeof(int));
    cudaMalloc(&d_y, sizeof(int));
    cudaMalloc(&d_r0, iterations * sizeof(int));
    cudaMalloc(&d_r1, iterations * sizeof(int));
    cudaMalloc(&d_barrier, sizeof(int));
    
    cudaMemset(d_x, 0, sizeof(int));
    cudaMemset(d_y, 0, sizeof(int));
    cudaMemset(d_barrier, 0, sizeof(int));

    int blocks = inter_block ? 2 : 1;
    sb_kernel_litmus_test<<<blocks, 64>>>(d_x, d_y, d_r0, d_r1, d_barrier, iterations, variant, inter_block);
    cudaDeviceSynchronize();

    cudaMemcpy(h_r0, d_r0, iterations * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r1, d_r1, iterations * sizeof(int), cudaMemcpyDeviceToHost);

    int weak = 0;
    for(int i=0; i<iterations; i++) {
        if(h_r0[i] == 0 && h_r1[i] == 0) weak++;
    }

    std::cout << std::left << std::setw(30) << label 
              << "| Weak: " << std::setw(6) << weak 
              << "(" << std::fixed << std::setprecision(2) << (100.0 * weak / iterations) << "%)\n";

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_r0); cudaFree(d_r1); cudaFree(d_barrier);
    delete[] h_r0; delete[] h_r1;
}


/*
 * Message Passing (MP) kernel.
 *
 * We reuse x and y as:
 *   x == data    (payload)
 *   y == flag    (synchronisation flag)
 *
 * Initial state each iteration:
 *   data (x) = 0
 *   flag (y) = 0
 *
 * Threads:
 *   P0 (producer): writes data then flag, using chosen semantics.
 *   P1 (consumer): reads flag then data, using chosen semantics.
 *
 * We record on the consumer side:
 *   result_0[i] = observed flag (r_flag)
 *   result_1[i] = observed data (r_data)
 *
 * "Bad" MP behaviour (what we try to provoke under weak semantics):
 *   r_flag == 1 && r_data == 0
 * i.e. consumer sees the flag set but still sees stale data.
 *
 * Layout:
 *   - inter_block = true:
 *       block 0, thread 0 -> P0 (producer)
 *       block 1, thread 0 -> P1 (consumer)
 *
 *   - inter_block = false:
 *       block 0, thread 0  -> P0
 *       block 0, thread 32 -> P1 (different warp to avoid lockstep effects)
 */
__global__ void mp_kernel_litmus_test(int *x, int *y,
                                      int *result_0, int *result_1,
                                      int *sync_barrier,
                                      int iterations,
                                      int variant,
                                      bool inter_block)
{
    // Identify which CUDA thread plays which logical role.
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    bool is_p0 = false;  // producer
    bool is_p1 = false;  // consumer

    if (inter_block) {
        // Two-block setup: one logical thread per block.
        if (tid == 0 && bid == 0) is_p0 = true;
        if (tid == 0 && bid == 1) is_p1 = true;
    } else {
        // Single-block setup: use different warps inside the same CTA.
        if (bid == 0) {
            if (tid == 0)  is_p0 = true;
            if (tid == 32) is_p1 = true;  // different warp to reduce lockstep bias
        }
    }

    if (!is_p0 && !is_p1)
        return;  // all other threads idle

    // Aliases for clarity
    int *data = x;
    int *flag = y;

    for (int i = 0; i < iterations; ++i)
    {
        // --------------------------------------------------------------------
        // 1. Reset data and flag to 0 at the start of each iteration
        // --------------------------------------------------------------------
        if (inter_block) {
            if (is_p0) {
                // Only P0 resets the shared variables.
                // __threadfence() ensures the reset is visible to P1
                // before it moves on to the MP pattern.
                *data = 0;
                *flag = 0;
                __threadfence();
            }
            // Both P0 and P1 participate in the global spin barrier.
            // Each iteration uses a larger threshold so the arrival counter
            // monotonically increases.
            global_spin_barrier(sync_barrier, (i * 2) + 2);
        } else {
            // Intra-CTA setup: block-level reset.
            __syncthreads();  // ensure previous iteration's work is done
            if (tid == 0) {
                *data = 0;
                *flag = 0;
            }
            __syncthreads();         // make reset visible inside CTA
            __threadfence_block();   // order writes within the block
        }

        // --------------------------------------------------------------------
        // 2. Message-passing pattern for this iteration
        // --------------------------------------------------------------------
        //
        // Producer (P0):
        //   - Writes data = 1
        //   - Then "publishes" flag = 1 with chosen semantics
        //
        // Consumer (P1):
        //   - First loads flag
        //   - Then loads data
        //
        // On the host, you classify "bad" MP outcomes as:
        //   result_0[i] == 1 && result_1[i] == 0
        // where result_0 is r_flag, result_1 is r_data.
        //
        int r_flag = -1;
        int r_data = -1;

        // --- Producer side ---
        if (is_p0) {
            switch (variant) {
                case TEST_MP_WEAK:
                    // Plain global stores (implicitly .weak).
                    // No ordering or synchronisation guarantees.
                    ptx_st_weak(data, 1);
                    ptx_st_weak(flag, 1);
                    break;

                case TEST_MP_RELAXED_GPU:
                    // Strong relaxed stores at GPU scope.
                    // They participate in the memory model but do NOT
                    // enforce any particular cross-thread ordering.
                    ptx_st_relaxed(data, 1, SCOPE_GPU);
                    ptx_st_relaxed(flag, 1, SCOPE_GPU);
                    break;

                case TEST_MP_RELAXED_SYS:
                    // Same as above but system scope (host/other GPUs).
                    ptx_st_relaxed(data, 1, SCOPE_SYS);
                    ptx_st_relaxed(flag, 1, SCOPE_SYS);
                    break;

                case TEST_MP_REL_ACQ_GPU:
                    // Classic message-passing with release:
                    //
                    //   data = 1          (relaxed)
                    //   flag = 1 (release.gpu)
                    //
                    // The PTX model guarantees that if a GPU-scope acquire
                    // load of 'flag' sees the 1, then all prior writes
                    // (including to 'data') become visible at GPU scope.
                    ptx_st_relaxed(data, 1, SCOPE_GPU);
                    ptx_st_release(flag, 1, SCOPE_GPU);
                    break;

                case TEST_MP_REL_ACQ_SYS:
                    // Same, but system-scope: this is what you'd use if
                    // the host or another GPU also participates.
                    ptx_st_relaxed(data, 1, SCOPE_SYS);
                    ptx_st_release(flag, 1, SCOPE_SYS);
                    break;

                case TEST_MP_FENCE_SC_GPU:
                    // Fence-based message passing (device scope):
                    //
                    //   data = 1 (relaxed.gpu)
                    //   fence.sc.gpu
                    //   flag = 1 (relaxed.gpu)
                    //
                    // A matching fence on the consumer side enforces an
                    // SC-like ordering at GPU scope, so observing flag==1
                    // should imply seeing data==1.
                    ptx_st_relaxed(data, 1, SCOPE_GPU);
                    ptx_fence_sc_gpu();
                    ptx_st_relaxed(flag, 1, SCOPE_GPU);
                    break;

                case TEST_MP_FENCE_SC_SYS:
                    // Same idea with system-scope SC fences.
                    ptx_st_relaxed(data, 1, SCOPE_SYS);
                    ptx_fence_sc_sys();
                    ptx_st_relaxed(flag, 1, SCOPE_SYS);
                    break;
            }
        }

        // --- Consumer side ---
        if (is_p1) {
            switch (variant) {
                case TEST_MP_WEAK:
                    // Plain weak loads: hardware is free to reorder or
                    // delay visibility. MP can fail: flag==1, data==0.
                    r_flag = ptx_ld_weak(flag);
                    r_data = ptx_ld_weak(data);
                    break;

                case TEST_MP_RELAXED_GPU:
                    // Strong relaxed loads at GPU scope, but no ordering.
                    // MP is still allowed to fail: the flag write can become
                    // visible before the data write.
                    r_flag = ptx_ld_relaxed(flag, SCOPE_GPU);
                    r_data = ptx_ld_relaxed(data, SCOPE_GPU);
                    break;

                case TEST_MP_RELAXED_SYS:
                    // Same semantics, wider scope.
                    r_flag = ptx_ld_relaxed(flag, SCOPE_SYS);
                    r_data = ptx_ld_relaxed(data, SCOPE_SYS);
                    break;

                case TEST_MP_REL_ACQ_GPU:
                    // Acquire on the flag, relaxed on data:
                    //
                    //   r_flag = ld.acquire.gpu(flag)
                    //   r_data = ld.relaxed.gpu(data)
                    //
                    // If r_flag == 1 and that 1 comes from the producer's
                    // release store, the PTX model guarantees that r_data
                    // must see data==1 at GPU scope.
                    r_flag = ptx_ld_acquire(flag, SCOPE_GPU);
                    r_data = ptx_ld_relaxed(data, SCOPE_GPU);
                    break;

                case TEST_MP_REL_ACQ_SYS:
                    // Same with system-scope acquire.
                    r_flag = ptx_ld_acquire(flag, SCOPE_SYS);
                    r_data = ptx_ld_relaxed(data, SCOPE_SYS);
                    break;

                case TEST_MP_FENCE_SC_GPU:
                    // Fence-based message passing on the consumer:
                    //
                    //   r_flag = ld.relaxed.gpu(flag)
                    //   fence.sc.gpu
                    //   r_data = ld.relaxed.gpu(data)
                    //
                    // Together with the producer's fence, this enforces
                    // SC at GPU scope for these operations.
                    r_flag = ptx_ld_relaxed(flag, SCOPE_GPU);
                    ptx_fence_sc_gpu();
                    r_data = ptx_ld_relaxed(data, SCOPE_GPU);
                    break;

                case TEST_MP_FENCE_SC_SYS:
                    // System-scope fence variant.
                    r_flag = ptx_ld_relaxed(flag, SCOPE_SYS);
                    ptx_fence_sc_sys();
                    r_data = ptx_ld_relaxed(data, SCOPE_SYS);
                    break;
            }

            // Store consumer observations for this iteration.
            result_0[i] = r_flag;
            result_1[i] = r_data;
        }

        // --------------------------------------------------------------------
        // 3. End-of-iteration barrier (inter-block case)
        // --------------------------------------------------------------------
        if (inter_block) {
            // Make sure both producer and consumer have finished their MP
            // operations before P0 resets data/flag in the next iteration.
            global_spin_barrier(sync_barrier, (i * 2) + 3);
        }
        // (Optional phase shifting could be added here with a small busy loop.)
    }
}

void run_mp_test(int iterations, bool inter_block, int variant, const char* label) {
    int *d_data, *d_flag, *d_rf, *d_rd, *d_barrier;
    int *h_rf = new int[iterations];
    int *h_rd = new int[iterations];

    CUDA_CHECK(cudaMalloc(&d_data, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rf,   iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rd,   iterations * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_barrier, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_data, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_barrier, 0, sizeof(int)));

    int blocks = inter_block ? 2 : 1;
    mp_kernel_litmus_test<<<blocks, 64>>>(d_data, d_flag, d_rf, d_rd,
                                          d_barrier, iterations, variant, inter_block);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_rf, d_rf, iterations * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rd, d_rd, iterations * sizeof(int), cudaMemcpyDeviceToHost));

    int weak = 0;
    for (int i = 0; i < iterations; ++i) {
        // MP weak outcome: flag == 1 but data still 0
        if (h_rf[i] == 1 && h_rd[i] == 0) weak++;
    }

    std::cout << std::left << std::setw(30) << label
              << "| Weak: " << std::setw(6) << weak
              << "(" << std::fixed << std::setprecision(4)
              << (100.0 * weak / iterations) << "%)\n";

    cudaFree(d_data); cudaFree(d_flag);
    cudaFree(d_rf);   cudaFree(d_rd);
    cudaFree(d_barrier);
    delete[] h_rf; delete[] h_rd;
}

#define ITERATIONS 5000000

int main() {
    const int N = ITERATIONS;
    // std::cout << "Store Buffer (SB) Test | Iterations: " << N << "\n";
    // std::cout << "--------------------------------------------------------\n";
    
    // 1. Weak & Relaxed (Should Fail)
    // run_test(N, true, TEST_SB_WEAK,        "Inter-Block WEAK");
    // run_test(N, true, TEST_SB_RELAXED_GPU, "Inter-Block RELAXED (GPU)");
    // run_test(N, true, TEST_SB_RELAXED_SYS, "Inter-Block RELAXED (SYS)");
    
    // // 2. Bad Scope (Should Fail Hard)
    // run_test(N, true, TEST_SB_RELAXED_CTA, "Inter-Block RELAXED (CTA)");

    // // 3. Release/Acquire (Should Fail - Rel/Acq is not sequential consistency)
    // run_test(N, true, TEST_SB_REL_ACQ_GPU, "Inter-Block ACQ/REL (GPU)");

    // // 4. Fences (Should Pass / 0%)
    // run_test(N, true, TEST_SB_FENCE_SC_GPU, "Inter-Block FENCE SC (GPU)");
    // run_test(N, true, TEST_SB_FENCE_SC_SYS, "Inter-Block FENCE SC (SYS)");


    std::cout << "Message Passing (MP) Test | Iterations: " << N << "\n";
    std::cout << "--------------------------------------------------------\n";
    
    run_mp_test(N, true, TEST_MP_WEAK,        "Inter-Block WEAK");
    run_mp_test(N, true, TEST_MP_RELAXED_GPU, "Inter-Block RELAXED (GPU)");
    run_mp_test(N, true, TEST_MP_RELAXED_SYS, "Inter-Block RELAXED (SYS)");

    run_mp_test(N, true, TEST_MP_REL_ACQ_GPU, "Inter-Block ACQ/REL (GPU)");
    run_mp_test(N, true, TEST_MP_REL_ACQ_SYS, "Inter-Block ACQ/REL (SYS)");

    run_mp_test(N, true, TEST_MP_FENCE_SC_GPU, "Inter-Block FENCE SC (GPU)");
    run_mp_test(N, true, TEST_MP_FENCE_SC_SYS, "Inter-Block FENCE SC (SYS)");

    return 0;
}