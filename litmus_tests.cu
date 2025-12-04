#include <iostream>
#include <map>

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

enum TestType : int
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
    TEST_LB,
    TEST_MP,
};

enum ScopeStrategy
{
    SCOPE_CTA = 0, // Threads in the same block (different warps)
    SCOPE_GPU = 1, // Threads in different blocks
    SCOPE_SYS = 2  // 
};

// --- WEAK (Default) ---
// Default standard load/store. The hardware is free to reorder
__device__ __forceinline__ void ptx_st_weak(int *addr, int val, int scope)
{
    // non-synchronize; could be mapped to non-atomic c++ store
    if (scope == SCOPE_CTA) {
        asm volatile("st.global.cta.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
    } else if (scope == SCOPE_GPU) {
        asm volatile("st.global.gpu.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
    } else {
        asm volatile("st.global.sys.u32 [%0], %1;" ::"l"(addr), "r"(val) : "memory");
    }
}
__device__ __forceinline__ int ptx_ld_weak(int *addr, int scope)
{
    int val;
    if (scope == SCOPE_CTA) {
        asm volatile("ld.global.cta.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    } else if (scope == SCOPE_GPU) {
        asm volatile("ld.global.gpu.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    } else {
        asm volatile("ld.global.sys.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
    }
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
    int old;
    if (scope == SCOPE_CTA) {
        asm volatile("atom.global.relaxed.cta.exch.u32 %0, [%1], %2;" : "=r"(old) : "l"(addr), "r"(val) : "memory");
    } else if (scope == SCOPE_GPU) {
        asm volatile("atom.global.relaxed.gpu.exch.u32 %0, [%1], %2;" : "=r"(old) : "l"(addr), "r"(val) : "memory");
    } else {
        asm volatile("atom.global.relaxed.sys.exch.u32 %0, [%1], %2;" : "=r"(old) : "l"(addr), "r"(val) : "memory");
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

    int r0 = -1; // P0's observation of Y
    int r1 = -1; // P1's observation of X

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
                    ptx_st_weak(x, 1, SCOPE_GPU);
                    r0 = ptx_ld_weak(y, SCOPE_GPU);
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
                    ptx_st_weak(y, 1, SCOPE_GPU);
                    r1 = ptx_ld_weak(x, SCOPE_GPU);
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

#define ITERATIONS 5000000
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

int main() {
    const int N = ITERATIONS;
    std::cout << "Store Buffer (SB) Test | Iterations: " << N << "\n";
    std::cout << "--------------------------------------------------------\n";
    
    // 1. Weak & Relaxed (Should Fail)
    run_test(N, true, TEST_SB_WEAK,        "Inter-Block WEAK");
    run_test(N, true, TEST_SB_RELAXED_GPU, "Inter-Block RELAXED (GPU)");
    run_test(N, true, TEST_SB_RELAXED_SYS, "Inter-Block RELAXED (SYS)");
    
    // 2. Bad Scope (Should Fail Hard)
    run_test(N, true, TEST_SB_RELAXED_CTA, "Inter-Block RELAXED (CTA)");

    // 3. Release/Acquire (Should Fail - Rel/Acq is not sequential consistency)
    run_test(N, true, TEST_SB_REL_ACQ_GPU, "Inter-Block ACQ/REL (GPU)");

    // 4. Fences (Should Pass / 0%)
    run_test(N, true, TEST_SB_FENCE_SC_GPU, "Inter-Block FENCE SC (GPU)");
    run_test(N, true, TEST_SB_FENCE_SC_SYS, "Inter-Block FENCE SC (SYS)");

    return 0;
}