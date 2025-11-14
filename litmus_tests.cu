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

/*
Warp-divergence could cause kernel to behave SC like. 
the reason is the if (tid==0) part, which will activate only one lane when execute, causing divergence
*/
__global__ void sb_litmus_test(int *x, int *y, int *r0, int *r1, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < iterations; i++)
    {
        if (tid == 0)
        {
            // Thread 0: x = 1; r0 = y
            asm volatile("st.global.relaxed.sys.u32 [%0], 1;" ::"l"(x) : "memory");
            // asm volatile("fence.sc.sys;");  // Optional: test with or without fence
            asm volatile("ld.global.relaxed.sys.u32 %0, [%1];" : "=r"(r0[i]) : "l"(y) : "memory");
        }
        else if (tid == 1)
        {
            // Thread 1: y = 1; r1 = x
            asm volatile("st.global.relaxed.sys.u32 [%0], 1;" ::"l"(y) : "memory");
            // asm volatile("fence.sc.sys;");  // Optional: test with or without fence
            asm volatile("ld.global.relaxed.sys.u32 %0, [%1];" : "=r"(r1[i]) : "l"(x) : "memory");
        }
        __syncthreads();

        // Reset for next iteration
        if (tid == 0)
        {
            *x = 0;
            *y = 0;
        }
        __syncthreads();
    }
}

enum WeakOrder : int
{
    RELAXED_SYS = 0,
    RELAXED_GPU = 1,
    FENCE_SC_SYS = 2,
    REL_ACQ_GPU = 3,
};

__global__ void sb_litmus_test_param(int *x, int *y,
                                     int *r0, int *r1,
                                     int iterations,
                                     int variant)
{
    int tid = threadIdx.x;
    if (tid >= 2) return;  

    int *my_store = (tid == 0) ? x : y;
    int *my_load  = (tid == 0) ? y : x;
    int *my_out   = (tid == 0) ? r0 : r1;

    for (int i = 0; i < iterations; ++i)
    {
        unsigned tmp = 0;

        if (variant == RELAXED_SYS)
        {
            asm volatile(
                "st.global.relaxed.sys.u32 [%0], %1;"
                :: "l"(my_store), "r"(1) : "memory");
            asm volatile(
                "ld.global.relaxed.sys.u32 %0, [%1];"
                : "=r"(tmp) : "l"(my_load) : "memory");
        }
        else if (variant == RELAXED_GPU)
        {
            asm volatile(
                "st.global.relaxed.gpu.u32 [%0], %1;"
                :: "l"(my_store), "r"(1) : "memory");
            asm volatile(
                "ld.global.relaxed.gpu.u32 %0, [%1];"
                : "=r"(tmp) : "l"(my_load) : "memory");
        }
        else if (variant == FENCE_SC_SYS)
        {
            asm volatile(
                "st.global.relaxed.sys.u32 [%0], %1;"
                :: "l"(my_store), "r"(1) : "memory");
            asm volatile("fence.sc.sys;" ::: "memory");
            asm volatile(
                "ld.global.relaxed.sys.u32 %0, [%1];"
                : "=r"(tmp) : "l"(my_load) : "memory");
        }
        else if (variant == REL_ACQ_GPU)
        {
            asm volatile(
                "st.global.release.gpu.u32 [%0], %1;"
                :: "l"(my_store), "r"(1) : "memory");
            asm volatile(
                "ld.global.acquire.gpu.u32 %0, [%1];"
                : "=r"(tmp) : "l"(my_load) : "memory");
        }

        my_out[i] = (int)tmp;

        __syncthreads();

        if (tid == 0)
        {
            *x = 0;
            *y = 0;
        }

        __syncthreads();
    }
}

#define ITERATIONS 5000000
#define CACHE_LINE_SIZE 128

struct alignas(CACHE_LINE_SIZE) AlignedInt
{
    int value;
};

void run_sb_test()
{
    AlignedInt *d_x, *d_y;
    int *d_r0, *d_r1;
    int *h_r0, *h_r1;

    CUDA_CHECK(cudaMalloc(&d_x, sizeof(AlignedInt)));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(AlignedInt)));
    CUDA_CHECK(cudaMalloc(&d_r0, ITERATIONS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r1, ITERATIONS * sizeof(int)));

    h_r0 = new int[ITERATIONS];
    h_r1 = new int[ITERATIONS];

    CUDA_CHECK(cudaMemset(d_x, 0, sizeof(AlignedInt)));
    CUDA_CHECK(cudaMemset(d_y, 0, sizeof(AlignedInt)));

    // Launch kernel with 2 threads
    // sb_litmus_test<<<1, 2>>>(&d_x->value, &d_y->value, d_r0, d_r1, ITERATIONS);
    sb_litmus_test_param<<<1, 2>>>(&d_x->value, &d_y->value, d_r0, d_r1, ITERATIONS, FENCE_SC_SYS);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_r0, d_r0, ITERATIONS * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_r1, d_r1, ITERATIONS * sizeof(int), cudaMemcpyDeviceToHost));

    std::map<std::pair<int, int>, int> outcomes;
    for (int i = 0; i < ITERATIONS; i++)
    {
        outcomes[{h_r0[i], h_r1[i]}]++;
    }

    std::cout << "Store Buffer (SB) Test Results:\n";
    for (auto &[outcome, count] : outcomes)
    {
        std::cout << "r0=" << outcome.first << ", r1=" << outcome.second
                  << " : " << count << " times ("
                  << (100.0 * count / ITERATIONS) << "%)\n";
    }

    // Check for weak behavior (r0=0, r1=0)
    if (outcomes[{0, 0}] > 0)
    {
        std::cout << "*** WEAK BEHAVIOR OBSERVED: r0=0, r1=0 ***\n";
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_r0);
    cudaFree(d_r1);
    delete[] h_r0;
    delete[] h_r1;
}

int main()
{
    std::cout << "PTX Memory Model Litmus Tests\n";
    std::cout << "=============================\n";

    run_sb_test();
    return 0;
}
