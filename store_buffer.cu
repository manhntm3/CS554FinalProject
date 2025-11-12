#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

__global__ void sb_litmus_test(int* x, int* y, int* r0, int* r1, int iterations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    for (int i = 0; i < iterations; i++) {
        if (tid == 0) {
            // Thread 0: x = 1; r0 = y
            asm volatile("st.global.relaxed.sys.u32 [%0], 1;" :: "l"(x) : "memory");
            asm volatile("fence.sc.sys;");  // Optional: test with or without fence
            asm volatile("ld.global.relaxed.sys.u32 %0, [%1];" : "=r"(r0[i]) : "l"(y) : "memory");
        } else if (tid == 1) {
            // Thread 1: y = 1; r1 = x
            asm volatile("st.global.relaxed.sys.u32 [%0], 1;" :: "l"(y) : "memory");
            asm volatile("fence.sc.sys;");  // Optional: test with or without fence
            asm volatile("ld.global.relaxed.sys.u32 %0, [%1];" : "=r"(r1[i]) : "l"(x) : "memory");
        }
        __syncthreads();
        
        // Reset for next iteration
        if (tid == 0) {
            *x = 0;
            *y = 0;
        }
        __syncthreads();
    }
}
