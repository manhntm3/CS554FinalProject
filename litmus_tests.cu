#include <iostream>
#include <map>

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

#define ITERATIONS 1000000
#define CACHE_LINE_SIZE 128

struct alignas(CACHE_LINE_SIZE) AlignedInt {
    int value;
};

void run_sb_test() {
    AlignedInt *d_x, *d_y, *d_r0, *d_r1;
    int *h_r0, *h_r1;
    
    // Allocate device memory
    cudaMalloc(&d_x, sizeof(AlignedInt));
    cudaMalloc(&d_y, sizeof(AlignedInt));
    cudaMalloc(&d_r0, ITERATIONS * sizeof(int));
    cudaMalloc(&d_r1, ITERATIONS * sizeof(int));
    
    // Allocate host memory
    h_r0 = new int[ITERATIONS];
    h_r1 = new int[ITERATIONS];
    
    // Initialize
    cudaMemset(d_x, 0, sizeof(AlignedInt));
    cudaMemset(d_y, 0, sizeof(AlignedInt));
    
    // Launch kernel with 2 threads
    sb_litmus_test<<<1, 2>>>(&d_x->value, &d_y->value, &d_r0->value, &d_r1->value, ITERATIONS);
    
    // Copy results back
    cudaMemcpy(h_r0, d_r0, ITERATIONS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r1, d_r1, ITERATIONS * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Analyze results
    std::map<std::pair<int, int>, int> outcomes;
    for (int i = 0; i < ITERATIONS; i++) {
        outcomes[{h_r0[i], h_r1[i]}]++;
    }
    
    std::cout << "Store Buffer (SB) Test Results:\n";
    for (auto& [outcome, count] : outcomes) {
        std::cout << "r0=" << outcome.first << ", r1=" << outcome.second 
                  << " : " << count << " times (" 
                  << (100.0 * count / ITERATIONS) << "%)\n";
    }
    
    // Check for weak behavior (r0=0, r1=0)
    if (outcomes[{0, 0}] > 0) {
        std::cout << "*** WEAK BEHAVIOR OBSERVED: r0=0, r1=0 ***\n";
    }
    
    // Cleanup
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_r0); cudaFree(d_r1);
    delete[] h_r0; delete[] h_r1;
}

int main() {
    std::cout << "PTX Memory Model Litmus Tests\n";
    std::cout << "==============================\n\n";
    
    run_sb_test();    
    return 0;
}
