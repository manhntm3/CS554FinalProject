# cuda architecture 89 - Nvidia ada 4000
CUDA_ARCH 	?= sm_89
NVCC 		?= nvcc
PTX_VERSION ?= 8.9

NVCC_FLAGS  := -arch=$(CUDA_ARCH) \
			   -std=c++20 \
			   -O2 \
			   --ptxas-options=-v \
			   -Xptxas -O0 \
			   -lineinfo \
			   -Xcompiler \
			   -fno-strict-aliasing
# -Xptxas -O0 keeps PTXAS from reordering ptx inst

DEBUG_FLAGS := -keep \
			   --ptxas-options=-v \
			   -Xptxas=-v

all: litmus_tests

litmus_tests: litmus_tests.cu
	$(NVCC) $(NVCC_FLAGS) $(DEBUG_FLAGS) -o litmus_tests litmus_tests.cu

debug: litmus_tests.cu
	$(NVCC) $(NVCC_FLAGS) $(DEBUG_FLAGS) -o litmus_tests_debug litmus_tests.cu

sass:
	cuobjdump --dump-sass ./litmus_tests | sed -n '1,200p'

ptx:
	cuobjdump --dump-ptx ./litmus_tests | sed -n '1,200p'

clean:
	rm -f litmus_tests litmus_tests_debug *.ptx *.cubin *.fatbin *.fatbin.c *cudafe1.c *cudafe1.cpp *.o *.ii *.gpu *.reg.c *.stub.c