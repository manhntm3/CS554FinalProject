# Metal litmus tests — Apple Silicon (tested on M1 Pro / macOS 13+)
#
# Requirements:
#   - Full Xcode OR the standalone "Metal Developer Tools for macOS"
#     (Command Line Tools alone DO NOT ship `xcrun metal`).
#   - metal-cpp headers in ./metal-cpp/ (already vendored)
#
# Build pipeline:
#   1. .metal  -> .air        (metal compiler)
#   2. .air    -> .metallib   (metallib packager)
#   3. .cpp    -> binary      (clang++, links Metal/Foundation/QuartzCore)
# The binary loads litmus_tests.metallib from CWD at startup.

METAL    ?= xcrun -sdk macosx metal
METALLIB ?= xcrun -sdk macosx metallib
CXX      ?= clang++

# -O2 on host. metal toolchain is conservative on its own; we audit AIR
# separately via the `air-dump` target since there is no PTX-style
# -O0 knob for the device compiler.
METAL_FLAGS := -std=metal3.0 -O2 -gline-tables-only
CXX_FLAGS   := -std=c++20 -O2 -Wall -Wno-unused-private-field \
               -I./metal-cpp \
               -framework Metal -framework Foundation -framework QuartzCore

BIN := litmus_tests
LIB := litmus_tests.metallib

.PHONY: all run clean air-dump

all: $(BIN) $(LIB)

# --- Device-side ----------------------------------------------------
litmus_tests.air: litmus_tests.metal
	$(METAL) $(METAL_FLAGS) -c $< -o $@

$(LIB): litmus_tests.air
	$(METALLIB) $< -o $@

# --- Host-side ------------------------------------------------------
$(BIN): litmus_tests.cpp
	$(CXX) $(CXX_FLAGS) $< -o $@

# --- Convenience ----------------------------------------------------
run: all
	./$(BIN)

# Dump LLVM-IR-like AIR text for hand audit of compiler reordering.
# Equivalent in spirit to `cuobjdump --dump-ptx` for the CUDA project.
air-dump: litmus_tests.metal
	$(METAL) $(METAL_FLAGS) -S $< -o litmus_tests.air.txt
	@echo "Wrote litmus_tests.air.txt"

clean:
	rm -f $(BIN) $(LIB) *.air *.air.txt
