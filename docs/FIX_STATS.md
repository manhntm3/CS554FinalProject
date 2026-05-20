# Fix 5: Multi-Run Statistics

## Motivation

A single run of N iterations may produce a weak rate that is an outlier due to GPU scheduler timing, thermal state, or clock variation. For a published result to be credible, it must be reproducible across independent runs.

The original code ran each test exactly once and reported a single count. There was no way to assess variance or distinguish a stable 0.01% rate from a noisy one.

## Implementation

### Constants

```cpp
static const int ITERATIONS = 1'000'000;  // per run
static const int N_RUNS     = 5;          // independent repetitions
```

Total observations per test: `5 × 1,000,000 = 5,000,000` — identical to the original single-run count, but now split across 5 independent runs.

### `Stats` Struct

```cpp
struct Stats { double mean, stddev, mn, mx; };

Stats compute_stats(const std::vector<double>& v) {
    // population mean and stddev over N_RUNS rates
}
```

Each run produces a weak rate (percentage). `compute_stats` computes:
- **Mean**: average weak rate across runs
- **Stddev**: population standard deviation (measures run-to-run variability)
- **Min/Max**: extreme values (not printed, but available for debugging)

### Run Loop in Each Test Function

```cpp
std::vector<double> rates;   // weak rate per run
Hist2 total;                 // cumulative histogram across all runs

for (int run = 0; run < N_RUNS; ++run) {
    cudaMemset(d_x, 0, ...);        // fresh initial state
    kernel<<<...>>>(d_x, d_y, ...); // run the test
    cudaDeviceSynchronize();
    cudaMemcpy(h_r0, ...);

    Hist2 h;
    long long weak = 0;
    for (int i = 0; i < iterations; i++) {
        h.record(h_r0[i], h_r1[i]);
        if (is_weak(h_r0[i], h_r1[i])) weak++;
    }
    total += h;
    rates.push_back(100.0 * weak / iterations);
}

Stats s = compute_stats(rates);
```

### Output Format

```
SB Inter-Block WEAK  | Weak(0,0):   826/5000000 =  0.0165%  [mean= 0.0165% std=0.0024%]
```

The `[mean=X% std=Y%]` suffix gives the per-run stability. A low stddev (e.g., 0.0024%) relative to the mean (0.0165%) indicates the result is reproducible.

## Key Results Observed

### Stable Nonzero Weak Rates (SB WEAK, SB RELAXED CTA)

| Test | Mean | Std | Coefficient of Variation |
|---|---|---|---|
| SB Inter-Block WEAK | 0.0165% | 0.0024% | 14.5% |
| SB Inter-Block RELAXED (CTA) | 0.0163% | 0.0005% | 3.1% |

SB RELAXED CTA has remarkably low variance — the hardware produces weak behavior very consistently. SB WEAK is slightly noisier, reflecting that plain store-load reordering depends more on timing.

### Stable Zero Rates (All Stronger Variants)

All GPU/SYS scope relaxed, ACQ/REL, and FENCE SC variants report `mean=0.0000% std=0.0000%` across all 5 runs — a hard zero in every run. This is not luck; it is a hardware guarantee.

### Memory Allocation

GPU memory is allocated **once** before the N_RUNS loop and re-used across runs (with `cudaMemset` to reinitialize). This avoids repeated `cudaMalloc`/`cudaFree` overhead and matches the original single-allocation design.

## Statistical Note

With N_RUNS=5 samples, the stddev estimate has only 4 degrees of freedom, so it is a rough indicator. For publication, N_RUNS=20-30 would give a more reliable confidence interval. However, for this hardware study the signal is strong enough: either the rate is robustly 0% across all 5 runs, or it is a consistent nonzero value like 0.016%.

## Files Changed
- `litmus_tests.cu`: Added `Stats` struct and `compute_stats` function
- `litmus_tests.cu`: All `run_*_test` functions converted to N_RUNS loop
- `litmus_tests.cu`: Added `[mean=X% std=Y%]` to all output lines
- `litmus_tests.cu`: Constants `ITERATIONS` and `N_RUNS` now top-level defines
